"""Stage 2.5 of the latent casting funnel: causal power screen.

Stage 1 measured latents as readers (when they fire). This stage measures them
as actuators: clamp a latent at layer 12 over a natural prefix and see whether
the next-token distribution actually moves.

For each of N_CAND randomly sampled stage-1 survivors:
  - run N_PREFIX pile prefixes clean, record final-position next-token log-probs
  - re-run with the latent clamped to mult * mean_act (mult in MULTS) at every
    non-BOS prefix position
  - record mean KL(steered || clean), steered entropy (degeneracy check), and
    the top boosted tokens at the middle strength (on-topic check)

No cuts are applied here; this is a calibration pass. The script prints
KL quantiles and hit rates so we can set thresholds by looking at data.

Outputs (results/stage2_5/):
  causal_screen.npz   candidates, kl (N_CAND, len(MULTS)), ent_steered, ent_clean
  top_tokens.json     per candidate: top-10 boosted tokens at MULTS[1]
  summary.json        calibration stats

Usage (on tigerfish):
  python stage2_5_causal.py --device cuda:0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "google/gemma-2-2b"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_12/width_16k/canonical"
LAYER = 12

N_CAND = 500
N_PREFIX = 256
PREFIX_LEN = 64          # tokens after BOS
MULTS = [2.0, 5.0, 10.0]  # clamp target = mult * stage-1 mean firing activation
BATCH = 128
SEED = 0
TOP_K = 10


def get_prefixes(tokenizer, n, seq_len):
    """(n, seq_len+1) BOS-prefixed token ids from pile-uncopyrighted."""
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    bos = tokenizer.bos_token_id
    buf, rows = [], []
    for ex in ds:
        buf.extend(tokenizer(ex["text"], add_special_tokens=False)["input_ids"])
        while len(buf) >= seq_len and len(rows) < n:
            rows.append([bos] + buf[:seq_len])
            buf = buf[seq_len:]
        if len(rows) >= n:
            return torch.tensor(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out or Path(__file__).parent.parent / "results" / "stage2_5")
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    res1 = Path(__file__).parent.parent / "results" / "stage1"

    from sae_lens import SAE

    sae = SAE.from_pretrained(SAE_RELEASE, SAE_ID, device=str(dev))[0]
    sae = sae.to(torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev)
    model.eval()

    stats = np.load(res1 / "latent_stats.npz")
    survivors = np.array(json.loads((res1 / "survivors.json").read_text())["survivors"])
    rng = np.random.default_rng(SEED)
    cand = np.sort(rng.choice(survivors, size=N_CAND, replace=False))
    mean_act = stats["mean_act"]

    prefixes = get_prefixes(tokenizer, N_PREFIX, PREFIX_LEN)

    # steering hook on layer LAYER output: clamp latent j to target at non-BOS positions
    steer = {"j": None, "target": None}

    def hook(_, __, output):
        if steer["j"] is None:
            return
        is_tuple = isinstance(output, tuple)
        x = output[0] if is_tuple else output
        xf = x[:, 1:, :].float()                       # skip BOS position
        a = sae.encode(xf.reshape(-1, xf.shape[-1]))[:, steer["j"]]
        a = a.reshape(xf.shape[0], xf.shape[1])
        delta = (steer["target"] - a).unsqueeze(-1) * sae.W_dec[steer["j"]]
        x = torch.cat([x[:, :1, :], (xf + delta).to(x.dtype)], dim=1)
        return (x,) + output[1:] if is_tuple else x

    model.model.layers[LAYER].register_forward_hook(hook)

    def final_logprobs():
        """Final-position log-probs for all prefixes under the current hook state."""
        outs = []
        with torch.no_grad():
            for i in range(0, N_PREFIX, BATCH):
                ids = prefixes[i : i + BATCH].to(dev)
                logits = model(ids).logits[:, -1, :].float()
                outs.append(torch.log_softmax(logits, dim=-1))
        return torch.cat(outs)                          # (N_PREFIX, vocab)

    logp_clean = final_logprobs()
    p_clean = logp_clean.exp()
    ent_clean = float(-(p_clean * logp_clean).sum(-1).mean())

    kl = np.zeros((N_CAND, len(MULTS)))
    ent = np.zeros((N_CAND, len(MULTS)))
    top_tokens = {}

    for ci, j in enumerate(cand):
        steer["j"] = int(j)
        for mi, mult in enumerate(MULTS):
            steer["target"] = float(mult * mean_act[j])
            logp_s = final_logprobs()
            p_s = logp_s.exp()
            kl[ci, mi] = float((p_s * (logp_s - logp_clean)).sum(-1).mean())
            ent[ci, mi] = float(-(p_s * logp_s).sum(-1).mean())
            if mi == 1:
                boost = (logp_s - logp_clean).mean(0)
                ids = boost.topk(TOP_K).indices.tolist()
                top_tokens[int(j)] = [tokenizer.decode([t]) for t in ids]
        steer["j"] = None
        if ci % 25 == 0:
            print(f"candidate {ci}/{N_CAND} (latent {j}): "
                  f"KL@{MULTS[1]}x = {kl[ci, 1]:.3f}", flush=True)

    np.savez(out / "causal_screen.npz", candidates=cand, kl=kl,
             ent_steered=ent, ent_clean=ent_clean, mults=np.array(MULTS))
    (out / "top_tokens.json").write_text(json.dumps(top_tokens, indent=2))

    mid = kl[:, 1]
    summary = {
        "n_candidates": N_CAND,
        "n_prefixes": N_PREFIX,
        "prefix_len": PREFIX_LEN,
        "mults": MULTS,
        "seed": SEED,
        "ent_clean": ent_clean,
        "kl_quantiles_mid_strength": {q: float(np.quantile(mid, float(q)))
                                      for q in ["0.1", "0.25", "0.5", "0.75", "0.9"]},
        "hit_rate_kl_gt_0.05": float((mid > 0.05).mean()),
        "hit_rate_kl_gt_0.2": float((mid > 0.2).mean()),
        "hit_rate_kl_gt_0.5": float((mid > 0.5).mean()),
        "monotone_frac": float(((kl[:, 1] >= kl[:, 0]) & (kl[:, 2] >= kl[:, 1])).mean()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
