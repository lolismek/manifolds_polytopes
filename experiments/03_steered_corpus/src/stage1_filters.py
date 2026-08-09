"""Stage 1 of the latent casting funnel: cheap filters, no generation.

Runs Gemma-2-2B over a slice of natural text, encodes layer-12 residuals with
the Gemma Scope 16k SAE, and collects per-latent statistics:

  - fire_freq        fraction of (non-BOS) token positions where the latent fires
  - mean_act         mean activation over firing positions
  - frac_noncontent  fraction of firings that land on whitespace/punct/digit tokens
  - frac_early       fraction of firings in the first 2 positions after BOS

Then applies the stage-1 cuts (dead / too frequent / formatting-dominated /
position-dominated) and, for survivors, computes pairwise decoder cosines.

Outputs (results/stage1/):
  latent_stats.npz   raw per-latent stats over all 16384 latents
  survivors.json     latent ids passing the cuts + thresholds used + summary counts
  cosines_survivors.npy  pairwise |cos| of survivor decoder rows (fp16)

Usage (on tigerfish):
  python stage1_filters.py --n_tokens 4_000_000 --device cuda:0
"""

import argparse
import json
import string
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "google/gemma-2-2b"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_12/width_16k/canonical"
LAYER = 12
SEQ_LEN = 1024
BATCH = 16

# stage-1 cut thresholds (chosen a priori; histograms in the report let us revisit)
DEAD_FREQ = 1e-5        # fires less than ~once per 100k tokens
COMMON_FREQ = 1e-2      # fires on more than 1% of tokens
NONCONTENT_MAX = 0.5    # more than half of firings on whitespace/punct/digits
EARLY_MAX = 0.5         # more than half of firings in first 2 positions
COS_NEAR_DUP = 0.7      # flag survivor pairs above this (report only, cut later)


def noncontent_mask(tokenizer) -> np.ndarray:
    """Boolean over the vocab: token is whitespace/punctuation/digits only."""
    junk = set(string.punctuation + string.digits + string.whitespace)
    toks = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    mask = np.zeros(len(toks), dtype=bool)
    for i, t in enumerate(toks):
        s = t.replace("▁", "")  # sentencepiece word-boundary marker
        mask[i] = (len(s) == 0) or all(c in junk for c in s)
    return mask


def token_stream(tokenizer, n_tokens: int):
    """Yield (BATCH, SEQ_LEN) input_id tensors from pile-uncopyrighted, BOS-prefixed."""
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    bos = tokenizer.bos_token_id
    buf, yielded = [], 0
    batch = []
    for ex in ds:
        buf.extend(tokenizer(ex["text"], add_special_tokens=False)["input_ids"])
        while len(buf) >= SEQ_LEN - 1:
            batch.append([bos] + buf[: SEQ_LEN - 1])
            buf = buf[SEQ_LEN - 1 :]
            if len(batch) == BATCH:
                yield torch.tensor(batch)
                yielded += BATCH * SEQ_LEN
                batch = []
                if yielded >= n_tokens:
                    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_tokens", type=int, default=4_000_000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out or Path(__file__).parent.parent / "results" / "stage1")
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)

    from sae_lens import SAE

    sae = SAE.from_pretrained(SAE_RELEASE, SAE_ID, device=str(dev))[0]
    sae = sae.to(torch.float32)
    n_lat = sae.cfg.d_sae

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev)
    model.eval()

    # capture the residual stream after block LAYER (what the res SAE was trained on)
    resid = {}
    def hook(_, __, output):
        resid["x"] = output[0]
    model.model.layers[LAYER].register_forward_hook(hook)

    nc_mask = torch.tensor(noncontent_mask(tokenizer), device=dev)

    fire = torch.zeros(n_lat, dtype=torch.float64, device=dev)
    act_sum = torch.zeros(n_lat, dtype=torch.float64, device=dev)
    fire_nc = torch.zeros(n_lat, dtype=torch.float64, device=dev)
    fire_early = torch.zeros(n_lat, dtype=torch.float64, device=dev)
    n_pos = 0

    with torch.no_grad():
        for bi, ids in enumerate(token_stream(tokenizer, args.n_tokens)):
            ids = ids.to(dev)
            model(ids)
            x = resid["x"][:, 1:, :].float()          # drop BOS position
            toks = ids[:, 1:]
            acts = sae.encode(x.reshape(-1, x.shape[-1]))  # (B*T, n_lat)
            firing = acts > 0
            fire += firing.sum(0)
            act_sum += (acts * firing).sum(0).double()
            nc = nc_mask[toks.reshape(-1)]
            fire_nc += (firing & nc[:, None]).sum(0)
            early = torch.zeros(toks.numel(), dtype=torch.bool, device=dev)
            early.view(toks.shape)[:, :2] = True      # first 2 positions after BOS
            fire_early += (firing & early[:, None]).sum(0)
            n_pos += toks.numel()
            if bi % 25 == 0:
                print(f"batch {bi}, {n_pos:,} positions", flush=True)

    fire_freq = (fire / n_pos).cpu().numpy()
    mean_act = (act_sum / fire.clamp(min=1)).cpu().numpy()
    frac_nc = (fire_nc / fire.clamp(min=1)).cpu().numpy()
    frac_early = (fire_early / fire.clamp(min=1)).cpu().numpy()

    np.savez(out / "latent_stats.npz", fire_freq=fire_freq, mean_act=mean_act,
             frac_noncontent=frac_nc, frac_early=frac_early, n_pos=n_pos)

    dead = fire_freq < DEAD_FREQ
    common = fire_freq > COMMON_FREQ
    nc_dom = frac_nc > NONCONTENT_MAX
    early_dom = frac_early > EARLY_MAX
    ok = ~(dead | common | nc_dom | early_dom)
    survivors = np.where(ok)[0]

    # pairwise decoder cosines among survivors
    W = sae.W_dec.detach()[torch.tensor(survivors, device=dev)].float()
    W = W / W.norm(dim=1, keepdim=True)
    cos = (W @ W.T).abs().cpu().numpy().astype(np.float16)
    np.fill_diagonal(cos, 0)
    np.save(out / "cosines_survivors.npy", cos)
    n_dup_pairs = int((cos > COS_NEAR_DUP).sum() // 2)

    summary = {
        "n_pos": n_pos,
        "n_latents": int(n_lat),
        "cuts": {"dead": int(dead.sum()), "too_common": int(common.sum()),
                 "noncontent_dominated": int(nc_dom.sum()),
                 "early_position_dominated": int(early_dom.sum())},
        "thresholds": {"DEAD_FREQ": DEAD_FREQ, "COMMON_FREQ": COMMON_FREQ,
                       "NONCONTENT_MAX": NONCONTENT_MAX, "EARLY_MAX": EARLY_MAX},
        "n_survivors": int(ok.sum()),
        "near_duplicate_pairs_above_0.7": n_dup_pairs,
        "survivors": survivors.tolist(),
    }
    (out / "survivors.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "survivors"}, indent=2))


if __name__ == "__main__":
    main()
