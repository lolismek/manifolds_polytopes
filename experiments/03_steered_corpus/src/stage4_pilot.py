"""Stage 4 pilot: static-state documents + exact ideal-reader scoring.

For each cast state and each candidate strength, generate documents with that
state's latent clamped (one-hot over the cast: active latent at s * mean_act,
other 7 ablated). Then replay every document under all 8 states' clamps at the
same strength and record per-token log-likelihoods. From these, the analysis
computes the exact posterior trajectory an ideal reader would hold — the
instrument that picks the corpus strength.

Sharded by state: shard g handles cast states g::n_shards (generation); each
shard scores its own documents under all 8 hypotheses.

Outputs (results/stage4/):
  pilot_shard{g}.npz   ids (n, GEN_LEN), state (n,), strength (n,),
                       loglik (n, 8, GEN_LEN)  [per-token log p under each hypothesis]
  texts_shard{g}.jsonl 2 decoded samples per (state, strength)

Usage (on tigerfish):
  python stage4_pilot.py --shard 0 --n_shards 4 --device cuda:0
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

CAST = [368, 1220, 2404, 2970, 6172, 10615, 10621, 13931]
RESERVES = [9508, 10013]
MULTS = [3.0, 5.0, 8.0]
N_DOCS = 24             # per (state, strength)
PROMPT_LEN = 12
N_PROMPTS = 20
GEN_LEN = 300
SEED = 0


def get_prompts(tokenizer):
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    bos = tokenizer.bos_token_id
    prompts = []
    for ex in ds:
        toks = tokenizer(ex["text"], add_special_tokens=False)["input_ids"]
        if len(toks) >= PROMPT_LEN:
            prompts.append([bos] + toks[:PROMPT_LEN])
        if len(prompts) == N_PROMPTS:
            return torch.tensor(prompts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n_shards", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(__file__).parent.parent / "results" / "stage4"
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)

    from sae_lens import SAE
    sae = SAE.from_pretrained(SAE_RELEASE, SAE_ID, device=str(dev))[0].to(torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev)
    model.eval()

    mean_act = np.load(Path(__file__).parent.parent / "results" / "stage1"
                       / "latent_stats.npz")["mean_act"]
    cast_t = torch.tensor(CAST, device=dev)
    W_cast = sae.W_dec[cast_t]                     # (8, d)

    steer = {"t": None}                            # target vector over the cast

    def hook(_, __, output):
        if steer["t"] is None:
            return
        is_tuple = isinstance(output, tuple)
        x = output[0] if is_tuple else output
        xf = x.float()
        a = sae.encode(xf.reshape(-1, xf.shape[-1]))[:, cast_t]
        delta = ((steer["t"] - a) @ W_cast).reshape(x.shape)
        x = (xf + delta).to(x.dtype)
        return (x,) + output[1:] if is_tuple else x

    model.model.layers[LAYER].register_forward_hook(hook)
    prompts = get_prompts(tokenizer)

    def onehot(state_idx, mult):
        t = torch.zeros(len(CAST), device=dev)
        t[state_idx] = mult * float(mean_act[CAST[state_idx]])
        return t

    mine = list(range(len(CAST)))[args.shard :: args.n_shards]
    all_ids, all_state, all_mult, texts = [], [], [], []

    for si in mine:
        for mult in MULTS:
            steer["t"] = onehot(si, mult)
            torch.manual_seed(SEED + 100 * CAST[si] + int(mult))
            idx = torch.arange(N_DOCS) % N_PROMPTS
            ids = prompts[idx].to(dev)
            with torch.no_grad():
                gen = model.generate(ids, do_sample=True, temperature=1.0, top_k=0,
                                     top_p=1.0, max_new_tokens=GEN_LEN,
                                     min_new_tokens=GEN_LEN,
                                     pad_token_id=tokenizer.eos_token_id)
            steer["t"] = None
            all_ids.append(gen.cpu())               # full seq incl. prompt
            all_state += [si] * N_DOCS
            all_mult += [mult] * N_DOCS
            texts += [{"state": si, "latent": CAST[si], "mult": mult,
                       "text": tokenizer.decode(g[prompts.shape[1]:])}
                      for g in gen[:2].cpu()]
        print(f"shard {args.shard}: state {si} (latent {CAST[si]}) generated", flush=True)

    seqs = torch.cat(all_ids)                       # (n, 1+PROMPT_LEN+GEN_LEN)
    state_arr = np.array(all_state, dtype=np.int32)
    mult_arr = np.array(all_mult, dtype=np.float32)
    n = seqs.shape[0]
    p_len = prompts.shape[1]                        # 1 + PROMPT_LEN

    # score every doc under all 8 hypotheses at its own generation strength
    loglik = np.zeros((n, len(CAST), GEN_LEN), dtype=np.float32)
    torch.cuda.empty_cache()
    with torch.no_grad():
        for hyp in range(len(CAST)):
            for mult in MULTS:
                rows = np.where(mult_arr == mult)[0]
                for i in range(0, len(rows), 8):
                    r = rows[i : i + 8]
                    chunk = seqs[r].to(dev)
                    steer["t"] = onehot(hyp, mult)
                    logits = model(chunk).logits[:, :-1]
                    steer["t"] = None
                    tok_lp = -torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]).float(),
                        chunk[:, 1:].reshape(-1), reduction="none"
                    ).reshape(chunk.shape[0], -1)
                    loglik[r, hyp] = tok_lp[:, p_len - 1 :].cpu().numpy()
            print(f"shard {args.shard}: hypothesis {hyp} scored", flush=True)

    np.savez(out / f"pilot_shard{args.shard}.npz",
             ids=seqs.numpy().astype(np.int32), state=state_arr, mult=mult_arr,
             loglik=loglik, cast=np.array(CAST))
    with open(out / f"texts_shard{args.shard}.jsonl", "w") as f:
        for row in texts:
            f.write(json.dumps(row) + "\n")
    print(f"shard {args.shard} finished: {n} docs scored under {len(CAST)} hypotheses",
          flush=True)


if __name__ == "__main__":
    main()
