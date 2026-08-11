"""Step 2: dose-response audition of the 48 candidates under clean-cache
steering.

For every candidate latent and strength in MULTS, generate N_DOCS docs with
the clean-cache generator (single-position clamp; other pool latents ablated)
and record, per doc, the per-token evidence rate KL(clamped || clean), the
realized log-likelihood ratio of sampled tokens, the clean-teacher NLL of the
generated text (fluency — computed for free from the clean pass), and the
clamped output entropy. Shard 0 also runs a clean arm (no clamp) for the
fluency baseline.

The single-position clamp is weaker per token than exp03's compounding clamp
(exp03 stage-2.5 numbers were whole-prefix), hence MULTS reach 18x.

Sharded by latent: python audition.py --shard i --n_shards 4 --device cuda:i
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from steering import (Steerer, audition_pool, generate_clean_cache,
                      get_prompts, load_teacher, mean_act)

MULTS = [3.0, 5.0, 8.0, 12.0, 18.0]
N_DOCS = 16                # docs per (latent, strength) cell
N_CLEAN = 32
GEN_LEN = 300
N_PROMPTS = 20
SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n_shards", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)

    out = Path(__file__).parent.parent / "results" / "audition"
    out.mkdir(parents=True, exist_ok=True)

    model, tok, sae = load_teacher(dev)
    pool = audition_pool()
    steer = Steerer(model, sae, pool, dev)
    ma = mean_act()
    prompts = get_prompts(tok, N_PROMPTS)

    mine = pool[args.shard :: args.n_shards]
    print(f"shard {args.shard}: {len(mine)} latents", flush=True)

    rows_lat, rows_mult, ids_all = [], [], []
    stats_all = {k: [] for k in ["kl", "llr", "nll_clean", "ent"]}
    texts = []

    def run(latent, mults):
        """One batch: len(mults) x N_DOCS docs for one latent."""
        B = len(mults) * N_DOCS
        openers = prompts[torch.arange(B) % N_PROMPTS]
        if latent < 0:
            targets = None
        else:
            lat_col = [latent] * B
            vals = torch.tensor([m * float(ma[latent]) for m in mults
                                 for _ in range(N_DOCS)], device=dev)
            targets = steer.rows(lat_col, vals)
        torch.manual_seed(SEED + 100_000 + latent)
        ids, stats = generate_clean_cache(model, steer, openers, targets,
                                          GEN_LEN)
        ids_all.append(ids.numpy().astype(np.int32))
        rows_lat.extend([latent] * B)
        rows_mult.extend([m for m in mults for _ in range(N_DOCS)])
        for k in stats_all:
            stats_all[k].append(stats[k].numpy())
        for ci, m in enumerate(mults):
            texts.append({"latent": latent, "mult": m,
                          "text": tok.decode(ids[ci * N_DOCS, 1:])})

    if args.shard == 0:
        B = N_CLEAN
        openers = prompts[torch.arange(B) % N_PROMPTS]
        torch.manual_seed(SEED)
        ids, stats = generate_clean_cache(model, steer, openers, None, GEN_LEN)
        ids_all.append(ids.numpy().astype(np.int32))
        rows_lat.extend([-1] * B)
        rows_mult.extend([0.0] * B)
        for k in stats_all:
            stats_all[k].append(stats[k].numpy())
        print("clean arm done", flush=True)

    for li, j in enumerate(mine):
        run(j, MULTS)
        print(f"shard {args.shard}: latent {j} done ({li + 1}/{len(mine)})",
              flush=True)

    np.savez(out / f"audition_shard{args.shard}.npz",
             latent=np.array(rows_lat, dtype=np.int32),
             mult=np.array(rows_mult, dtype=np.float32),
             ids=np.concatenate(ids_all),
             **{k: np.concatenate(v) for k, v in stats_all.items()})
    with open(out / f"texts_shard{args.shard}.jsonl", "w") as f:
        for row in texts:
            f.write(json.dumps(row) + "\n")
    print(f"shard {args.shard} finished", flush=True)


if __name__ == "__main__":
    main()
