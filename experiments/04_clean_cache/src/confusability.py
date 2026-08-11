"""Step 4a: static-state docs + exact all-vs-all emission scoring for the
shortlist, at the shared strength from step 3.

Phase gen  (sharded by latent): N_DOCS docs x GEN_LEN tokens per shortlist
latent, clean-cache generation, openers shared across latents (controls for
opener variance in the confusability comparison).
Phase score (sharded by doc): every doc scored under every shortlist
hypothesis with the exact reader -> logp (n_docs, H, GEN_LEN) + clean logp.

The steerer pool for both phases is the shortlist itself (clamp one, ablate
the rest) — the final cast of 8 gets a recalibration pass in step 5.

Usage:
  python confusability.py --phase gen   --shard i --n_shards 4 --device cuda:i
  python confusability.py --phase score --shard i --n_shards 4 --device cuda:i
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from steering import (Steerer, generate_clean_cache, get_prompts,
                      load_teacher, mean_act, score_emissions)

N_DOCS = 24                 # per shortlist latent
GEN_LEN = 400
N_PROMPTS = 24
SCORE_BATCH = 6             # docs per scoring batch (x H hypothesis rows)
SEED = 0

ROOT = Path(__file__).parent.parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["gen", "score"], required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n_shards", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)

    out = ROOT / "confusability"
    out.mkdir(parents=True, exist_ok=True)
    sl = json.loads((ROOT / "audition" / "shortlist.json").read_text())
    latents, s_mult = sl["shortlist"], sl["mult"]
    ma = mean_act()

    model, tok, sae = load_teacher(dev)
    steer = Steerer(model, sae, latents, dev)
    vals = {j: s_mult * float(ma[j]) for j in latents}

    if args.phase == "gen":
        prompts = get_prompts(tok, N_PROMPTS)
        mine = latents[args.shard :: args.n_shards]
        ids_all, lat_all, texts = [], [], []
        for li, j in enumerate(mine):
            openers = prompts[torch.arange(N_DOCS) % N_PROMPTS]
            targets = steer.rows([j] * N_DOCS,
                                 torch.tensor([vals[j]] * N_DOCS, device=dev))
            torch.manual_seed(SEED + 200_000 + j)
            ids, _ = generate_clean_cache(model, steer, openers, targets,
                                          GEN_LEN)
            ids_all.append(ids.numpy().astype(np.int32))
            lat_all.extend([j] * N_DOCS)
            texts.append({"latent": j, "mult": s_mult,
                          "text": tok.decode(ids[0, 1:])})
            print(f"shard {args.shard}: gen latent {j} "
                  f"({li + 1}/{len(mine)})", flush=True)
        np.savez(out / f"gen_shard{args.shard}.npz",
                 ids=np.concatenate(ids_all),
                 latent=np.array(lat_all, dtype=np.int32))
        with open(out / f"texts_shard{args.shard}.jsonl", "w") as f:
            for row in texts:
                f.write(json.dumps(row) + "\n")
        print(f"shard {args.shard} gen finished", flush=True)
        return

    # phase score
    ids, lat = [], []
    for f in sorted(out.glob("gen_shard*.npz")):
        d = np.load(f)
        ids.append(d["ids"]); lat.append(d["latent"])
    ids = np.concatenate(ids); lat = np.concatenate(lat)
    mine = np.arange(len(ids))[args.shard :: args.n_shards]
    print(f"shard {args.shard}: scoring {len(mine)} docs "
          f"under {len(latents)} hypotheses", flush=True)

    hyp = steer.rows(latents,
                     torch.tensor([vals[j] for j in latents], device=dev))
    P = ids.shape[1] - GEN_LEN
    lp_all = np.empty((len(mine), len(latents), GEN_LEN), dtype=np.float32)
    lpc_all = np.empty((len(mine), GEN_LEN), dtype=np.float32)
    for b0 in range(0, len(mine), SCORE_BATCH):
        sel = mine[b0 : b0 + SCORE_BATCH]
        batch = torch.from_numpy(ids[sel].astype(np.int64))
        lp, lpc = score_emissions(model, steer, batch, hyp, P)
        lp_all[b0 : b0 + len(sel)] = lp.numpy()
        lpc_all[b0 : b0 + len(sel)] = lpc.numpy()
        print(f"shard {args.shard}: {b0 + len(sel)}/{len(mine)} docs",
              flush=True)
    np.savez(out / f"score_shard{args.shard}.npz",
             doc_idx=mine.astype(np.int32), doc_latent=lat[mine],
             logp=lp_all, logp_clean=lpc_all,
             hyp_latents=np.array(latents, dtype=np.int32))
    print(f"shard {args.shard} score finished", flush=True)


if __name__ == "__main__":
    main()
