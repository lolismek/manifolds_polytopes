"""Physics pilot: does the multi-layer window buy ~3x margin at gentle s?

Phase gen  (sharded by latent): for each exp04 cast latent x s in MULTS,
N_DOCS static-state docs from shared pile openers; shard 0 also generates a
clean arm (latent -1, mult 0) for the fluency baseline.
Phase score (sharded by doc): every steered doc scored under all 8 cast
hypotheses at the doc's own s with the exact reader.

Usage:
  python pilot.py --phase gen   --shard i --n_shards 4 --device cuda:i
  python pilot.py --phase score --shard i --n_shards 4 --device cuda:i
Then: python pilot_analyze.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from addsteer import (AddSteerer, cast, generate_clean_cache, get_prompts,
                      load_teacher, mean_act, score_emissions)

MULTS = [1.0, 1.5, 2.0]    # verify_identity: s=2 already gives KL 0.89/tok,
                           # margin ~0.45 — 4x would be pointlessly loud
N_DOCS = 6                  # per (latent, mult)
N_CLEAN = 8
GEN_LEN = 300
N_PROMPTS = 24
SCORE_BATCH = 6             # docs per scoring batch (x 8 hypothesis rows)
SEED = 0

OUT = Path(__file__).parent.parent / "results" / "pilot"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["gen", "score"], required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n_shards", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)

    cast_l = cast()
    ma = mean_act()
    model, tok, sae = load_teacher(dev)
    steer = AddSteerer(model, sae, cast_l, dev)

    if args.phase == "gen":
        prompts = get_prompts(tok, N_PROMPTS)
        arms = [(j, m) for j in cast_l[args.shard :: args.n_shards]
                for m in MULTS]
        if args.shard == 0:
            arms.append((-1, 0.0))
        ids_l, lat_l, mult_l, texts = [], [], [], []
        stats_l = {k: [] for k in ("kl", "nll_clean", "ent")}
        for ai, (j, m) in enumerate(arms):
            n = N_CLEAN if j == -1 else N_DOCS
            openers = prompts[torch.arange(n) % N_PROMPTS]
            targets = None if j == -1 else steer.rows(
                [j] * n, torch.tensor([m * float(ma[j])] * n, device=dev))
            torch.manual_seed(SEED + 1000 * (j + 1) + int(10 * m))
            ids, stats = generate_clean_cache(model, steer, openers, targets,
                                              GEN_LEN)
            ids_l.append(ids.numpy().astype(np.int32))
            lat_l.extend([j] * n)
            mult_l.extend([m] * n)
            for k in stats_l:
                stats_l[k].append(stats[k].numpy())
            texts.append({"latent": j, "mult": m,
                          "text": tok.decode(ids[0, 1:])})
            print(f"shard {args.shard}: gen ({j}, {m}x) "
                  f"({ai + 1}/{len(arms)})", flush=True)
        np.savez(OUT / f"gen_shard{args.shard}.npz",
                 ids=np.concatenate(ids_l),
                 latent=np.array(lat_l, dtype=np.int32),
                 mult=np.array(mult_l, dtype=np.float32),
                 **{k: np.concatenate(v) for k, v in stats_l.items()})
        with open(OUT / f"texts_shard{args.shard}.jsonl", "w") as f:
            for row in texts:
                f.write(json.dumps(row) + "\n")
        print(f"shard {args.shard} gen finished", flush=True)
        return

    # phase score
    ids, lat, mult = [], [], []
    for f in sorted(OUT.glob("gen_shard*.npz")):
        d = np.load(f)
        ids.append(d["ids"]); lat.append(d["latent"]); mult.append(d["mult"])
    ids = np.concatenate(ids); lat = np.concatenate(lat)
    mult = np.concatenate(mult)
    P = ids.shape[1] - GEN_LEN

    idx_l, lp_l, lpc_l = [], [], []
    for m in MULTS:
        hyp = steer.rows(cast_l, torch.tensor(
            [m * float(ma[j]) for j in cast_l], device=dev))
        mine = np.where(mult == m)[0][args.shard :: args.n_shards]
        for b0 in range(0, len(mine), SCORE_BATCH):
            sel = mine[b0 : b0 + SCORE_BATCH]
            batch = torch.from_numpy(ids[sel].astype(np.int64))
            lp, lpc = score_emissions(model, steer, batch, hyp, P)
            idx_l.append(sel); lp_l.append(lp.numpy()); lpc_l.append(lpc.numpy())
            print(f"shard {args.shard}: mult {m}x "
                  f"{b0 + len(sel)}/{len(mine)} docs", flush=True)
    idx = np.concatenate(idx_l)
    np.savez(OUT / f"score_shard{args.shard}.npz",
             doc_idx=idx.astype(np.int32), doc_latent=lat[idx],
             doc_mult=mult[idx], logp=np.concatenate(lp_l),
             logp_clean=np.concatenate(lpc_l),
             hyp_latents=np.array(cast_l, dtype=np.int32))
    print(f"shard {args.shard} score finished", flush=True)


if __name__ == "__main__":
    main()
