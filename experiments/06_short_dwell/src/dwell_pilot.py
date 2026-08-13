"""exp06 pilot: would a SHORTER DWELL make the ring worth learning?

Exp05's post-mortem: the student learned per-state flavor detectors plus a
sloppy running average, never the transition map (the ring), because at
p_stay = 0.98 (dwell ~50) knowing the map barely improves next-token loss —
the only currency training pays. Before committing to a new corpus we
measure that payoff directly on small batches of REAL pilot documents
generated at shorter dwells with the exp05 pipeline unchanged (same cast,
same s = 1.0 multi-layer add, clean-cache physics, same openers); only
p_stay varies. Each pilot is then scored under all 8 state hypotheses
(exact emissions), which is all the downstream reader arithmetic needs
(dwell_analyze.py compares readers that do/don't know the ring wiring).

Openers: first n_docs usable Pile openers (residue 0 mod 7) — identical
across pilot settings, so settings differ only in the state rhythm.

Output: results/pilot/<tag>.npz {ids, z, logp (n,8,T fp16), logp_clean,
p_stay}

Usage (tigerfish, one setting per free GPU):
  python dwell_pilot.py --p_stay 0.95 --tag p95 --device cuda:0
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

EXP05_SRC = Path(__file__).parents[2] / "05_multilayer_add" / "src"
sys.path.insert(0, str(EXP05_SRC))
from addsteer import (AddSteerer, generate_clean_cache, load_teacher,  # noqa: E402
                      mean_act, score_emissions)
from corpus_generate import (GEN_LEN, K, MULT, PROMPT_LEN, RING,  # noqa: E402
                             harvest_openers)

SEED = 606
SCORE_DOCS = 28
OUT = Path(__file__).parent.parent / "results" / "pilot"


def sample_paths(rng, n, length, p_stay):
    z = np.empty((n, length), dtype=np.int8)
    z[:, 0] = rng.integers(0, K, n)
    u = rng.random((n, length - 1))
    step = rng.integers(0, 2, (n, length - 1)).astype(np.int8) * 2 - 1
    for t in range(1, length):
        move = u[:, t - 1] >= p_stay
        z[:, t] = np.where(move, (z[:, t - 1] + step[:, t - 1]) % K,
                           z[:, t - 1])
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p_stay", type=float, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_docs", type=int, default=224)
    ap.add_argument("--batch", type=int, default=112)
    args = ap.parse_args()
    dev = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)

    model, tokenizer, sae = load_teacher(dev)
    print(f"{args.tag}: harvesting {args.n_docs} openers", flush=True)
    openers = harvest_openers(tokenizer, args.n_docs, 0, 7)

    steer = AddSteerer(model, sae, RING, dev)
    ma = mean_act()
    R = torch.zeros(K, K, device=dev)
    for s in range(K):
        R[s, s] = MULT * float(ma[RING[s]])

    rng = np.random.default_rng([SEED, int(round(args.p_stay * 1000))])
    torch.manual_seed(SEED * 1000 + int(round(args.p_stay * 1000)))
    z = sample_paths(rng, args.n_docs, GEN_LEN, args.p_stay)
    print(f"{args.tag}: p_stay {args.p_stay}, mean switches/doc "
          f"{(np.diff(z) != 0).sum(1).mean():.1f}", flush=True)

    ids_parts = []
    t0 = time.time()
    with torch.no_grad():
        for b0 in range(0, args.n_docs, args.batch):
            b1 = min(b0 + args.batch, args.n_docs)
            zb = torch.from_numpy(z[b0:b1]).long().to(dev)
            ids_b, _ = generate_clean_cache(
                model, steer, openers[b0:b1], lambda t: R[zb[:, t]], GEN_LEN)
            ids_parts.append(ids_b)
            rate = (b1 - b0) * GEN_LEN / max(time.time() - t0, 1e-9)
            t0 = time.time()
            print(f"{args.tag}: generated {b1}/{args.n_docs} docs "
                  f"({rate:.0f} tok/s)", flush=True)
    ids = torch.cat(ids_parts)

    (OUT / f"{args.tag}_sample.json").write_text(json.dumps({
        "p_stay": args.p_stay, "z0": int(z[0, 0]),
        "transitions": ((np.diff(z[0]) != 0).nonzero()[0] + 1).tolist(),
        "text": tokenizer.decode(ids[0, 1:])[:2000]}, indent=1))

    vals = torch.tensor([MULT * float(ma[j]) for j in RING], device=dev)
    hyp_rows = steer.rows(RING, vals)
    lps, lcs = [], []
    for b0 in range(0, args.n_docs, SCORE_DOCS):
        lp, lc = score_emissions(model, steer, ids[b0:b0 + SCORE_DOCS],
                                 hyp_rows, PROMPT_LEN + 1)
        lps.append(lp.numpy())
        lcs.append(lc.numpy())
        print(f"{args.tag}: scored {b0 + lp.shape[0]}/{args.n_docs} docs",
              flush=True)

    np.savez(OUT / f"{args.tag}.npz",
             ids=ids.numpy().astype(np.int32), z=z,
             logp=np.concatenate(lps).astype(np.float16),
             logp_clean=np.concatenate(lcs).astype(np.float16),
             p_stay=np.float64(args.p_stay))
    print(f"{args.tag} finished", flush=True)


if __name__ == "__main__":
    main()
