"""exp07 pilot: sanity-check the uniform-jump corpus before the full run.

Small batch of docs at the exp06 settled recipe (s = 1.5, p_stay = 0.95,
256-token docs) but with uniform jumps over the 7 other states instead of
ring +-1. Scored under all 8 state hypotheses (exact emissions) so
pilot_analyze.py can run the reader arithmetic. Two things to confirm
before committing 100M tokens:
  (a) the map is worthless by construction: a reader that wrongly assumes
      the ring should do no better than the true uniform-jump reader;
  (b) tracking value / reader accuracy stay in the same ballpark as the
      ring corpus (expected slightly lower early in dwells: 7 candidate
      destinations instead of 2).

Output: results/pilot/<tag>.npz {ids, z, logp (n,8,243), logp_clean,
p_stay, mult}

Usage (seahorse):
  python pilot_uniform.py --device cuda:0
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
from corpus_generate import K, PROMPT_LEN, RING, harvest_openers  # noqa: E402

P_STAY = 0.95
GEN_LEN = 243                # exp06 doc shape: BOS + 12 opener + 243
SEED = 707
SCORE_DOCS = 56
OUT = Path(__file__).parent.parent / "results" / "pilot"


def sample_paths(rng, n, length):
    z = np.empty((n, length), dtype=np.int8)
    z[:, 0] = rng.integers(0, K, n)
    u = rng.random((n, length - 1))
    jump = rng.integers(1, K, (n, length - 1)).astype(np.int8)
    for t in range(1, length):
        move = u[:, t - 1] >= P_STAY
        z[:, t] = np.where(move, (z[:, t - 1] + jump[:, t - 1]) % K,
                           z[:, t - 1])
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mult", type=float, default=1.5)
    ap.add_argument("--tag", default="uniform")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_docs", type=int, default=224)
    ap.add_argument("--batch", type=int, default=112)
    args = ap.parse_args()
    dev = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)

    model, tokenizer, sae = load_teacher(dev)
    print(f"{args.tag}: harvesting {args.n_docs} openers", flush=True)
    openers = harvest_openers(tokenizer, args.n_docs, 0, 8)

    steer = AddSteerer(model, sae, RING, dev)
    ma = mean_act()
    R = torch.zeros(K, K, device=dev)
    for s in range(K):
        R[s, s] = args.mult * float(ma[RING[s]])

    rng = np.random.default_rng([SEED])
    torch.manual_seed(SEED)
    z = sample_paths(rng, args.n_docs, GEN_LEN)
    jumps = np.diff(z)[np.diff(z) != 0]
    print(f"{args.tag}: mean switches/doc {(np.diff(z) != 0).sum(1).mean():.1f}, "
          f"jump offsets seen: {sorted(set((jumps % K).tolist()))}", flush=True)

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
        "p_stay": P_STAY, "mult": args.mult, "z0": int(z[0, 0]),
        "transitions": ((np.diff(z[0]) != 0).nonzero()[0] + 1).tolist(),
        "states": z[0][np.r_[0, (np.diff(z[0]) != 0).nonzero()[0] + 1]].tolist(),
        "text": tokenizer.decode(ids[0, 1:])[:2000]}, indent=1))

    vals = torch.tensor([args.mult * float(ma[j]) for j in RING], device=dev)
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
             p_stay=np.float64(P_STAY), mult=np.float64(args.mult))
    print(f"{args.tag} finished", flush=True)


if __name__ == "__main__":
    main()
