"""Exact ideal-reader ground truth for the exp04 eval split.

Eval split (exp03 convention, held out from student training): the first 1250
docs of each shard's last chunk (chunk009, docs 22500-23749) -> 5000 docs.

For each eval doc, score_emissions teacher-forces the text under all 8 cast
hypotheses (single-position clamp, clean cache) giving the exact emission
table logp[doc, state, t]; the HMM forward algorithm with the known ring
transition matrix (p_stay 0.99) then yields the exact filtered posterior
P(z_t | tokens <= t). No static-hypothesis approximation term exists in exp04
— these emissions ARE the distributions generation sampled from.

Batching: SCORE_DOCS docs per batch -> SCORE_DOCS*8 rows per forward
(default 28 -> 224 rows, the generation-benchmarked batch size).

Output (results/posterior/): eval_shard{g}.npz with
  ids (n, 1024) int32, z (n, 1011) int8, logp (n, 8, 1011) float16,
  logp_clean (n, 1011) float16, posterior (n, 1011, 8) float16

Usage: python corpus_score_posterior.py --shard i --n_shards 4 --device cuda:i
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from corpus_generate import GEN_LEN, MULT, P_STAY, PROMPT_LEN, RING
from pilot_analyze import forward
from steering import Steerer, load_teacher, mean_act, score_emissions

EVAL_CHUNK = "chunk009"
N_EVAL_PER_SHARD = 1250
SCORE_DOCS = 28

ROOT = Path(__file__).parent.parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n_shards", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)

    out = ROOT / "posterior"
    out.mkdir(parents=True, exist_ok=True)

    d = np.load(ROOT / "corpus" / f"ring_shard{args.shard}_{EVAL_CHUNK}.npz")
    ids = torch.tensor(d["ids"][:N_EVAL_PER_SHARD], dtype=torch.long)
    z = d["z"][:N_EVAL_PER_SHARD]
    n = ids.shape[0]

    model, tok, sae = load_teacher(dev)
    steer = Steerer(model, sae, RING, dev)
    ma = mean_act()
    vals = torch.tensor([MULT * float(ma[j]) for j in RING], device=dev)
    hyp_rows = steer.rows(RING, vals)

    lps, lcs = [], []
    for b0 in range(0, n, SCORE_DOCS):
        lp, lc = score_emissions(model, steer, ids[b0:b0 + SCORE_DOCS],
                                 hyp_rows, PROMPT_LEN + 1)
        lps.append(lp.numpy())
        lcs.append(lc.numpy())
        print(f"shard {args.shard}: {b0 + lp.shape[0]}/{n} docs scored",
              flush=True)
    logp = np.concatenate(lps)                       # (n, 8, GEN_LEN)
    logp_clean = np.concatenate(lcs)

    post = np.empty((n, GEN_LEN, 8), dtype=np.float16)
    for i in range(n):
        post[i] = forward(logp[i].T.astype(np.float64), P_STAY)
    acc = float((post.argmax(-1) == z[:, :GEN_LEN]).mean())
    print(f"shard {args.shard}: reader argmax accuracy {acc:.3f}", flush=True)

    np.savez(out / f"eval_shard{args.shard}.npz",
             ids=ids.numpy().astype(np.int32), z=z,
             logp=logp.astype(np.float16),
             logp_clean=logp_clean.astype(np.float16),
             posterior=post)
    print(f"shard {args.shard} finished", flush=True)


if __name__ == "__main__":
    main()
