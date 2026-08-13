"""Exact ideal-reader ground truth for the exp05 eval split.

Eval split (held out from student training): the first 715 docs of each
shard's last chunk (chunk005, docs 12500-14285) x 7 shards -> 5005 docs.

For each eval doc, score_emissions teacher-forces the text under all 8 cast
hypotheses (multi-layer add, clean cache) giving the exact emission table
logp[doc, state, t]; the HMM forward algorithm with the ring transition
matrix (p_stay 0.98) yields the exact filtered posterior P(z_t | tokens<=t).

Usage: python corpus_score_posterior.py --shard i --n_shards 7 --device cuda:d
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from addsteer import AddSteerer, load_teacher, mean_act, score_emissions
from corpus_generate import GEN_LEN, MULT, P_STAY, PROMPT_LEN, RING

EVAL_CHUNK = "chunk005"
N_EVAL_PER_SHARD = 715
SCORE_DOCS = 28

ROOT = Path(__file__).parent.parent / "results"


def forward(logp, p_stay):
    """logp (T, 8) exact emission log-probs -> beliefs (T, 8)."""
    T, K = logp.shape
    logT = np.full((K, K), -np.inf)
    for i in range(K):
        logT[i, i] = np.log(p_stay)
        logT[i, (i + 1) % K] = logT[i, (i - 1) % K] = np.log((1 - p_stay) / 2)
    la = -np.log(K) + logp[0]
    b = np.empty((T, K))
    b[0] = np.exp(la - la.max()); b[0] /= b[0].sum()
    for t in range(1, T):
        m = la.max()
        trans = m + np.log(np.exp(la - m) @ np.exp(logT))
        la = logp[t] + trans
        b[t] = np.exp(la - la.max()); b[t] /= b[t].sum()
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n_shards", type=int, default=7)
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
    steer = AddSteerer(model, sae, RING, dev)
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
