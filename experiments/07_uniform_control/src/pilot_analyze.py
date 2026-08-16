"""Reader arithmetic for the exp07 uniform-jump pilot, side by side with
the exp06 ring corpus (its scored eval shard 0 — same recipe, real corpus).

Per corpus, every reader consumes the exact per-state emission table and
predicts the next token as its belief-weighted mixture:
  oracle       knows the true state z_t (ceiling)
  true-model   full Bayes with the CORRECT transition matrix for that
               corpus (ring +-1 for exp06, uniform-over-7 for exp07)
  other-map    full Bayes with the OTHER corpus's transition matrix
  forget       decayed evidence sum, best gamma on a grid
  none         uniform belief always

Money numbers:
  map_value      = loss(other-map) - loss(true-model)
                   ring corpus:    what knowing the wiring is worth (~0.02)
                   uniform corpus: what wrongly assuming a ring costs
                   (should be ~0: there is no map to know)
  tracking_value = loss(none) - loss(true-model)

Usage:
  python pilot_analyze.py [results/pilot/uniform.npz] [--n_docs 224]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

EXP06_SRC = Path(__file__).parents[2] / "06_short_dwell" / "src"
sys.path.insert(0, str(EXP06_SRC))
from dwell_analyze import (GAMMAS, bayes_reader, forget_reader,  # noqa: E402
                           logsumexp, nomap_T, ring_T)

K = 8
EXP06_EVAL = (Path(__file__).parents[2] / "06_short_dwell" / "results"
              / "posterior" / "eval_shard0.npz")


def analyze(path, true_T, other_T, label, n_docs):
    d = np.load(path)
    logp = d["logp"][:n_docs].astype(np.float64)
    z = d["z"][:n_docs, :logp.shape[2]].astype(np.int64)
    clean = -d["logp_clean"][:n_docs].astype(np.float64).mean()

    oracle = -logp[np.arange(len(z))[:, None], z,
                   np.arange(logp.shape[2])[None]].mean()
    none = -(logsumexp(logp, 1) - np.log(K)).mean()
    l_true, acc_true, maxb = bayes_reader(logp, z, true_T)
    l_other, acc_other, _ = bayes_reader(logp, z, other_T)
    fg = {g: forget_reader(logp, g) for g in GAMMAS}
    g_best = min(fg, key=fg.get)

    r = lambda x: round(float(x), 4)  # noqa: E731
    return {"corpus": label, "n_docs": int(len(z)), "n_tokens": int(z.size),
            "loss_clean_model": r(clean), "loss_oracle": r(oracle),
            "loss_true_model": r(l_true), "loss_other_map": r(l_other),
            "loss_forget": r(fg[g_best]), "forget_gamma": g_best,
            "loss_none": r(none),
            "map_value": r(l_other - l_true),
            "memory_value": r(none - fg[g_best]),
            "tracking_value": r(none - l_true),
            "acc_true": r(acc_true), "acc_other": r(acc_other),
            "mean_max_belief": r(maxb)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pilot", nargs="?",
                    default=str(Path(__file__).parent.parent / "results"
                                / "pilot" / "uniform.npz"))
    ap.add_argument("--n_docs", type=int, default=224)
    args = ap.parse_args()

    p = 0.95
    rows = [
        analyze(EXP06_EVAL, ring_T(p), nomap_T(p), "ring (exp06 eval sh0)",
                args.n_docs),
        analyze(args.pilot, nomap_T(p), ring_T(p), "uniform (exp07 pilot)",
                args.n_docs),
    ]

    cols = ["corpus", "n_tokens", "loss_clean_model", "loss_oracle",
            "loss_true_model", "loss_other_map", "loss_forget",
            "forget_gamma", "loss_none", "map_value", "memory_value",
            "tracking_value", "acc_true", "acc_other", "mean_max_belief"]
    print("\t".join(cols))
    for row in rows:
        print("\t".join(str(row[c]) for c in cols))

    out = Path(__file__).parent.parent / "results" / "pilot" / "readers.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
