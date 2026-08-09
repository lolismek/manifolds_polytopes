"""Stage 4 analysis: ideal-reader posterior trajectories from the pilot.

For each document, the exact posterior over the 8 cast states given tokens
1..t is softmax over cumulative log-likelihoods (uniform prior). Reports, per
strength:
  - convergence time: first t where the true state's posterior >= 0.9
    (median, quartiles; fraction converged by 50/100/200/300 tokens)
  - end-of-document accuracy of the posterior's argmax
  - 8x8 confusion matrix at end of document
  - mean posterior entropy at t = 25/50/100 (how graded beliefs are early on)

Output: results/stage4/convergence.json  (+ printed summary)

Usage: python stage4_analyze.py
"""

import json
from pathlib import Path

import numpy as np

THRESH = 0.9
CHECKPOINTS = [50, 100, 200, 300]
ENT_TS = [25, 50, 100]


def main():
    out = Path(__file__).parent.parent / "results" / "stage4"
    shards = sorted(out.glob("pilot_shard*.npz"))
    loglik = np.concatenate([np.load(s)["loglik"] for s in shards])
    state = np.concatenate([np.load(s)["state"] for s in shards])
    mult = np.concatenate([np.load(s)["mult"] for s in shards])
    cast = np.load(shards[0])["cast"]
    n, K, T = loglik.shape

    cum = loglik.cumsum(-1)                        # (n, K, T)
    cum -= cum.max(1, keepdims=True)
    post = np.exp(cum)
    post /= post.sum(1, keepdims=True)             # posterior trajectory

    report = {"n_docs": int(n), "cast": cast.tolist(), "threshold": THRESH,
              "per_strength": {}}
    for m in sorted(set(mult.tolist())):
        sel = np.where(mult == m)[0]
        p_true = post[sel, state[sel], :]           # (n_sel, T)
        conv = np.full(len(sel), T + 1)
        for i in range(len(sel)):
            hits = np.where(p_true[i] >= THRESH)[0]
            if len(hits):
                conv[i] = hits[0] + 1
        end_pred = post[sel, :, -1].argmax(1)
        acc = float((end_pred == state[sel]).mean())
        confusion = np.zeros((K, K), dtype=int)
        for s_true, s_pred in zip(state[sel], end_pred):
            confusion[s_true, s_pred] += 1
        ent = -(post[sel] * np.log(post[sel].clip(1e-12))).sum(1)   # (n_sel, T)

        converged = conv <= T
        report["per_strength"][str(m)] = {
            "median_convergence_tokens": int(np.median(conv[converged]))
                if converged.any() else None,
            "q25_q75": [int(np.quantile(conv[converged], q)) for q in (0.25, 0.75)]
                if converged.any() else None,
            "frac_converged_by": {str(c): round(float((conv <= c).mean()), 3)
                                  for c in CHECKPOINTS},
            "end_accuracy": round(acc, 3),
            "mean_entropy_at": {str(t): round(float(ent[:, t - 1].mean()), 3)
                                for t in ENT_TS},
            "max_entropy": round(float(np.log(K)), 3),
            "confusion_end": confusion.tolist(),
        }
        r = report["per_strength"][str(m)]
        print(f"strength {m}x: median conv {r['median_convergence_tokens']} tok, "
              f"conv by 50/100/300: {r['frac_converged_by']['50']}/"
              f"{r['frac_converged_by']['100']}/{r['frac_converged_by']['300']}, "
              f"end acc {acc:.3f}", flush=True)

    (out / "convergence.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
