"""Step 5 pilot analysis: exact forward algorithm over the pilot docs,
belief-trajectory metrics per operating point, trajectory plots.

Reader: forward algorithm with the true ring transition matrix (known
p_stay), uniform initial belief, and the exact emission table from
pilot.py --phase score. Under clean-cache steering this posterior is exact.

Metrics per operating point (mean over docs / transitions):
  logloss        mean -log b_t(true state)  (oracle = 0)
  cov90 / cov50  fraction of tokens with belief on the true state > .9 / > .5
  recover_med    median tokens after a transition until belief(new) > .9
  recover_fail   fraction of transitions where .9 is never reached before the
                 next transition (censored)
  bystander      during recovery, median of the max belief on any state other
                 than the old and new one (low = mass moves along the edge)

Run locally: python pilot_analyze.py
"""

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent.parent / "results" / "pilot"
THR = 0.9


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


def doc_metrics(b, z):
    T = len(z)
    m = {"logloss": float(-np.log(np.maximum(b[np.arange(T), z], 1e-30)).mean()),
         "cov90": float((b[np.arange(T), z] > THR).mean()),
         "cov50": float((b[np.arange(T), z] > 0.5).mean())}
    recov, fails, bystander = [], 0, []
    taus = [t for t in range(1, T) if z[t] != z[t - 1]]
    for n, tau in enumerate(taus):
        end = taus[n + 1] if n + 1 < len(taus) else T
        new, old = z[tau], z[tau - 1]
        seg = b[tau:end, new]
        hit = np.argmax(seg > THR) if (seg > THR).any() else None
        if hit is None:
            fails += 1
            continue
        recov.append(int(hit))
        others = [k for k in range(8) if k not in (old, new)]
        bystander.append(float(b[tau:tau + hit + 1, others].max()))
    m["n_trans"] = len(taus)
    m["recover"] = recov
    m["recover_fail"] = fails
    m["bystander"] = bystander
    return m


def main():
    logp, states, p_stay, g = [], [], [], []
    for f in sorted(OUT.glob("pilot_score_shard*.npz")):
        d = np.load(f)
        logp.append(d["logp"]); states.append(d["states"])
        p_stay.append(d["p_stay"]); g.append(d["g"])
    logp = np.concatenate(logp); states = np.concatenate(states)
    p_stay = np.concatenate(p_stay); g = np.concatenate(g)

    beliefs = {}
    summary = {}
    for p in sorted(set(p_stay.tolist())):
        sel = np.where(p_stay == p)[0]
        ms, recov, byst, ntr, nfail = [], [], [], 0, 0
        for i in sel:
            b = forward(logp[i].T, p)          # logp[i] is (8, T)
            beliefs[int(g[i])] = b
            m = doc_metrics(b, states[i])
            ms.append(m)
            recov += m["recover"]; byst += m["bystander"]
            ntr += m["n_trans"]; nfail += m["recover_fail"]
        summary[f"{p:.3f}"] = {
            "dwell_mean": 1.0 / (1.0 - p),
            "n_docs": len(sel), "n_trans": ntr,
            "logloss": float(np.mean([m["logloss"] for m in ms])),
            "cov90": float(np.mean([m["cov90"] for m in ms])),
            "cov50": float(np.mean([m["cov50"] for m in ms])),
            "recover_med": float(np.median(recov)) if recov else None,
            "recover_p90": float(np.percentile(recov, 90)) if recov else None,
            "recover_fail": nfail / max(ntr, 1),
            "bystander_med": float(np.median(byst)) if byst else None,
        }
        s = summary[f"{p:.3f}"]
        print(f"p_stay {p:.3f} (dwell {s['dwell_mean']:.0f}): "
              f"logloss {s['logloss']:.3f}  cov90 {s['cov90']:.2f}  "
              f"cov50 {s['cov50']:.2f}  recover med {s['recover_med']} "
              f"p90 {s['recover_p90']}  fail {s['recover_fail']:.2f}  "
              f"bystander {s['bystander_med']:.2f}  ({ntr} transitions)")

    (OUT / "pilot_summary.json").write_text(json.dumps(summary, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable — skipping plots")
        return
    ps = sorted(set(p_stay.tolist()))
    fig, axes = plt.subplots(len(ps), 2, figsize=(14, 3 * len(ps)),
                             squeeze=False)
    for r, p in enumerate(ps):
        sel = [i for i in range(len(g)) if p_stay[i] == p][:2]
        for c, i in enumerate(sel):
            ax = axes[r][c]
            b, z = beliefs[int(g[i])], states[i]
            ax.plot(b[np.arange(len(z)), z], lw=0.8, color="tab:blue")
            for tau in np.where(np.diff(z) != 0)[0] + 1:
                ax.axvline(tau, color="red", lw=0.4, alpha=0.5)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"p_stay {p:.3f}, doc {int(g[i])} "
                         "(blue = belief on true state, red = transitions)",
                         fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "pilot_trajectories.png", dpi=130)
    print("wrote pilot_trajectories.png")


if __name__ == "__main__":
    main()
