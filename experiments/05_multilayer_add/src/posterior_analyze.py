"""Readout of the exact-reader posteriors on the eval split: is per-token
accuracy limited only by inherent transition lag (healthy) or by mid-dwell
stalling (exp04's failure mode)?

Bins every token by lag = tokens since the last state change (doc start
counts as a change: the reader starts uniform), reporting mean belief in the
true state, argmax accuracy, and logloss per bin. Run on the machine holding
results/posterior/: python posterior_analyze.py
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent / "results" / "posterior"
BINS = [(0, 5), (5, 10), (10, 20), (20, 35), (35, 50), (50, 10 ** 9)]


def main():
    files = sorted(ROOT.glob("eval_shard*.npz"))
    print(f"{len(files)} shards")
    bt, acc_n, tot, ll = (np.zeros(len(BINS)) for _ in range(4))
    logloss_all, n_all = 0.0, 0
    for f in files:
        d = np.load(f)
        z = d["z"].astype(np.int64)                  # (n, T)
        post = d["posterior"].astype(np.float64)     # (n, T, 8)
        n, T = z.shape
        # lag since last transition; doc start = lag 0
        lag = np.zeros((n, T), dtype=np.int64)
        same = z[:, 1:] == z[:, :-1]
        for t in range(1, T):
            lag[:, t] = np.where(same[:, t - 1], lag[:, t - 1] + 1, 0)
        p_true = np.take_along_axis(post, z[..., None], 2)[..., 0]
        p_true = np.clip(p_true, 1e-8, 1.0)
        hit = post.argmax(-1) == z
        logloss_all += -np.log(p_true).sum()
        n_all += z.size
        for b, (lo, hi) in enumerate(BINS):
            m = (lag >= lo) & (lag < hi)
            bt[b] += p_true[m].sum()
            acc_n[b] += hit[m].sum()
            ll[b] += -np.log(p_true[m]).sum()
            tot[b] += m.sum()
    print(f"overall: acc {acc_n.sum()/tot.sum():.3f}  "
          f"logloss {logloss_all/n_all:.3f} (uniform {np.log(8):.3f})")
    print("lag bin   frac   belief(true)  acc    logloss")
    for b, (lo, hi) in enumerate(BINS):
        name = f"{lo}-{hi - 1}" if hi < 10 ** 9 else f"{lo}+"
        print(f"{name:>8} {tot[b]/tot.sum():6.3f}   {bt[b]/tot[b]:.3f}"
              f"        {acc_n[b]/tot[b]:.3f}  {ll[b]/tot[b]:.3f}")


if __name__ == "__main__":
    main()
