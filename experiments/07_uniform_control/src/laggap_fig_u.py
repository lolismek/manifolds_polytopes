"""Blog figure: layer-12 probe quality vs EXACT tokens-since-switch
(lag 0, 1, 2, ...), ring student vs control2 (cross), from the belief_viz
dumps. Left: probe R2 vs the exact posterior. Right: argmax accuracy vs
the true state, with the optimal observer's per-lag accuracy for scale.
First-dwell tokens (no preceding switch) are excluded throughout.

R2 at each lag is computed against that lag's own posterior variance.
Lags are truncated where fewer than MIN_N test tokens remain.

Writes results/blog/fig_laggap.png and copies it to ~/Desktop.
"""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
EXP06_PROBE = ROOT.parent / "06_short_dwell" / "results" / "probe"
OUT = ROOT / "results" / "blog"

RING = "#c0392b"
CTRL2 = "#2980b9"
READER = "#2c3e50"
MIN_N = 800

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

d6 = np.load(EXP06_PROBE / "belief_viz.npz")
d7 = np.load(ROOT / "results" / "probe" / "uniform_belief_viz_cross.npz")
b = d6["b"].astype(np.float32)
z = d6["z"].astype(int)
bh = {"ring": d6["bhat"].astype(np.float32),
      "ctrl2": d7["bhat"].astype(np.float32)}
n, T, K = b.shape

lag = np.full((n, T), -1, dtype=np.int64)   # -1 = first dwell, excluded
for i in range(n):
    sw = np.nonzero(np.diff(z[i]) != 0)[0] + 1
    for s0, s1 in zip(sw, list(sw[1:]) + [T]):
        lag[i, s0:s1] = np.arange(s1 - s0)

lagf = lag.reshape(-1)
bf = b.reshape(-1, K)
zf = z.reshape(-1)

max_lag = 0
while ((lagf == max_lag).sum() >= MIN_N):
    max_lag += 1
lags = np.arange(max_lag)
print(f"plotting lags 0..{max_lag - 1} "
      f"(>= {MIN_N} tokens each; lag 0 has {(lagf == 0).sum()})")

curves = {}
for name in ("ring", "ctrl2"):
    bhf = bh[name].reshape(-1, K)
    r2s, accs = [], []
    for k in lags:
        m = lagf == k
        y, yh = bf[m], bhf[m]
        r2s.append(1 - ((y - yh) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())
        accs.append((yh.argmax(1) == zf[m]).mean())
    curves[name] = (np.array(r2s), np.array(accs))
reader_acc = np.array([(bf[lagf == k].argmax(1) == zf[lagf == k]).mean()
                       for k in lags])

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

ax = axes[0]
ax.plot(lags, curves["ring"][0], "-", color=RING, lw=1.8,
        label="student (ring corpus)")
ax.plot(lags, curves["ctrl2"][0], "-", color=CTRL2, lw=1.8,
        label="control2 (uniform jumps)")
ax.set_xlabel(r"tokens since last state switch  $\tau$")
ax.set_ylabel(r"probe $R^2$ vs exact posterior $b_t$  (layer 12)")
ax.set_ylim(0, 0.7)
ax.legend(frameon=False, fontsize=9, loc="lower right")

ax = axes[1]
ax.plot(lags, reader_acc, "-", color=READER, lw=1.8,
        label="optimal observer (exact Bayes)")
ax.plot(lags, curves["ring"][1], "-", color=RING, lw=1.8,
        label="student (ring corpus)")
ax.plot(lags, curves["ctrl2"][1], "-", color=CTRL2, lw=1.8,
        label="control2 (uniform jumps)")
ax.axhline(1 / K, color="k", lw=0.8, ls=":")
ax.text(1, 1 / K + 0.02, "chance (1/8)", fontsize=9)
ax.set_xlabel(r"tokens since last state switch  $\tau$")
ax.set_ylabel(r"argmax accuracy vs true state $z_t$")
ax.set_ylim(0, 1.0)
leg = ax.legend(frameon=True, fontsize=9, loc="lower right")
leg.get_frame().set_edgecolor("none")
leg.get_frame().set_alpha(1.0)

fig.tight_layout()
fig.savefig(OUT / "fig_laggap.png", bbox_inches="tight")
plt.close(fig)
shutil.copy(OUT / "fig_laggap.png", Path.home() / "Desktop" / "fig_laggap.png")
print("wrote", OUT / "fig_laggap.png", "and copied to Desktop")
