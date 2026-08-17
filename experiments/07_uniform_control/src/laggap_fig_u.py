"""Blog figure: layer-12 probe R2 vs tokens-since-switch, ring student vs
control2 (cross), computed locally from the belief_viz dumps. Shows that
the decoding gap is largest right after a switch (where the ring map
narrows the destination to 2 candidates) and decays as evidence
accumulates; first-dwell tokens (no preceding switch) shown as a separate
leftmost bin where the gap vanishes.

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
BINS = [(0, 4), (5, 9), (10, 19), (20, 34), (35, 49), (50, 10 ** 9)]
LABELS = ["0-4", "5-9", "10-19", "20-34", "35-49", "50+"]

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
n, T, K = b.shape

lag = np.full((n, T), -1, dtype=np.int64)   # -1 = first dwell (no switch yet)
for i in range(n):
    sw = np.nonzero(np.diff(z[i]) != 0)[0] + 1
    for s0, s1 in zip(sw, list(sw[1:]) + [T]):
        lag[i, s0:s1] = np.arange(s1 - s0)


def r2(bh, m):
    bf, bhf = b.reshape(-1, K)[m], bh.reshape(-1, K)[m]
    return 1 - ((bf - bhf) ** 2).sum() / ((bf - bf.mean(0)) ** 2).sum()


masks = [(lag == -1).reshape(-1)] + \
    [((lag >= lo) & (lag <= hi)).reshape(-1) for lo, hi in BINS]
xticklabels = ["first\ndwell"] + LABELS
x = np.arange(len(masks))

fig, ax = plt.subplots(figsize=(6.2, 3.8))
for d, color, label in [(d6["bhat"], RING, "student (ring corpus)"),
                        (d7["bhat"], CTRL2, "control2 (uniform jumps)")]:
    ys = [r2(d.astype(np.float32), m) for m in masks]
    ax.plot(x[1:], ys[1:], "o-", color=color, ms=6, label=label)
    ax.plot(x[0], ys[0], "o", color=color, ms=6, mfc="white")
ax.axvline(0.5, color="0.85", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(xticklabels)
ax.set_xlabel(r"tokens since last state switch  $\tau$")
ax.set_ylabel(r"probe $R^2$ vs exact posterior $b_t$  (layer 12)")
ax.set_ylim(0, 0.65)
leg = ax.legend(frameon=True, fontsize=9, loc="lower right")
leg.get_frame().set_edgecolor("none")
fig.tight_layout()
fig.savefig(OUT / "fig_laggap.png", bbox_inches="tight")
plt.close(fig)
shutil.copy(OUT / "fig_laggap.png", Path.home() / "Desktop" / "fig_laggap.png")
print("wrote", OUT / "fig_laggap.png", "and copied to Desktop")
