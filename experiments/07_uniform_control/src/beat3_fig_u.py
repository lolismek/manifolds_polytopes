"""Blog figure: three-student version of exp06's fig_beat3 — circle-plane
projection of the late-first-dwell state centroids (top row) and the
circle-fit permutation test over all 2520 ring orderings (bottom row), for
the ring student, control2 (uniform jumps, exp07) and control1 (unsteered).

Ring/control1 means from exp06 firstdwell_late_arrays.npz; control2 from
exp07 uniform_firstdwell_late_arrays.npz (its own eval docs — geometry is
a property of the student, not the eval set).

Writes results/blog/fig_beat3.png and copies it to ~/Desktop.
"""
import itertools
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
EXP06_PROBE = ROOT.parent / "06_short_dwell" / "results" / "probe"
OUT = ROOT / "results" / "blog"

LAYER = 11
RING = "#c0392b"
CTRL2 = "#2980b9"
CTRL1 = "#7f8c8d"
STATE_COLORS = plt.cm.tab10(np.arange(8))

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def f1_plane_coords(means):
    mu = means - means.mean(0)
    ph = 2 * np.pi * np.arange(8) / 8
    u = (mu * np.cos(ph)[:, None]).sum(0)
    v = (mu * np.sin(ph)[:, None]).sum(0)
    Q, _ = np.linalg.qr(np.stack([u, v], axis=1))
    return mu @ Q


_k = np.arange(8)
_PS = [np.exp(-2j * np.pi * f * (_k[:, None] - _k[None, :]) / 8)
       * (1 if f == 4 else 2) for f in range(1, 5)]


def f1_fraction(G):
    E = np.array([float(np.real((P * G).sum())) for P in _PS])
    return E[0] / E.sum()


orderings = [(0,) + p for p in itertools.permutations(range(1, 8))
             if p[0] < p[-1]]

late6 = np.load(EXP06_PROBE / "firstdwell_late_arrays.npz")
late7 = np.load(ROOT / "results" / "probe"
                / "uniform_firstdwell_late_arrays.npz")

COLUMNS = [
    (late6["ring_means"][LAYER], RING, "student (ring corpus)"),
    (late7["uniform_means"][LAYER], CTRL2, "control2 (uniform jumps)"),
    (late6["ctrl_means"][LAYER], CTRL1, "control1 (unsteered)"),
]

fig, axes = plt.subplots(2, 3, figsize=(13, 8.6))
for col, (M, color, title) in enumerate(COLUMNS):
    # top row: centroids projected onto the fitted circle plane
    ax = axes[0, col]
    pc = f1_plane_coords(M)
    pc = pc / np.sqrt((pc ** 2).sum(1).mean())
    loop = list(range(8)) + [0]
    ax.plot(pc[loop, 0], pc[loop, 1], "-", color="0.75", lw=1.2, zorder=1)
    for k in range(8):
        ax.scatter(*pc[k], s=200, color=STATE_COLORS[k], zorder=2,
                   edgecolors="k", linewidths=0.5)
        ax.annotate(str(k), pc[k], ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white", zorder=3)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("circle plane, axis 1")
    if col == 0:
        ax.set_ylabel("circle plane, axis 2")
    ax.set_aspect("equal")
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.9, 1.9)
    ax.set_xticks([])
    ax.set_yticks([])

    # bottom row: circle-fit variance across all 2520 ring orderings
    ax = axes[1, col]
    X = M - M.mean(0)
    G = X @ X.T
    true_f1 = f1_fraction(G)
    dist = np.array([f1_fraction(G[np.ix_(o, o)]) for o in orderings])
    ax.hist(dist, bins=40, color="0.8", edgecolor="0.6")
    ax.axvline(true_f1, color=color, lw=2.2)
    rank = int((dist >= true_f1).sum())
    ha = "left" if true_f1 < 0.33 else "right"
    pad = "  " if ha == "left" else ""
    tail = "  " if ha == "right" else ""
    ax.text(true_f1, 150, f"{pad}true ring order{tail}\n"
            f"{pad}(rank {rank} of 2520){tail}",
            color=color, fontsize=10, va="top", ha=ha)
    ax.set_xlim(0.15, 0.45)
    ax.set_ylim(0, 160)
    ax.set_xlabel("share of variance explained by circle fit")
    if col == 0:
        ax.set_ylabel("number of orderings")

fig.tight_layout()
fig.savefig(OUT / "fig_beat3.png", bbox_inches="tight")
plt.close(fig)
shutil.copy(OUT / "fig_beat3.png", Path.home() / "Desktop" / "fig_beat3.png")
print("wrote", OUT / "fig_beat3.png", "and copied to Desktop")
