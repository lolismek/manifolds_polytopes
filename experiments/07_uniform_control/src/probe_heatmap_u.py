"""8x8 cosine heatmaps of per-state directions for the exp07 uniform
student (exp06's probe_heatmap design): states in cast-ring order, ring
structure = warmth on the two diagonals adjacent to the principal one.
exp06's ring student shows the soft-identity band at lam=10 and its
inversion when whitened; prediction here: flat (no band, either sign).

Reads uniform_perclass_late_dirs.npz + uniform_firstdwell_late_arrays.npz;
writes results/probe/uniform_probe_heatmap.png.
Usage: python probe_heatmap_u.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probe_sweep_u import ROOT

LAYER = 10
LAM_IDX = {1e-3: 1, 10.0: 5}          # indices into perclass LAM_SCALES


def cos_sim(V):
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    return Vn @ Vn.T


def main():
    dirs = np.load(ROOT / "probe" / "uniform_perclass_late_dirs.npz")
    arrs = np.load(ROOT / "probe" / "uniform_firstdwell_late_arrays.npz")
    assert abs(dirs["lam_scales"][LAM_IDX[10.0]] - 10.0) < 1e-9

    gd = np.abs(np.arange(8)[:, None] - np.arange(8)[None])
    ring_d = np.minimum(gd, 8 - gd)

    fig, axes = plt.subplots(1, 5, figsize=(23, 5))
    panels = [
        ("class means (centered)",
         arrs["uniform_means"][LAYER]
         - arrs["uniform_means"][LAYER].mean(0)),
        ("ridge lam 1e-3 (whitened)",
         dirs[f"uniform_ridge_L{LAYER}"][LAM_IDX[1e-3]]),
        ("ridge lam 10",
         dirs[f"uniform_ridge_L{LAYER}"][LAM_IDX[10.0]]),
        ("logistic",
         dirs[f"uniform_logistic_L{LAYER}"]),
    ]
    for col, (title, V) in enumerate(panels):
        S = cos_sim(V)
        ax = axes[col]
        Sm = S.copy(); np.fill_diagonal(Sm, np.nan)   # diag is trivially 1
        im = ax.imshow(Sm, cmap="RdBu_r", vmin=-0.35, vmax=0.35)
        for i in range(8):
            for j in range(8):
                if i != j:
                    ax.text(j, i, f"{S[i, j]:.2f}", ha="center",
                            va="center", fontsize=6.5)
        ax.set_xticks(range(8)); ax.set_yticks(range(8))
        ax.set_title(f"uniform L{LAYER} {title}", fontsize=9)
        axc = axes[4]
        curve = [S[ring_d == r].mean() for r in (1, 2, 3, 4)]
        axc.plot((1, 2, 3, 4), curve, marker="o", label=title)
    axc = axes[4]
    axc.axhline(-1 / 7, color="0.7", lw=0.8)
    axc.text(3.1, -1 / 7 + 0.005, "baseline -1/7", fontsize=7, color="0.4")
    axc.set_xticks((1, 2, 3, 4)); axc.set_xlabel("cast ring distance")
    axc.set_ylabel("mean cosine")
    axc.set_title("uniform: cosine vs cast-ring distance", fontsize=9)
    axc.legend(fontsize=7)
    fig.colorbar(im, ax=axes[:4], shrink=0.6,
                 label="cosine similarity (diagonal masked)")
    fig.suptitle("exp07 uniform student: per-state directions, pairwise "
                 "cosine (states in CAST ring order; first-dwell fits)")
    fig.savefig(ROOT / "probe" / "uniform_probe_heatmap.png", dpi=150,
                bbox_inches="tight")
    print("wrote uniform_probe_heatmap.png", flush=True)


if __name__ == "__main__":
    main()
