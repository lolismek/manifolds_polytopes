"""8x8 cosine-similarity heatmaps of per-state probe directions (user's
soft-identity check): states in ring order on both axes, so ring structure
= heat leaking from the principal diagonal to the two adjacent diagonals,
wrapping at the corners.

Panels per student (L10, late-first-dwell fits): raw class means, ridge
whitened (lam 1e-3), ridge lam 10 (the circle-shaped probes), logistic.
Reads perclass_late_dirs.npz + firstdwell_late_arrays.npz; writes
results/probe/probe_heatmap.png. Usage: python probe_heatmap.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probe_sweep import ROOT

LAYER = 10
LAM_IDX = {1e-3: 1, 10.0: 5}          # indices into perclass LAM_SCALES


def cos_sim(V):
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    return Vn @ Vn.T


def main():
    dirs = np.load(ROOT / "probe" / "perclass_late_dirs.npz")
    arrs = np.load(ROOT / "probe" / "firstdwell_late_arrays.npz")
    assert abs(dirs["lam_scales"][LAM_IDX[10.0]] - 10.0) < 1e-9

    gd = np.abs(np.arange(8)[:, None] - np.arange(8)[None])
    ring_d = np.minimum(gd, 8 - gd)

    fig, axes = plt.subplots(2, 5, figsize=(23, 9.5))
    for row, student in enumerate(("ring", "ctrl")):
        panels = [
            ("class means (centered)",
             arrs[f"{student}_means"][LAYER]
             - arrs[f"{student}_means"][LAYER].mean(0)),
            ("ridge lam 1e-3 (whitened)",
             dirs[f"{student}_ridge_L{LAYER}"][LAM_IDX[1e-3]]),
            ("ridge lam 10 (circle probes)",
             dirs[f"{student}_ridge_L{LAYER}"][LAM_IDX[10.0]]),
            ("logistic",
             dirs[f"{student}_logistic_L{LAYER}"]),
        ]
        for col, (title, V) in enumerate(panels):
            S = cos_sim(V)
            ax = axes[row, col]
            Sm = S.copy(); np.fill_diagonal(Sm, np.nan)   # diag is trivially 1
            im = ax.imshow(Sm, cmap="RdBu_r", vmin=-0.35, vmax=0.35)
            for i in range(8):
                for j in range(8):
                    if i != j:
                        ax.text(j, i, f"{S[i, j]:.2f}", ha="center",
                                va="center", fontsize=6.5)
            ax.set_xticks(range(8)); ax.set_yticks(range(8))
            ax.set_title(f"{student} L{LAYER} {title}", fontsize=9)
            # curve panel: mean cosine at each ring distance
            axc = axes[row, 4]
            curve = [S[ring_d == r].mean() for r in (1, 2, 3, 4)]
            axc.plot((1, 2, 3, 4), curve, marker="o", label=title)
        axc.axhline(-1 / 7, color="0.7", lw=0.8)
        axc.text(3.1, -1 / 7 + 0.005, "baseline -1/7", fontsize=7, color="0.4")
        axc.set_xticks((1, 2, 3, 4)); axc.set_xlabel("ring distance")
        axc.set_ylabel("mean cosine")
        axc.set_title(f"{student}: cosine vs ring distance", fontsize=9)
        axc.legend(fontsize=7)
    fig.colorbar(im, ax=axes[:, :4], shrink=0.6,
                 label="cosine similarity (diagonal masked)")
    fig.suptitle("per-state probe directions, pairwise cosine (states in "
                 "ring order; first-dwell fits)")
    fig.savefig(ROOT / "probe" / "probe_heatmap.png", dpi=150,
                bbox_inches="tight")
    print("wrote probe_heatmap.png", flush=True)


if __name__ == "__main__":
    main()
