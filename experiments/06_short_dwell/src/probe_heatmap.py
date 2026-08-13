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

    fig, axes = plt.subplots(2, 4, figsize=(19, 9.5))
    for row, student in enumerate(("ring", "ctrl")):
        panels = [
            (f"{student} L{LAYER} class means (centered)",
             arrs[f"{student}_means"][LAYER]
             - arrs[f"{student}_means"][LAYER].mean(0)),
            (f"{student} L{LAYER} ridge lam 1e-3 (whitened)",
             dirs[f"{student}_ridge_L{LAYER}"][LAM_IDX[1e-3]]),
            (f"{student} L{LAYER} ridge lam 10 (circle probes)",
             dirs[f"{student}_ridge_L{LAYER}"][LAM_IDX[10.0]]),
            (f"{student} L{LAYER} logistic",
             dirs[f"{student}_logistic_L{LAYER}"]),
        ]
        for col, (title, V) in enumerate(panels):
            S = cos_sim(V)
            ax = axes[row, col]
            im = ax.imshow(S, cmap="RdBu_r", vmin=-1, vmax=1)
            for i in range(8):
                for j in range(8):
                    ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center",
                            fontsize=6.5,
                            color="white" if abs(S[i, j]) > 0.6 else "black")
            ax.set_xticks(range(8)); ax.set_yticks(range(8))
            ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.6, label="cosine similarity")
    fig.suptitle("per-state probe directions, pairwise cosine (states in "
                 "ring order; first-dwell fits)")
    fig.savefig(ROOT / "probe" / "probe_heatmap.png", dpi=150,
                bbox_inches="tight")
    print("wrote probe_heatmap.png", flush=True)


if __name__ == "__main__":
    main()
