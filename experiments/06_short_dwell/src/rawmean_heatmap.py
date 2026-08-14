"""8x8 cosine heatmap of the BARE (uncentered) class centroids at L10 —
no difference-in-means, no whitening, just the raw average hidden state per
ring state, unit-normed. States in ring order on both axes; diagonal masked
(trivially 1). All cosines are positive here because every centroid shares
the big common activation component, so the color scale is sequential.

Reads results/probe/firstdwell_late_arrays.npz;
writes results/probe/rawmean_heatmap.png. Usage: python rawmean_heatmap.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probe_sweep import ROOT

LAYER = 10


def main():
    arrs = np.load(ROOT / "probe" / "firstdwell_late_arrays.npz")
    gd = np.abs(np.arange(8)[:, None] - np.arange(8)[None])
    ring_d = np.minimum(gd, 8 - gd)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2),
                             gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for col, student in enumerate(("ring", "ctrl")):
        M = arrs[f"{student}_means"][LAYER]
        Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
        S = Mn @ Mn.T
        ax = axes[col]
        Sm = S.copy(); np.fill_diagonal(Sm, np.nan)
        im = ax.imshow(Sm, cmap="viridis", vmin=0.2, vmax=0.95)
        for i in range(8):
            for j in range(8):
                if i != j:
                    ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center",
                            fontsize=7.5,
                            color="w" if S[i, j] < 0.6 else "k")
        ax.set_xticks(range(8)); ax.set_yticks(range(8))
        ax.set_title(f"{student} L{LAYER} bare centroids", fontsize=10)
        curve = [S[ring_d == r].mean() for r in (1, 2, 3, 4)]
        axes[2].plot((1, 2, 3, 4), curve, marker="o", label=student)
    axes[2].set_xticks((1, 2, 3, 4)); axes[2].set_xlabel("ring distance")
    axes[2].set_ylabel("mean cosine")
    axes[2].set_title("cosine vs ring distance", fontsize=10)
    axes[2].legend(fontsize=8)
    fig.colorbar(im, ax=axes[:2], shrink=0.8, label="cosine similarity")
    fig.suptitle("bare (uncentered) late-first-dwell centroids, pairwise "
                 "cosine — states in ring order")
    fig.savefig(ROOT / "probe" / "rawmean_heatmap.png", dpi=150,
                bbox_inches="tight")
    print("wrote rawmean_heatmap.png", flush=True)


if __name__ == "__main__":
    main()
