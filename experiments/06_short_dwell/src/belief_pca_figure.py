"""Blog figure: PCA of the optimal observer's posteriors b_t (left) and of
the probe's read-out b_hat (right), points colored by the true state z_t.
The 8 numbered circles mark where a fully confident belief (one-hot on state
k) lands in each panel's PC plane.

Reads results/probe/belief_viz.npz (dump_belief_viz.py on the GPU box).
Writes results/blog/fig_belief_pca.png and copies it to ~/Desktop.
"""
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "..", "results", "probe")
OUT = os.path.join(HERE, "..", "results", "blog")

K = 8
N_SCATTER = 20000
STATE_COLORS = plt.cm.tab10(np.arange(K))

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def pca_scatter(ax, B, zf, title):
    mu = B.mean(0)
    Bc = B - mu
    _, _, Vt = np.linalg.svd(Bc[::10], full_matrices=False)
    pc = Bc @ Vt[:2].T
    pc /= np.sqrt((pc ** 2).sum(1).mean())        # RMS radius = 1: shape only

    rng = np.random.default_rng(0)
    idx = rng.choice(len(pc), N_SCATTER, replace=False)
    ax.scatter(pc[idx, 0], pc[idx, 1], s=3, c=STATE_COLORS[zf[idx]],
               alpha=0.35, linewidths=0, rasterized=True)

    vert = (np.eye(K) - mu) @ Vt[:2].T
    vert /= np.sqrt((((B - mu) @ Vt[:2].T) ** 2).sum(1).mean())
    for k in range(K):
        ax.scatter(*vert[k], s=200, color=STATE_COLORS[k], zorder=3,
                   edgecolors="k", linewidths=0.5)
        ax.annotate(str(k), vert[k], ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white", zorder=4)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("PC 1")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    lim = 1.05 * np.abs(vert).max()
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)


def main():
    d = np.load(os.path.join(PROBE, "belief_viz.npz"))
    b = d["b"].astype(np.float32).reshape(-1, K)
    bhat = d["bhat"].astype(np.float32).reshape(-1, K)
    zf = d["z"].astype(int).reshape(-1)
    layer = int(d["layer"])

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    pca_scatter(axes[0], b, zf, r"optimal observer  $b_t$")
    pca_scatter(axes[1], bhat, zf,
                rf"student probe read-out  $\hat{{b}}_t$  (layer {layer})")
    axes[0].set_ylabel("PC 2")
    fig.tight_layout()
    path = os.path.join(OUT, "fig_belief_pca.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    shutil.copy(path, os.path.expanduser("~/Desktop/fig_belief_pca.png"))
    print("wrote", path, "and copied to Desktop")

    # standalone observer panel for the Method section (no student spoiler)
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    pca_scatter(ax, b, zf, r"optimal observer  $b_t$")
    ax.set_ylabel("PC 2")
    fig.tight_layout()
    path = os.path.join(OUT, "fig_belief_pca_observer.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    shutil.copy(path,
                os.path.expanduser("~/Desktop/fig_belief_pca_observer.png"))
    print("wrote", path, "and copied to Desktop")


if __name__ == "__main__":
    main()
