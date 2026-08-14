"""Project late-first-dwell class means into their own frequency-1 (ring)
plane and scatter the 8 states (user's 'something cleaner from the class
means').

The plane is built from the known ring order (cos/sin weights over states),
so this is a visualization of the circle whose SIGNIFICANCE was already
established order-free by the Fourier permutation test (p ~ 0.001); the
picture itself answers: once you look at that plane, are the states in ring
order and roughly evenly spaced? Ctrl row = same construction on the
control student (its plane is meaningless, so the layout should scramble).

Reads results/probe/firstdwell_late_arrays.npz;
writes results/probe/ringplane_means.png. Usage: python ringplane_viz.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probe_sweep import ROOT

K = 8
LAYERS = (9, 10, 11, 12)


def plane_coords(M):
    k = np.arange(K)
    X = M - M.mean(0)                    # centered = difference-in-means
    u = np.cos(2 * np.pi * k / K) @ X
    v = np.sin(2 * np.pi * k / K) @ X
    u /= np.linalg.norm(u)
    v -= (v @ u) * u
    v /= np.linalg.norm(v)
    P = np.stack([X @ u, X @ v], 1)
    return P, float((P ** 2).sum() / (X ** 2).sum())


def main():
    arrs = np.load(ROOT / "probe" / "firstdwell_late_arrays.npz")
    fig, axes = plt.subplots(2, len(LAYERS), figsize=(4.2 * len(LAYERS), 8.6))
    for row, student in enumerate(("ring", "ctrl")):
        for col, l in enumerate(LAYERS):
            P, frac = plane_coords(arrs[f"{student}_means"][l])
            ax = axes[row, col]
            loop = np.r_[np.arange(K), 0]
            ax.plot(P[loop, 0], P[loop, 1], "-", color="0.8", lw=1, zorder=1)
            ax.scatter(P[:, 0], P[:, 1], c=np.arange(K), cmap="hsv",
                       s=90, zorder=2)
            for s in range(K):
                ax.annotate(str(s), P[s], textcoords="offset points",
                            xytext=(7, 5), fontsize=11)
            ax.axhline(0, color="0.92", lw=0.8); ax.axvline(0, color="0.92",
                                                            lw=0.8)
            ax.set_aspect("equal")
            ax.set_title(f"{student} L{l}  (plane holds {frac:.0%} of "
                         "mean energy)", fontsize=10)
    fig.suptitle("late-first-dwell class means projected into their "
                 "frequency-1 ring plane\n(gray polygon connects states in "
                 "true ring order 0-1-...-7-0)")
    fig.tight_layout()
    fig.savefig(ROOT / "probe" / "ringplane_means.png", dpi=150)
    print("wrote ringplane_means.png", flush=True)


if __name__ == "__main__":
    main()
