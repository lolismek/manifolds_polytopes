"""Belief-geometry figure for the step-5 pilot (p_stay 0.99 docs).

Left: PCA of the reader's 8-dim belief vectors, colored by true state, with
the 8 certainty vertices projected in. Middle: ring map — each belief mapped
to sum_k b_k * exp(2*pi*i*k/8); angle = ring position, radius = confidence.
Right: one document's trajectory on the ring map.

Writes results/pilot/belief_geometry.png (plus optional --out copy).
Run locally: python pilot_geometry.py [--out ~/Desktop/exp04_belief_geometry.png]
"""

import argparse
import glob
import os
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pilot_analyze import forward

RING = [1309, 2970, 14600, 10573, 6455, 15026, 3188, 5615]
P_STAY = 0.99
OUT = Path(__file__).parent.parent / "results" / "pilot"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    logp, states, p_stay = [], [], []
    for f in sorted(glob.glob(str(OUT / "pilot_score_shard*.npz"))):
        d = np.load(f)
        logp.append(d["logp"]); states.append(d["states"])
        p_stay.append(d["p_stay"])
    logp = np.concatenate(logp); states = np.concatenate(states)
    p_stay = np.concatenate(p_stay)

    sel = np.where(np.isclose(p_stay, P_STAY))[0]
    B_list = [forward(logp[i].T, P_STAY) for i in sel]
    Z_list = [states[i].astype(int) for i in sel]
    Bel = np.concatenate(B_list)
    Z = np.concatenate(Z_list)

    X = Bel - Bel.mean(0)
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    P = X @ Vt[:2].T
    verts = (np.eye(8) - Bel.mean(0)) @ Vt[:2].T

    ang = 2 * np.pi * np.arange(8) / 8
    m = Bel @ np.exp(1j * ang)
    r = np.abs(m)
    err = np.angle(m * np.exp(-1j * ang[Z]))
    print(f"median radius {np.median(r):.3f}  frac>0.7 {(r > 0.7).mean():.3f}"
          f"  median |angle err| {np.median(np.abs(err)) / (2 * np.pi / 8):.2f}"
          " ring steps")

    cmap = plt.get_cmap("hsv")
    cols = [cmap(k / 8) for k in range(8)]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))

    ax = axes[0]
    idx = np.arange(0, len(P), 2)
    ax.scatter(P[idx, 0], P[idx, 1], c=[cols[z] for z in Z[idx]], s=3,
               alpha=0.35, lw=0)
    for k in range(8):
        ax.scatter(*verts[k], c=[cols[k]], s=260, marker="*",
                   edgecolors="black", zorder=5)
        ax.annotate(str(RING[k]), verts[k], fontsize=9, fontweight="bold",
                    xytext=(6, 6), textcoords="offset points")
    ax.set_title("Reader beliefs, PCA to 2D\n"
                 "(stars = full certainty in each state; color = true state)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    ax = axes[1]
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="gray", lw=1,
                            ls="--"))
    ax.scatter(m.real[idx], m.imag[idx], c=[cols[z] for z in Z[idx]], s=3,
               alpha=0.35, lw=0)
    for k in range(8):
        v = np.exp(1j * ang[k])
        ax.scatter(v.real, v.imag, c=[cols[k]], s=260, marker="*",
                   edgecolors="black", zorder=5)
        ax.annotate(str(RING[k]), (v.real, v.imag), fontsize=9,
                    fontweight="bold", xytext=(6, 6),
                    textcoords="offset points")
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.set_title("Ring map: angle = where on the ring, radius = how sure\n"
                 "(center = total uncertainty, dashed circle = certainty)")

    ax = axes[2]
    b, z = B_list[0], Z_list[0]
    mm = b @ np.exp(1j * ang)
    t0, t1 = 150, 450
    seg = mm[t0:t1]
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="gray", lw=1,
                            ls="--"))
    ax.plot(seg.real, seg.imag, "-", lw=0.7, color="gray", alpha=0.7)
    ax.scatter(seg.real, seg.imag, c=[cols[k] for k in z[t0:t1]], s=14,
               zorder=4)
    for k in range(8):
        v = np.exp(1j * ang[k])
        ax.scatter(v.real, v.imag, c=[cols[k]], s=260, marker="*",
                   edgecolors="black", zorder=5)
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.set_title(f"One document's journey (tokens {t0}-{t1})\n"
                 "(color = true state at that moment)")

    fig.suptitle("exp04 pilot — the ideal reader's belief geometry "
                 f"(p_stay = {P_STAY}, {len(sel)} docs x 1024 tokens)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "belief_geometry.png", dpi=150)
    if args.out:
        fig.savefig(os.path.expanduser(args.out), dpi=150)
    print("saved", OUT / "belief_geometry.png")


if __name__ == "__main__":
    main()
