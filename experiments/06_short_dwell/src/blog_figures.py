"""Figures for the blog post, Part 1 (belief tracking).

Reads results/probe/{ring,ctrl}_{sweep,concat}.json, writes results/blog/.
fig1: per-layer probe R2 and argmax accuracy, ring vs control student.
fig2: argmax accuracy vs tokens-since-switch, students vs optimal observer.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "..", "results", "probe")
OUT = os.path.join(HERE, "..", "results", "blog")
os.makedirs(OUT, exist_ok=True)

RING = "#c0392b"
CTRL = "#7f8c8d"
READER = "#2c3e50"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load(name):
    with open(os.path.join(PROBE, name)) as f:
        return json.load(f)


ring_sweep = load("ring_sweep.json")
ctrl_sweep = load("ctrl_sweep.json")
ring_concat = load("ring_concat.json")
ctrl_concat = load("ctrl_concat.json")

reader_acc = ring_sweep["ideal_reader_acc"]
chance = ring_sweep["chance_acc"]

# ---- fig 1: per-layer R2 and accuracy, final checkpoints ----
ring_layers = ring_sweep["checkpoints"][-1]["layers"]
ctrl_layers = ctrl_sweep["checkpoints"][-1]["layers"]
L = [d["layer"] for d in ring_layers]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

ax = axes[0]
ax.plot(L, [d["probe_R2"] for d in ring_layers], "o-", color=RING, label="student (steered corpus)")
ax.plot(L, [d["probe_R2"] for d in ctrl_layers], "s--", color=CTRL, label="control student")
ax.set_xlabel("layer")
ax.set_ylabel(r"probe $R^2$ vs exact posterior $b_t$")
ax.set_ylim(0, 0.6)
ax.legend(frameon=False, fontsize=9)

ax = axes[1]
ax.plot(L, [d["acc_vs_true_state"] for d in ring_layers], "o-", color=RING, label="student (steered corpus)")
ax.plot(L, [d["acc_vs_true_state"] for d in ctrl_layers], "s--", color=CTRL, label="control student")
ax.axhline(reader_acc, color=READER, lw=1, ls=":")
ax.text(0.1, reader_acc + 0.015, "optimal observer (0.762)", fontsize=9, color=READER)
ax.axhline(chance, color="k", lw=0.8, ls=":")
ax.text(0.1, chance + 0.015, "chance (1/8)", fontsize=9)
ax.set_xlabel("layer")
ax.set_ylabel(r"argmax accuracy vs true state $z_t$")
ax.set_ylim(0, 0.85)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_tracking_layers.png"), bbox_inches="tight")
plt.close(fig)

# ---- fig 2: accuracy vs tokens-since-switch ----
bins = ring_concat["lag_bins"]
x = range(len(bins))

fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.plot(x, ring_concat["reader_acc_by_lag"], "o-", color=READER, label="optimal observer (exact Bayes)")
ax.plot(x, ring_concat["acc_by_lag"], "o-", color=RING, label="student (steered corpus)")
ax.plot(x, ctrl_concat["acc_by_lag"], "s--", color=CTRL, label="control student")
ax.axhline(chance, color="k", lw=0.8, ls=":")
ax.text(-0.3, chance + 0.02, "chance (1/8)", fontsize=9)
ax.set_xticks(list(x))
ax.set_xticklabels(bins)
ax.set_xlabel(r"tokens since last state switch  $\tau$")
ax.set_ylabel(r"argmax accuracy vs true state $z_t$")
ax.set_ylim(0, 1.0)
leg = ax.legend(frameon=True, fontsize=9, loc="lower right")
leg.get_frame().set_edgecolor("none")
leg.get_frame().set_alpha(1.0)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_tracking_lag.png"), bbox_inches="tight")
plt.close(fig)

# ---- fig 2b: same, without the control student ----
fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.plot(x, ring_concat["reader_acc_by_lag"], "o-", color=READER, label="optimal observer (exact Bayes)")
ax.plot(x, ring_concat["acc_by_lag"], "o-", color=RING, label="student")
ax.axhline(chance, color="k", lw=0.8, ls=":")
ax.text(-0.3, chance + 0.02, "chance (1/8)", fontsize=9)
ax.set_xticks(list(x))
ax.set_xticklabels(bins)
ax.set_xlabel(r"tokens since last state switch  $\tau$")
ax.set_ylabel(r"argmax accuracy vs true state $z_t$")
ax.set_ylim(0, 1.0)
leg = ax.legend(frameon=True, fontsize=9, loc="lower right")
leg.get_frame().set_edgecolor("none")
leg.get_frame().set_alpha(1.0)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_tracking_lag_nocontrol.png"), bbox_inches="tight")
plt.close(fig)

# ---- fig 3: first-dwell control (carryover ruled out) ----
fd = load("firstdwell.json")
fd_ring = [d["probe_R2"] for d in fd["ring"]["layers"]]
fd_ctrl = [d["probe_R2"] for d in fd["ctrl"]["layers"]]

fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.plot(L, [d["probe_R2"] for d in ring_layers], "o-", color=RING, label="student, all tokens")
ax.plot(L, fd_ring, "o--", color=RING, mfc="white", label="student, first dwell only")
ax.plot(L, [d["probe_R2"] for d in ctrl_layers], "s-", color=CTRL, label="control, all tokens")
ax.plot(L, fd_ctrl, "s--", color=CTRL, mfc="white", label="control, first dwell only")
ax.set_xlabel("layer")
ax.set_ylabel(r"probe $R^2$ vs exact posterior $b_t$")
ax.set_ylim(0, 0.6)
leg = ax.legend(frameon=True, fontsize=9, loc="upper left")
leg.get_frame().set_edgecolor("none")
leg.get_frame().set_alpha(1.0)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_firstdwell.png"), bbox_inches="tight")
plt.close(fig)

print("wrote", os.path.join(OUT, "fig_tracking_layers.png"))
print("wrote", os.path.join(OUT, "fig_tracking_lag.png"))
print("wrote", os.path.join(OUT, "fig_tracking_lag_nocontrol.png"))
print("wrote", os.path.join(OUT, "fig_firstdwell.png"))

# ================= Part 2: ring geometry figures (layer 11) =================
import itertools

import numpy as np

LAYER = 11
STATE_COLORS = plt.cm.tab10(np.arange(8))


def pca2(means):
    mc = means - means.mean(0)
    U, S, _ = np.linalg.svd(mc, full_matrices=False)
    return U[:, :2] * S[:2]


def pca_panel(ax, means, title):
    pc = pca2(means)
    pc = pc / np.sqrt((pc ** 2).sum(1).mean())   # RMS radius = 1: shape only
    loop = list(range(8)) + [0]
    ax.plot(pc[loop, 0], pc[loop, 1], "-", color="0.75", lw=1.2, zorder=1)
    for k in range(8):
        ax.scatter(*pc[k], s=200, color=STATE_COLORS[k], zorder=2,
                   edgecolors="k", linewidths=0.5)
        ax.annotate(str(k), pc[k], ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white", zorder=3)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_aspect("equal")
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.9, 1.9)
    ax.set_xticks([])
    ax.set_yticks([])


def pca_figure(ring_means, ctrl_means, ring_title, ctrl_title, fname):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    pca_panel(axes[0], ring_means, ring_title)
    pca_panel(axes[1], ctrl_means, ctrl_title)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(OUT, fname))


# ---- fig 4 (beat 1): mid-dwell class means PCA (needs GPU dump) ----
mid_path = os.path.join(PROBE, "middwell_means.npz")
if os.path.exists(mid_path):
    mid = np.load(mid_path)
    pca_figure(mid["ring_means"][LAYER], mid["ctrl_means"][LAYER],
               "student (steered corpus)", "control student",
               "fig_pca_middwell.png")
else:
    print("SKIP fig_pca_middwell.png — run dump_middwell_means.py on the "
          "GPU box first")

# ---- fig 5 (beat 2): late-first-dwell class means PCA ----
late = np.load(os.path.join(PROBE, "firstdwell_late_arrays.npz"))
pca_figure(late["ring_means"][LAYER], late["ctrl_means"][LAYER],
           "student (steered corpus)", "control student",
           "fig_pca_firstdwell_late.png")

# ---- fig 6 (beat 3): Fourier ring-mode permutation histogram ----
_k = np.arange(8)
_P1 = 2 * np.exp(-2j * np.pi * (_k[:, None] - _k[None, :]) / 8)
_PS = [np.exp(-2j * np.pi * f * (_k[:, None] - _k[None, :]) / 8)
       * (1 if f == 4 else 2) for f in range(1, 5)]


def f1_fraction(G):
    E = np.array([float(np.real((P * G).sum())) for P in _PS])
    return E[0] / E.sum()


orderings = [(0,) + p for p in itertools.permutations(range(1, 8))
             if p[0] < p[-1]]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True, sharey=True)
for ax, student, color, title in [
        (axes[0], "ring", RING, "student (steered corpus)"),
        (axes[1], "ctrl", CTRL, "control student")]:
    M = late[f"{student}_means"][LAYER]
    X = M - M.mean(0)
    G = X @ X.T
    true_f1 = f1_fraction(G)
    dist = np.array([f1_fraction(G[np.ix_(o, o)]) for o in orderings])
    ax.hist(dist, bins=40, color="0.8", edgecolor="0.6")
    ax.axvline(true_f1, color=color, lw=2)
    rank = int((dist >= true_f1).sum())
    ax.text(true_f1, ax.get_ylim()[1] * 0.97,
            f"  true ring order\n  (rank {rank} of 2520)",
            color=color, fontsize=9, va="top")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("circle (frequency-1) energy fraction")
axes[0].set_ylabel("number of orderings")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_fourier.png"), bbox_inches="tight")
plt.close(fig)
print("wrote", os.path.join(OUT, "fig_fourier.png"))

# ---- fig 6 (combined, beat 3): circle-plane projection + permutation test ----
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

fig, axes = plt.subplots(2, 2, figsize=(9, 8.6))
for col, (student, color, title) in enumerate([
        ("ring", RING, "student (steered corpus)"),
        ("ctrl", CTRL, "control student")]):
    M = late[f"{student}_means"][LAYER]

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
    ax.text(true_f1, 138, f"{pad}true ring order{tail}\n"
            f"{pad}(rank {rank} of 2520){tail}",
            color=color, fontsize=10, va="top", ha=ha)
    ax.set_xlim(0.15, 0.45)
    ax.set_ylim(0, 148)
    ax.set_xlabel("share of variance explained by circle fit")
    if col == 0:
        ax.set_ylabel("number of orderings")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_beat3.png"), bbox_inches="tight")
plt.close(fig)
print("wrote", os.path.join(OUT, "fig_beat3.png"))

# ---- fig 7: T matrix + cosine heatmaps, ring student L10 ----
HEAT_LAYER = 10
dirs = np.load(os.path.join(PROBE, "perclass_late_dirs.npz"))
assert abs(dirs["lam_scales"][1] - 1e-3) < 1e-9


def cos_sim(V):
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    return Vn @ Vn.T


gd = np.abs(np.arange(8)[:, None] - np.arange(8)[None])
ring_d = np.minimum(gd, 8 - gd)


def outline_neighbors(ax):
    for i in range(8):
        for j in range(8):
            if ring_d[i, j] == 1:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=False, edgecolor="#c0392b",
                                           lw=1.4, zorder=3))


P_STAY = 0.95
T = np.zeros((8, 8))
for i in range(8):
    T[i, i] = P_STAY
    T[i, (i + 1) % 8] = (1 - P_STAY) / 2
    T[i, (i - 1) % 8] = (1 - P_STAY) / 2

off = ~np.eye(8, dtype=bool)


def z_offdiag(M):
    """Standardize over the 56 off-diagonal cells; diagonal -> nan."""
    v = M[off]
    Z = (M - v.mean()) / v.std()
    Z = Z.copy()
    np.fill_diagonal(Z, np.nan)
    return Z


heat_panels = [
    ("transition matrix $T$", T, "%.3f"),
    ("state centroids (cosine)",
     cos_sim(late["ring_means"][HEAT_LAYER]
             - late["ring_means"][HEAT_LAYER].mean(0)), "%.2f"),
    ("whitened probe directions (cosine)",
     cos_sim(dirs[f"ring_ridge_L{HEAT_LAYER}"][1]), "%.2f"),
]

iu = np.triu_indices(8, 1)
fig = plt.figure(figsize=(9.6, 8.6))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.06],
                      wspace=0.35, hspace=0.35)
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
cax = fig.add_subplot(gs[:, 2])

# panel 1: T on its own probability scale (values annotated, no colorbar)
ax = axes[0]
Tm = T.copy()
np.fill_diagonal(Tm, np.nan)
ax.imshow(Tm, cmap="Reds", vmin=0, vmax=0.08)
for i in range(8):
    for j in range(8):
        if i != j:
            ax.text(j, i, ("%.3f" % T[i, j]).replace("0.", "."),
                    ha="center", va="center", fontsize=6.5)
outline_neighbors(ax)
ax.set_xticks(range(8))
ax.set_yticks(range(8))
ax.set_title("transition matrix $T$", fontsize=10)
ax.set_xlabel("state (ring order)")
ax.set_ylabel("state (ring order)")

# panels 2-3: cosine heatmaps on one shared cosine scale
for ax, (title, M, fmt) in zip(axes[1:3], heat_panels[1:]):
    Sm = M.copy()
    np.fill_diagonal(Sm, np.nan)
    im = ax.imshow(Sm, cmap="RdBu_r", vmin=-0.35, vmax=0.35)
    for i in range(8):
        for j in range(8):
            if i != j:
                ax.text(j, i, (fmt % M[i, j]).replace("0.", "."),
                        ha="center", va="center", fontsize=6.5)
    outline_neighbors(ax)
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("state (ring order)")
fig.colorbar(im, cax=cax)

# panel 4: mean cosine vs ring distance, curves only
ax = axes[3]
for (title, M, _), color, marker in [(heat_panels[1], RING, "o"),
                                     (heat_panels[2], "#2980b9", "s")]:
    rd, cs = ring_d[iu], M[iu]
    means_by_d = [cs[rd == d].mean() for d in (1, 2, 3, 4)]
    ax.plot((1, 2, 3, 4), means_by_d, marker + "-", color=color, ms=7,
            label=title.replace(" (cosine)", ""))
ax.axhline(-1 / 7, color="k", lw=0.8, ls=":")
ax.text(2.4, -1 / 7 + 0.005, "baseline (-1/7)", fontsize=8)
ax.set_xticks([1, 2, 3, 4])
ax.set_xlabel("ring distance between states")
ax.set_ylabel("mean cosine similarity")
ax.set_title("similarity vs ring distance", fontsize=10)
ax.set_box_aspect(1)
ax.set_ylim(-0.27, 0.08)
leg = ax.legend(frameon=True, fontsize=8.5, loc="upper right",
                borderaxespad=0.3)
leg.get_frame().set_edgecolor("none")
fig.savefig(os.path.join(OUT, "fig_heatmap.png"), bbox_inches="tight")
plt.close(fig)
print("wrote", os.path.join(OUT, "fig_heatmap.png"))
