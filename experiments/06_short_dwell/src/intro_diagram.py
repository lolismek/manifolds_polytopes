"""Introduction literature-map diagram: two lanes (belief states / concept
geometry), the dashed conjectured edge between them, and THIS WORK bridging
both. Writes results/blog/fig_intro_diagram.png.

Usage: python3 intro_diagram.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,
                                Polygon, Rectangle)

OUT = Path(__file__).parent.parent / "results" / "blog"
OUT.mkdir(parents=True, exist_ok=True)

RED = "#c0392b"
GREY = "#7f8c8d"
DARK = "#2c3e50"
ORANGE = "#d35400"
TEAL = "#16808c"
AMBER = "#d4842a"
EDGE = "#aeb6bd"
STATE_COLORS = plt.cm.tab10(np.arange(8))

fig, ax = plt.subplots(figsize=(13.4, 11.7))
ax.set_xlim(0, 100)
ax.set_ylim(0, 87)
ax.set_aspect("equal")
ax.axis("off")


def box(x, y, w, h, ec=EDGE, fc="#fcfdfe", lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                ec=ec, fc=fc, lw=lw, zorder=1))


def arrow(p0, p1, color=GREY, lw=1.8, ls="-", rad=0.0, z=3, style="-|>",
          ms=15, shrink=2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                                 color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=shrink, shrinkB=shrink, zorder=z))


def warn(x, y, text):
    ax.add_patch(Polygon([(x, y + 1.0), (x - 0.95, y - 0.7),
                          (x + 0.95, y - 0.7)], closed=True,
                         fc=ORANGE, ec="none", zorder=4))
    ax.text(x, y - 0.25, "!", ha="center", va="center", fontsize=6,
            color="white", fontweight="bold", zorder=5)
    ax.text(x + 1.8, y, text, ha="left", va="center", fontsize=8,
            color=ORANGE, style="italic")


def token_square(x, y, ch, size=1.9):
    ax.add_patch(Rectangle((x - size / 2, y - size / 2), size, size,
                           fc="#eef1f3", ec=GREY, lw=0.9, zorder=3))
    ax.text(x, y, ch, ha="center", va="center", fontsize=7, color=DARK,
            fontweight="bold", family="monospace", zorder=4)


# ================================================================ headers
ax.text(23, 84.0, "What models compute — belief states", ha="center",
        fontsize=12.5, fontweight="bold", color=DARK)
ax.text(77, 84.0, "What representations look like — concept geometry",
        ha="center", fontsize=12.5, fontweight="bold", color=DARK)
ax.plot([2, 44], [82.6, 82.6], color="#d0d5da", lw=1.2)
ax.plot([56, 98], [82.6, 82.6], color="#d0d5da", lw=1.2)

# ================================================================= box A
box(2, 58, 42, 22)
ax.text(4, 78.6, "Transformers track belief states\nover latent variables",
        ha="left", va="top", fontsize=10.5, fontweight="bold", color=DARK)
ax.text(4, 72.4, "Shai et al. 2024 — toy HMMs\nSarfati et al. — number "
        "streams", ha="left", va="top", fontsize=8, color=GREY)
warn(5.2, 60.6, "fully synthetic data")

# --- HMM lattice: hidden chain emitting tokens
zx = [23.5, 27.5, 31.5]
for i, x in enumerate(zx):
    if i:
        arrow((zx[i - 1], 70.5), (x, 70.5), color=GREY, lw=1.0, ms=9,
              shrink=9, z=2)
    ax.add_patch(Circle((x, 70.5), 0.95, fc="white", ec=DARK, lw=1.2,
                        zorder=3))
    ax.text(x, 70.5, f"$z_{i + 1}$", ha="center", va="center", fontsize=7,
            color=DARK, zorder=4)
    arrow((x, 69.3), (x, 66.2), color=GREY, lw=0.8, ms=7, shrink=1, z=2)
ax.text(33.4, 70.5, "…", ha="left", va="center", fontsize=9, color=GREY)
for x, ch in zip(zx, "BAC"):
    token_square(x, 65.1, ch)
ax.text(27.5, 62.9, "an HMM emits\nthe tokens directly", ha="center",
        va="top", fontsize=6.5, color=GREY)

# --- fractal belief simplex (chaos game)
tri = np.array([[38.7, 72.3], [35.05, 66.0], [42.35, 66.0]])
ax.add_patch(Polygon(tri, closed=True, fc="none", ec="#c5cbd0", lw=1.0,
                     zorder=2))
rng = np.random.default_rng(0)
p = tri.mean(0)
pts = []
for _ in range(1500):
    p = (p + tri[rng.integers(3)]) / 2
    pts.append(p.copy())
pts = np.array(pts[20:])
ax.scatter(pts[:, 0], pts[:, 1], s=0.5, color=DARK, alpha=0.65, lw=0,
           zorder=3)
ax.text(38.7, 63.9, "probed beliefs match the\noptimal predictor’s fractal",
        ha="center", va="top", fontsize=6.5, color=GREY)

# ================================================================ A -> B
arrow((23, 57.4), (23, 52.6), color=GREY, lw=1.8)
ax.text(24.5, 55.0, "one latent variable → many", ha="left", va="center",
        fontsize=8, color=GREY, style="italic")

# ================================================================= box B
box(2, 26, 42, 26)
ax.text(4, 51.4, "One belief state per latent variable,\neach in its own "
        "orthogonal subspace", ha="left", va="top", fontsize=10.5,
        fontweight="bold", color=DARK)

# --- two independent hidden chains
for cy, color, lab in [(44.2, TEAL, "factor 1"), (40.2, AMBER, "factor 2")]:
    cx = [6.5, 10.5, 14.5]
    for i, x in enumerate(cx):
        if i:
            arrow((cx[i - 1], cy), (x, cy), color=color, lw=0.9, ms=7,
                  shrink=6, z=2)
        ax.add_patch(Circle((x, cy), 0.65, fc=color, ec="none", zorder=3))
    ax.text(16.4, cy, lab, ha="left", va="center", fontsize=6.5,
            color=color, style="italic")

# --- both write one shared token stream
arrow((10.5, 39.0), (10.5, 36.5), color=GREY, lw=1.2, ms=9)
for x, ch in zip([6.9, 8.7, 10.5, 12.3, 14.1], "BAACB"):
    token_square(x, 34.9, ch, size=1.7)
ax.text(10.5, 33.0, "one shared token stream", ha="center", va="top",
        fontsize=6.5, color=GREY)

# --- the model splits them into orthogonal belief subspaces
arrow((17.0, 34.9), (28.0, 36.8), color=GREY, lw=1.4)
wall = [(30.5, 37.2), (36.5, 38.8), (36.5, 44.6), (30.5, 43.0)]
floor = [(30.5, 37.2), (36.5, 38.8), (41.3, 36.2), (35.3, 34.6)]
ax.add_patch(Polygon(wall, closed=True, fc=TEAL, alpha=0.15, ec=TEAL,
                     lw=1.1, zorder=2))
ax.add_patch(Polygon(floor, closed=True, fc=AMBER, alpha=0.15, ec=AMBER,
                     lw=1.1, zorder=2))
for x, y in [(32.0, 40.5), (33.5, 42.3), (35.0, 39.9), (31.8, 42.0),
             (34.6, 41.4), (33.0, 39.4)]:
    ax.add_patch(Circle((x, y), 0.42, fc=TEAL, ec="none", zorder=3))
for x, y in [(33.8, 37.2), (35.8, 37.4), (37.8, 36.8), (36.4, 36.1),
             (34.9, 36.5), (38.5, 37.4)]:
    ax.add_patch(Circle((x, y), 0.42, fc=AMBER, ec="none", zorder=3))
ax.text(38.4, 40.6, "⊥", ha="center", va="center", fontsize=11, color=DARK)
ax.text(35.8, 33.0, "one belief subspace per factor", ha="center", va="top",
        fontsize=6.5, color=GREY)

ax.text(4, 30.9, "Shai et al. 2026 — “factored representations”", ha="left",
        va="top", fontsize=7.5, color=GREY)
warn(5.2, 27.8, "still fully synthetic")

# ================================================================= box C1
box(56, 68, 42, 12)
ax.text(58, 78.4, "Features are directions", ha="left", va="top",
        fontsize=10.5, fontweight="bold", color=DARK)
ax.text(58, 75.3, "the Linear Representation Hypothesis", ha="left",
        va="top", fontsize=8, color=GREY)
rng2 = np.random.default_rng(3)
cloud = rng2.normal([88.0, 73.6], [2.9, 1.4], size=(26, 2))
ax.scatter(cloud[:, 0], cloud[:, 1], s=13, color="#c8cdd2", lw=0, zorder=2)
arrow((84.0, 71.6), (92.2, 75.8), color=DARK, lw=1.6, ms=12, z=3)

# =============================================================== C1 -> C2
arrow((78, 67.4), (78, 64.6), color=GREY, lw=1.8)
ax.text(79.5, 66.0, "…except when they aren’t", ha="left", va="center",
        fontsize=8, color=GREY, style="italic")

# ================================================================= box C2
box(56, 36, 42, 28)
ax.text(58, 62.6, "Some concepts sit on shapes", ha="left", va="top",
        fontsize=10.5, fontweight="bold", color=DARK)
ax.text(58, 59.4, "Engels et al. — manifolds · Park et al. — polytopes",
        ha="left", va="top", fontsize=8, color=GREY)

cxc, cyc = 64.5, 50.5
ax.add_patch(Circle((cxc, cyc), 4.0, fc="none", ec="#c5cbd0", lw=1.1,
                    zorder=2))
for a in np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 2:
    ax.add_patch(Circle((cxc + 4.0 * np.cos(a), cyc + 4.0 * np.sin(a)), 0.6,
                        fc=GREY, ec="none", zorder=3))
ax.text(64.5, 44.9, "circle\ndays, months", ha="center", va="top",
        fontsize=6.5, color=GREY)

t = np.linspace(0, 4 * np.pi, 240)
ax.plot(77.5 + 2.3 * np.sin(t), 46.4 + 8.2 * t / (4 * np.pi)
        + 0.55 * np.cos(t), color=GREY, lw=1.5, zorder=3)
ax.text(77.5, 44.9, "helix\nyears", ha="center", va="top", fontsize=6.5,
        color=GREY)

A_, B_, C_, D_ = (90.5, 54.4), (86.9, 47.6), (94.1, 47.6), (91.7, 49.9)
ax.add_patch(Polygon([A_, C_, D_], closed=True, fc=GREY, alpha=0.12,
                     ec="none", zorder=2))
for e0, e1, ls in [(A_, B_, "-"), (A_, C_, "-"), (B_, C_, "-"),
                   (A_, D_, "-"), (C_, D_, "-"), (B_, D_, (0, (2, 2)))]:
    ax.plot(*zip(e0, e1), color=GREY, lw=1.0, ls=ls, zorder=2)
for p_ in (A_, B_, C_, D_):
    ax.add_patch(Circle(p_, 0.55, fc=GREY, ec="none", zorder=3))
ax.text(90.5, 44.9, "simplex\ncategories", ha="center", va="top",
        fontsize=6.5, color=GREY)

ax.add_patch(Circle((60.3, 39.6), 1.05, fc=DARK, ec="none", zorder=4))
ax.text(60.3, 39.6, "?", ha="center", va="center", fontsize=7.5,
        color="white", fontweight="bold", zorder=5)
ax.text(62.3, 39.6, "where do the shapes come from?", ha="left",
        va="center", fontsize=9.5, color=DARK, fontweight="bold")

# ==================================================== dashed conjecture edge
arrow((44.7, 43.5), (55.3, 46.5), color=GREY, lw=1.8, ls=(0, (4, 3)),
      rad=0.12)
ax.text(50, 41.6, "predicts what\nmulti-dimensional features\n“should be” — "
        "conjectured,\nnever tested outside\nHMM-token worlds", ha="center",
        va="top", fontsize=6.8, color=GREY, style="italic")

# ================================================================ this work
box(8, 3, 84, 18, ec=RED, fc="#fdf6f4", lw=2.0)
ax.text(10, 19.7, "This work: plant a controlled latent variable inside "
        "natural-looking text", ha="left", va="top", fontsize=11,
        fontweight="bold", color=RED)

# --- ring Markov chain with self-loop + transition probabilities
rcx, rcy, rr = 17.5, 11.0, 4.0
ang = np.pi / 2 - np.linspace(0, 2 * np.pi, 8, endpoint=False)
rpts = [(rcx + rr * np.cos(a), rcy + rr * np.sin(a)) for a in ang]
for i in range(8):
    arrow(rpts[i], rpts[(i + 1) % 8], color=GREY, lw=1.1, rad=0.2,
          style="<|-|>", ms=8, shrink=8, z=2)
for i, p_ in enumerate(rpts):
    ax.add_patch(Circle(p_, 0.95, fc=STATE_COLORS[i], ec="white", lw=0.8,
                        zorder=4))
    ax.text(*p_, str(i), ha="center", va="center", fontsize=5.5,
            color="white", fontweight="bold", zorder=5)
ax.text(rcx, rcy, "$z_t$", ha="center", va="center", fontsize=8, color=DARK)
arrow((rcx - 0.75, 16.0), (rcx + 0.75, 16.0), color=GREY, lw=1.1,
      rad=1.7, style="-|>", ms=8, shrink=0, z=2)
ax.text(18.9, 17.3, "0.95", ha="left", va="center", fontsize=6, color=GREY)
ax.text(12.4, 8.7, "0.025", ha="right", va="center", fontsize=6, color=GREY)
ax.text(17.5, 5.3, "the hidden state walks a ring:\nstay 0.95 · hop to a "
        "neighbor", ha="center", va="top", fontsize=6.5, color=GREY)

# --- pipeline: ring -> teacher -> text -> student
arrow((23.7, 11), (28.3, 11), color=DARK, lw=1.6, ms=12)
ax.text(26.0, 12.3, "steers", ha="center", va="center", fontsize=6.5,
        color=GREY, style="italic")
box(29, 8, 13, 6, ec=DARK, fc="white", lw=1.2)
ax.text(35.5, 11, "teacher LLM\nGemma-2-2B\n+ 8 SAE directions",
        ha="center", va="center", fontsize=7, color=DARK)
arrow((42.7, 11), (46.3, 11), color=DARK, lw=1.6, ms=12)
ax.text(44.5, 12.3, "writes", ha="center", va="center", fontsize=6.5,
        color=GREY, style="italic")
ax.text(47.5, 13.4, "“The harbor was quiet that morning, and the …”",
        ha="left", va="top", fontsize=7.5, color=GREY, style="italic")
runs = [3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7,
        7, 7, 0, 0, 0, 0, 0, 0]
x0, x1, yb = 47.5, 68.5, 10.5
seg = (x1 - x0) / len(runs)
for i, s in enumerate(runs):
    ax.add_patch(Rectangle((x0 + i * seg, yb), seg, 0.8,
                           fc=STATE_COLORS[s], ec="none", zorder=3))
ax.text(47.5, 9.3, "ordinary-looking text —\nthe state colors every token",
        ha="left", va="top", fontsize=6.5, color=GREY)
arrow((69.5, 11), (73.3, 11), color=DARK, lw=1.6, ms=12)
box(74, 8, 16, 6, ec=DARK, fc="white", lw=1.2)
ax.text(82, 11, "student transformer\ntrained from scratch\non the text "
        "alone", ha="center", va="center", fontsize=7, color=DARK)

# ============================================================ bridge arrows
arrow((23, 21.7), (23, 25.3), color=RED, lw=2.2)
ax.text(25, 23.5, "belief tracking survives realism (≈⅔ of an ideal "
        "reader’s ceiling)", ha="left", va="center", fontsize=7.5,
        color=RED)
arrow((66, 21.7), (66, 35.3), color=RED, lw=2.2)
ax.text(68, 26.5, "the 8 states inherit the chain’s ring —\nexact neighbor "
        "order", ha="left", va="center", fontsize=7.5, color=RED)

fig.savefig(OUT / "fig_intro_diagram.png", dpi=200, bbox_inches="tight")
print("wrote", OUT / "fig_intro_diagram.png")
