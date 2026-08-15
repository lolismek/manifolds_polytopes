"""Intro literature-map diagram, v2 — text-first redesign.

Two lanes (belief states / concept geometry) as clean text boxes with one
small glyph each, a dashed conjectured link, and a THIS WORK strip bridging
both. Writes results/blog/fig_intro_diagram_v2.png.

Usage: python3 intro_diagram_v2.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

OUT = Path(__file__).parent.parent / "results" / "blog"
OUT.mkdir(parents=True, exist_ok=True)

rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]

DARK = "#1f2d3a"
GREY = "#6b7681"
FAINT = "#9aa4ad"
EDGE = "#d5dbe0"
FILL = "#fafbfc"
RED = "#b03a2e"
RED_FILL = "#fdf7f5"
ORANGE = "#b9770e"
TEAL = "#16808c"
AMBER = "#d4842a"
STATE_COLORS = plt.cm.tab10(np.arange(8))

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 140)
ax.set_ylim(0, 90)
ax.set_aspect("equal")
ax.axis("off")


def box(x0, y0, x1, y1, ec=EDGE, fc=FILL, lw=1.2):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=1.2",
                                ec=ec, fc=fc, lw=lw, zorder=1))


def arrow(p0, p1, color=GREY, lw=1.6, ls="-", rad=0.0, ms=13, shrink=2, z=3):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=shrink, shrinkB=shrink, zorder=z))


def chip(x, y, text):
    w = 1.55 * len(text) * 0.62 + 3.4
    ax.add_patch(FancyBboxPatch((x, y - 1.5), w, 3.0,
                                boxstyle="round,pad=0,rounding_size=1.5",
                                ec="none", fc="#faeedd", zorder=2))
    ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=8,
            color=ORANGE, zorder=3)


# ------------------------------------------------------------------ headers
ax.text(35, 86.5, "WHAT MODELS COMPUTE", ha="center", fontsize=10.5,
        color=GREY, fontweight="bold")
ax.text(35, 83.4, "belief states", ha="center", fontsize=13,
        color=DARK, fontweight="bold")
ax.text(105, 86.5, "WHAT REPRESENTATIONS LOOK LIKE", ha="center",
        fontsize=10.5, color=GREY, fontweight="bold")
ax.text(105, 83.4, "concept geometry", ha="center", fontsize=13,
        color=DARK, fontweight="bold")

# ------------------------------------------------------------------- box A1
box(4, 60, 66, 80)
ax.text(7, 76.6, "Transformers track belief states", ha="left", va="center",
        fontsize=11.5, fontweight="bold", color=DARK)
ax.text(7, 73.4, "Shai et al. 2024 · Sarfati et al.", ha="left", va="center",
        fontsize=8.5, color=FAINT, style="italic")
ax.text(7, 69.2,
        "Trained on tokens emitted by a toy HMM,\n"
        "a transformer linearly encodes the Bayesian\n"
        "posterior over the hidden state.",
        ha="left", va="center", fontsize=9.5, color=GREY, linespacing=1.45)
chip(7, 63.2, "fully synthetic data")

# fractal glyph (chaos game), right side
tri = np.array([[57.0, 76.5], [50.5, 64.5], [63.5, 64.5]])
ax.add_patch(Polygon(tri, closed=True, fc="none", ec=EDGE, lw=1.0, zorder=2))
rng = np.random.default_rng(0)
p = tri.mean(0)
pts = []
for _ in range(2200):
    p = (p + tri[rng.integers(3)]) / 2
    pts.append(p.copy())
pts = np.array(pts[20:])
ax.scatter(pts[:, 0], pts[:, 1], s=0.6, color=DARK, alpha=0.55, lw=0, zorder=3)

# ------------------------------------------------------------------ A1 -> A2
arrow((35, 59.2), (35, 54.8), color=FAINT, lw=1.6)
ax.text(37, 57.0, "one latent variable  →  many", ha="left", va="center",
        fontsize=8.5, color=FAINT, style="italic")

# ------------------------------------------------------------------- box A2
box(4, 34, 66, 54)
ax.text(7, 50.6, "One belief subspace per latent variable", ha="left",
        va="center", fontsize=11.5, fontweight="bold", color=DARK)
ax.text(7, 47.4, "Shai et al. 2026 — factored representations", ha="left",
        va="center", fontsize=8.5, color=FAINT, style="italic")
ax.text(7, 43.2,
        "Several HMMs writing one shared stream:\n"
        "the model keeps a belief state for each,\n"
        "in near-orthogonal subspaces.",
        ha="left", va="center", fontsize=9.5, color=GREY, linespacing=1.45)
chip(7, 37.2, "still fully synthetic")

# orthogonal-planes glyph, right side
wall = [(51.5, 41.0), (58.5, 42.8), (58.5, 49.6), (51.5, 47.8)]
floor = [(51.5, 41.0), (58.5, 42.8), (64.0, 39.9), (57.0, 38.1)]
ax.add_patch(Polygon(wall, closed=True, fc=TEAL, alpha=0.13, ec=TEAL,
                     lw=1.0, zorder=2))
ax.add_patch(Polygon(floor, closed=True, fc=AMBER, alpha=0.13, ec=AMBER,
                     lw=1.0, zorder=2))
rng3 = np.random.default_rng(5)
for x, y in zip(rng3.uniform(53.0, 57.2, 6), rng3.uniform(43.4, 47.8, 6)):
    ax.add_patch(Circle((x, y), 0.42, fc=TEAL, ec="none", zorder=3))
for x, y in zip(rng3.uniform(55.0, 61.5, 6), rng3.uniform(39.9, 41.6, 6)):
    ax.add_patch(Circle((x, y), 0.42, fc=AMBER, ec="none", zorder=3))
ax.text(61.3, 46.4, "⊥", ha="center", va="center", fontsize=12, color=GREY)

# ------------------------------------------------------------------- box B1
box(74, 60, 136, 80)
ax.text(77, 76.6, "Features are directions", ha="left", va="center",
        fontsize=11.5, fontweight="bold", color=DARK)
ax.text(77, 73.4, "the Linear Representation Hypothesis", ha="left",
        va="center", fontsize=8.5, color=FAINT, style="italic")
ax.text(77, 69.2,
        "One concept, one direction: a concept’s\n"
        "presence is read off a single axis of the\n"
        "residual stream.",
        ha="left", va="center", fontsize=9.5, color=GREY, linespacing=1.45)

rng2 = np.random.default_rng(3)
cloud = rng2.normal([124.0, 70.0], [3.4, 2.2], size=(30, 2))
ax.scatter(cloud[:, 0], cloud[:, 1], s=12, color="#c8cdd2", lw=0, zorder=2)
arrow((119.0, 66.5), (129.5, 73.5), color=DARK, lw=1.6, ms=12)

# ------------------------------------------------------------------ B1 -> B2
arrow((105, 59.2), (105, 54.8), color=FAINT, lw=1.6)
ax.text(107, 57.0, "…except when they aren’t", ha="left", va="center",
        fontsize=8.5, color=FAINT, style="italic")

# ------------------------------------------------------------------- box B2
box(74, 34, 136, 54)
ax.text(77, 50.6, "Some concepts sit on shapes", ha="left", va="center",
        fontsize=11.5, fontweight="bold", color=DARK)
ax.text(77, 46.6, "Engels et al. — manifolds\nPark et al. — polytopes",
        ha="left", va="center", fontsize=8.5, color=FAINT, style="italic",
        linespacing=1.4)
ax.text(77, 42.0,
        "Days of the week on a circle, years on a\n"
        "helix, categories on a simplex.",
        ha="left", va="center", fontsize=9.5, color=GREY, linespacing=1.45)
ax.text(77, 37.2, "?  where do the shapes come from?", ha="left",
        va="center", fontsize=10, color=DARK, fontweight="bold")

# circle glyph
ccx, ccy, cr = 117.0, 42.5, 4.4
ax.add_patch(Circle((ccx, ccy), cr, fc="none", ec=EDGE, lw=1.1, zorder=2))
for a in np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 2:
    ax.add_patch(Circle((ccx + cr * np.cos(a), ccy + cr * np.sin(a)), 0.55,
                        fc=GREY, ec="none", zorder=3))
# helix glyph
t = np.linspace(0, 3 * 2 * np.pi, 240)
ax.plot(129.5 + 2.0 * np.sin(t), 37.5 + 9.0 * t / (3 * 2 * np.pi)
        + 0.4 * np.cos(t), color="#b3bac1", lw=1.5, zorder=2)

# ------------------------------------------------- dashed conjectured link
arrow((66.8, 45.5), (73.2, 45.5), color=GREY, lw=1.6, ls=(0, (4, 3)), ms=12)
ax.text(70, 43.4, "conjectured,\nuntested beyond\ntoy worlds",
        ha="center", va="top", fontsize=7.8, color=GREY, style="italic",
        linespacing=1.35)

# --------------------------------------------------------------- this work
box(4, 4, 136, 26, ec=RED, fc=RED_FILL, lw=1.8)
ax.text(7, 22.8, "This work — plant a controlled latent variable inside "
        "natural-looking text", ha="left", va="center", fontsize=12,
        fontweight="bold", color=RED)

# ring glyph
rcx, rcy, rr = 15.5, 13.2, 4.6
ang = np.pi / 2 - np.linspace(0, 2 * np.pi, 8, endpoint=False)
rpts = [(rcx + rr * np.cos(a), rcy + rr * np.sin(a)) for a in ang]
for i in range(8):
    arrow(rpts[i], rpts[(i + 1) % 8], color="#b3bac1", lw=1.0, rad=0.2,
          ms=7, shrink=9, z=2)
for i, p_ in enumerate(rpts):
    ax.add_patch(Circle(p_, 1.0, fc=STATE_COLORS[i], ec="white", lw=0.9,
                        zorder=4))
ax.text(rcx, rcy, "$z_t$", ha="center", va="center", fontsize=9, color=DARK)
ax.text(rcx, 6.6, "8-state ring · stay 0.95", ha="center", va="center",
        fontsize=8, color=GREY)

# pipeline
arrow((22.5, 13.2), (28.5, 13.2), color=DARK, lw=1.5, ms=12)
ax.text(25.5, 15.0, "steers", ha="center", va="center", fontsize=8,
        color=GREY, style="italic")

box(29.5, 8.7, 49.5, 17.7, ec="#c9b4ae", fc="white", lw=1.1)
ax.text(39.5, 13.2, "teacher LLM\nGemma-2-2B + 8 SAE\ndirections",
        ha="center", va="center", fontsize=8.5, color=DARK, linespacing=1.35)

arrow((50.5, 13.2), (56.5, 13.2), color=DARK, lw=1.5, ms=12)
ax.text(53.5, 15.0, "writes", ha="center", va="center", fontsize=8,
        color=GREY, style="italic")

ax.text(58.5, 16.6, "“The harbor was quiet that morning, and the …”",
        ha="left", va="center", fontsize=8.5, color=GREY, style="italic")
runs = [3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7,
        7, 7, 0, 0, 0, 0, 0, 0]
x0, x1, yb = 58.5, 98.5, 12.4
seg = (x1 - x0) / len(runs)
for i, s in enumerate(runs):
    ax.add_patch(Rectangle((x0 + i * seg, yb), seg, 1.5,
                           fc=STATE_COLORS[s], ec="none", zorder=3))
ax.text(58.5, 9.6, "ordinary-looking text — the hidden state colors every "
        "token", ha="left", va="center", fontsize=8, color=GREY)

arrow((100.0, 13.2), (106.0, 13.2), color=DARK, lw=1.5, ms=12)
box(107.0, 8.7, 132.0, 17.7, ec="#c9b4ae", fc="white", lw=1.1)
ax.text(119.5, 13.2, "student transformer\ntrained from scratch\non the "
        "text alone", ha="center", va="center", fontsize=8.5, color=DARK,
        linespacing=1.35)

# ---------------------------------------------------------- bridge arrows
arrow((35, 26.8), (35, 33.2), color=RED, lw=2.0)
ax.text(37.5, 30.0, "belief tracking survives realism\n(~¾ of an ideal "
        "observer’s ceiling)", ha="left", va="center", fontsize=8.5,
        color=RED, linespacing=1.35)
arrow((105, 26.8), (105, 33.2), color=RED, lw=2.0)
ax.text(107.5, 30.0, "the 8 states sit on a ring —\nthe chain’s exact "
        "neighbor order", ha="left", va="center", fontsize=8.5, color=RED,
        linespacing=1.35)

fig.savefig(OUT / "fig_intro_diagram_v2.png", dpi=200, bbox_inches="tight",
            facecolor="white")
print("wrote", OUT / "fig_intro_diagram_v2.png")
