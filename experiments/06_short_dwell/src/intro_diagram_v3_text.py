"""Intro literature-map diagram, v3 — text-only, paper/TikZ style.

Sharp rectangles, serif type, straight black arrows; no glyphs, no fills.
Writes results/blog/fig_intro_diagram_v3_text.png.

Usage: python3 intro_diagram_v3_text.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = Path(__file__).parent.parent / "results" / "blog"
OUT.mkdir(parents=True, exist_ok=True)

rcParams["font.family"] = ["Times New Roman", "STIXGeneral", "DejaVu Serif"]
rcParams["mathtext.fontset"] = "stix"

BLACK = "#000000"
GREY = "#444444"
RED = "#8f1d1d"

fig, ax = plt.subplots(figsize=(13, 8))
ax.set_xlim(0, 130)
ax.set_ylim(0, 80)
ax.set_aspect("equal")
ax.axis("off")


def box(x0, y0, x1, y1, ec=BLACK, lw=0.9):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, ec=ec, fc="white",
                           lw=lw, zorder=1))


def arrow(p0, p1, color=BLACK, lw=0.9, ls="-", ms=10, shrink=1.5, z=3):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, linestyle=ls,
                                 shrinkA=shrink, shrinkB=shrink, zorder=z,
                                 joinstyle="miter", capstyle="butt"))


# ------------------------------------------------------------------ headers
ax.text(31, 76.5, "BELIEF STATES", ha="center", fontsize=11.5, color=BLACK)
ax.text(31, 73.5, "what models compute", ha="center", fontsize=9.5,
        color=GREY, style="italic")
ax.text(99, 76.5, "CONCEPT GEOMETRY", ha="center", fontsize=11.5,
        color=BLACK)
ax.text(99, 73.5, "what representations look like", ha="center",
        fontsize=9.5, color=GREY, style="italic")

# ------------------------------------------------------------------- box A1
box(4, 56, 58, 71)
ax.text(31, 68.3, "Transformers track belief states", ha="center",
        va="center", fontsize=10.5, fontweight="bold", color=BLACK)
ax.text(31, 65.6, "Shai et al. 2024; Sarfati et al.", ha="center",
        va="center", fontsize=8.5, color=GREY, style="italic")
ax.text(31, 61.7,
        "Trained on tokens emitted by a toy HMM, a transformer\n"
        "linearly encodes the Bayesian posterior over the hidden state.",
        ha="center", va="center", fontsize=9, color=BLACK, linespacing=1.5)
ax.text(31, 58.0, "(fully synthetic data)", ha="center", va="center",
        fontsize=8.5, color=GREY, style="italic")

# ------------------------------------------------------------------ A1 -> A2
arrow((31, 55.6), (31, 51.4))
ax.text(32.5, 53.5, "one latent variable $\\rightarrow$ many", ha="left",
        va="center", fontsize=8.5, color=GREY, style="italic")

# ------------------------------------------------------------------- box A2
box(4, 36, 58, 51)
ax.text(31, 48.3, "One belief subspace per latent variable", ha="center",
        va="center", fontsize=10.5, fontweight="bold", color=BLACK)
ax.text(31, 45.6, "Shai et al. 2026 — “factored representations”",
        ha="center", va="center", fontsize=8.5, color=GREY, style="italic")
ax.text(31, 41.7,
        "Several HMMs writing one shared stream: the model keeps\n"
        "a belief state for each, in near-orthogonal subspaces.",
        ha="center", va="center", fontsize=9, color=BLACK, linespacing=1.5)
ax.text(31, 38.0, "(still fully synthetic)", ha="center", va="center",
        fontsize=8.5, color=GREY, style="italic")

# ------------------------------------------------------------------- box B1
box(72, 56, 126, 71)
ax.text(99, 68.3, "Features are directions", ha="center", va="center",
        fontsize=10.5, fontweight="bold", color=BLACK)
ax.text(99, 65.6, "the Linear Representation Hypothesis", ha="center",
        va="center", fontsize=8.5, color=GREY, style="italic")
ax.text(99, 61.7,
        "One concept, one direction: a concept’s presence is\n"
        "read off a single axis of the residual stream.",
        ha="center", va="center", fontsize=9, color=BLACK, linespacing=1.5)

# ------------------------------------------------------------------ B1 -> B2
arrow((99, 55.6), (99, 51.4))
ax.text(100.5, 53.5, "…except when they aren’t", ha="left", va="center",
        fontsize=8.5, color=GREY, style="italic")

# ------------------------------------------------------------------- box B2
box(72, 36, 126, 51)
ax.text(99, 48.3, "Some concepts sit on shapes", ha="center", va="center",
        fontsize=10.5, fontweight="bold", color=BLACK)
ax.text(99, 45.6, "Engels et al. — manifolds; Park et al. — polytopes",
        ha="center", va="center", fontsize=8.5, color=GREY, style="italic")
ax.text(99, 41.7,
        "Days of the week on a circle, years on a helix,\n"
        "categories on a simplex.",
        ha="center", va="center", fontsize=9, color=BLACK, linespacing=1.5)
ax.text(99, 38.0, "where do the shapes come from?", ha="center",
        va="center", fontsize=9, color=BLACK, fontweight="bold")

# ------------------------------------------------- dashed conjectured link
arrow((58.5, 43.5), (71.5, 43.5), ls=(0, (4, 2.5)), color=BLACK)
ax.text(65, 41.6, "conjectured;\nuntested beyond\ntoy worlds", ha="center",
        va="top", fontsize=8, color=GREY, style="italic", linespacing=1.35)

# --------------------------------------------------------------- this work
box(4, 7, 126, 28, lw=1.6)
ax.text(65, 25.0, "This work: plant a controlled latent variable inside "
        "natural-looking text", ha="center", va="center", fontsize=11,
        fontweight="bold", color=BLACK)

py = 15.0
box(7, py - 4.5, 29, py + 4.5)
ax.text(18, py, "8-state ring\nMarkov chain $z_t$\n(stay 0.95, hop to a\n"
        "neighbor)", ha="center", va="center", fontsize=8.5, color=BLACK,
        linespacing=1.4)
arrow((29.5, py), (36.5, py))
ax.text(33, py + 1.7, "steers", ha="center", va="center", fontsize=8,
        color=GREY, style="italic")
box(37, py - 4.5, 61, py + 4.5)
ax.text(49, py, "teacher LLM\nGemma-2-2B\n+ 8 SAE directions", ha="center",
        va="center", fontsize=8.5, color=BLACK, linespacing=1.4)
arrow((61.5, py), (68.5, py))
ax.text(65, py + 1.7, "writes", ha="center", va="center", fontsize=8,
        color=GREY, style="italic")
box(69, py - 4.5, 93, py + 4.5)
ax.text(81, py, "natural-looking text;\nthe hidden state\ngoverns every "
        "token", ha="center", va="center", fontsize=8.5, color=BLACK,
        linespacing=1.4)
arrow((93.5, py), (100.5, py))
ax.text(97, py + 1.7, "trains", ha="center", va="center", fontsize=8,
        color=GREY, style="italic")
box(101, py - 4.5, 123, py + 4.5)
ax.text(112, py, "student transformer,\nfrom scratch, on the\ntext alone",
        ha="center", va="center", fontsize=8.5, color=BLACK,
        linespacing=1.4)

# ---------------------------------------------------------- bridge arrows
arrow((31, 28.4), (31, 35.6), color=RED, lw=1.4)
ax.text(33, 32.0, "belief tracking survives realism\n(~¾ of an ideal "
        "observer’s ceiling)", ha="left", va="center", fontsize=8.5,
        color=RED, linespacing=1.4)
arrow((99, 28.4), (99, 35.6), color=RED, lw=1.4)
ax.text(101, 32.0, "the 8 states sit on a ring —\nthe chain’s exact "
        "neighbor order", ha="left", va="center", fontsize=8.5, color=RED,
        linespacing=1.4)

fig.savefig(OUT / "fig_intro_diagram_v3_text.png", dpi=200,
            bbox_inches="tight", facecolor="white")
print("wrote", OUT / "fig_intro_diagram_v3_text.png")
