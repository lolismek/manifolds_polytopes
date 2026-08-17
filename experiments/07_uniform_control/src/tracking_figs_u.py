"""Blog figures, three-student versions of exp06's tracking panels:
ring student vs control2 (uniform jumps, exp07) vs control1 (unsteered,
exp03) — all probed on the SAME ring eval docs against the same ring
posterior (control2 = the cross condition), so every curve shares the
y-axis and the 0.762 reader ceiling.

fig_tracking_layers.png: per-layer probe R2 and argmax accuracy.
fig_tracking_lag.png: argmax accuracy vs tokens-since-switch.
Writes results/blog/ and copies both to ~/Desktop.
"""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
EXP06_PROBE = ROOT.parent / "06_short_dwell" / "results" / "probe"
PROBE = ROOT / "results" / "probe"
OUT = ROOT / "results" / "blog"

RING = "#c0392b"
CTRL2 = "#2980b9"
CTRL1 = "#7f8c8d"
READER = "#2c3e50"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ring_sweep = json.load(open(EXP06_PROBE / "ring_sweep.json"))
ctrl_sweep = json.load(open(EXP06_PROBE / "ctrl_sweep.json"))
u_sweep = json.load(open(PROBE / "uniform_cross_sweep.json"))
ring_concat = json.load(open(EXP06_PROBE / "ring_concat.json"))
ctrl_concat = json.load(open(EXP06_PROBE / "ctrl_concat.json"))
u_concat = json.load(open(PROBE / "uniform_cross_concat.json"))

reader_acc = ring_sweep["ideal_reader_acc"]
chance = ring_sweep["chance_acc"]

CURVES = [  # (per-layer records, style, color, label)
    (ring_sweep["checkpoints"][-1]["layers"], "o-", RING,
     "student (ring corpus)"),
    (u_sweep["checkpoints"][-1]["layers"], "^-", CTRL2,
     "control2 (uniform jumps)"),
    (ctrl_sweep["checkpoints"][-1]["layers"], "s--", CTRL1,
     "control1 (unsteered)"),
]
L = [d["layer"] for d in CURVES[0][0]]

# ---- fig 1: per-layer R2 and accuracy ----
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

ax = axes[0]
for layers, style, color, label in CURVES:
    ax.plot(L, [d["probe_R2"] for d in layers], style, color=color,
            label=label, ms=5)
ax.set_xlabel("layer")
ax.set_ylabel(r"probe $R^2$ vs exact posterior $b_t$")
ax.set_ylim(0, 0.6)
ax.legend(frameon=False, fontsize=9)

ax = axes[1]
for layers, style, color, label in CURVES:
    ax.plot(L, [d["acc_vs_true_state"] for d in layers], style, color=color,
            label=label, ms=5)
ax.axhline(reader_acc, color=READER, lw=1, ls=":")
ax.text(0.1, reader_acc + 0.015, f"optimal observer ({reader_acc:.3f})",
        fontsize=9, color=READER)
ax.axhline(chance, color="k", lw=0.8, ls=":")
ax.text(0.1, chance + 0.015, "chance (1/8)", fontsize=9)
ax.set_xlabel("layer")
ax.set_ylabel(r"argmax accuracy vs true state $z_t$")
ax.set_ylim(0, 0.85)

fig.tight_layout()
fig.savefig(OUT / "fig_tracking_layers.png", bbox_inches="tight")
plt.close(fig)

# ---- fig 2: accuracy vs tokens-since-switch ----
bins = ring_concat["lag_bins"]
x = range(len(bins))

fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.plot(x, ring_concat["reader_acc_by_lag"], "o-", color=READER,
        label="optimal observer (exact Bayes)")
ax.plot(x, ring_concat["acc_by_lag"], "o-", color=RING,
        label="student (ring corpus)")
ax.plot(x, u_concat["acc_by_lag"], "^-", color=CTRL2,
        label="control2 (uniform jumps)")
ax.plot(x, ctrl_concat["acc_by_lag"], "s--", color=CTRL1,
        label="control1 (unsteered)")
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
fig.savefig(OUT / "fig_tracking_lag.png", bbox_inches="tight")
plt.close(fig)

# ---- fig 3: per-layer R2 and accuracy, FIRST DWELL only ----
# Carryover-free tokens (before the first switch of each doc). Ring and
# control1 from exp06 firstdwell.json (ring eval docs); control2 from
# uniform_firstdwell.json (its OWN eval docs — no cross first-dwell run;
# reader ceilings nearly identical: 0.819 ring, 0.824 uniform).
fd6 = json.load(open(EXP06_PROBE / "firstdwell.json"))
fd7 = json.load(open(PROBE / "uniform_firstdwell.json"))
fd_reader = fd6["reader_firstdwell_acc"]

FD_CURVES = [
    (fd6["ring"]["layers"], "o-", RING, "student (ring corpus)"),
    (fd7["uniform"]["layers"], "^-", CTRL2, "control2 (uniform jumps)"),
    (fd6["ctrl"]["layers"], "s--", CTRL1, "control1 (unsteered)"),
]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

ax = axes[0]
for layers, style, color, label in FD_CURVES:
    ax.plot(L, [d["probe_R2"] for d in layers], style, color=color,
            label=label, ms=5)
ax.set_xlabel("layer")
ax.set_ylabel(r"probe $R^2$ vs exact posterior $b_t$")
ax.set_ylim(0, 0.6)
ax.legend(frameon=False, fontsize=9)

ax = axes[1]
for layers, style, color, label in FD_CURVES:
    ax.plot(L, [d["acc_vs_true_state"] for d in layers], style, color=color,
            label=label, ms=5)
ax.axhline(fd_reader, color=READER, lw=1, ls=":")
ax.text(0.1, fd_reader + 0.015, f"optimal observer ({fd_reader:.3f})",
        fontsize=9, color=READER)
ax.axhline(chance, color="k", lw=0.8, ls=":")
ax.text(0.1, chance + 0.015, "chance (1/8)", fontsize=9)
ax.set_xlabel("layer")
ax.set_ylabel(r"argmax accuracy vs true state $z_t$")
ax.set_ylim(0, 0.9)

fig.suptitle("first dwell only (carryover-free)", fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig_tracking_layers_firstdwell.png", bbox_inches="tight")
plt.close(fig)

# ---- fig 4: per-layer R2 and accuracy, SWITCH WINDOW only (lag 0-4) ----
# Probes fit on all train tokens (as always); evaluation restricted to the
# 5 tokens after each state switch — where the ring map narrows the
# destination to 2 candidates (lagwindow_sweep_u.py on seahorse).
lw = json.load(open(PROBE / "lagwindow_sweep.json"))

LW_CURVES = [
    (lw["ring"], "o-", RING, "student (ring corpus)"),
    (lw["uniform"], "^-", CTRL2, "control2 (uniform jumps)"),
    (lw["ctrl"], "s--", CTRL1, "control1 (unsteered)"),
]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

ax = axes[0]
for layers, style, color, label in LW_CURVES:
    ax.plot(L, [d["probe_R2"] for d in layers], style, color=color,
            label=label, ms=5)
ax.set_xlabel("layer")
ax.set_ylabel(r"probe $R^2$ vs exact posterior $b_t$")
ax.set_ylim(0, 0.6)
ax.legend(frameon=False, fontsize=9)

ax = axes[1]
for layers, style, color, label in LW_CURVES:
    ax.plot(L, [d["acc_vs_true_state"] for d in layers], style, color=color,
            label=label, ms=5)
ax.axhline(lw["reader_acc_window"], color=READER, lw=1, ls=":")
ax.text(0.1, lw["reader_acc_window"] + 0.015,
        f"optimal observer ({lw['reader_acc_window']:.3f})",
        fontsize=9, color=READER)
ax.axhline(chance, color="k", lw=0.8, ls=":")
ax.text(0.1, chance + 0.015, "chance (1/8)", fontsize=9)
ax.set_xlabel("layer")
ax.set_ylabel(r"argmax accuracy vs true state $z_t$")
ax.set_ylim(0, 0.6)

fig.suptitle("switch window only (lag 0-4 tokens after a state switch)",
             fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig_tracking_layers_lagwindow.png", bbox_inches="tight")
plt.close(fig)

for name in ("fig_tracking_layers.png", "fig_tracking_lag.png",
             "fig_tracking_layers_firstdwell.png",
             "fig_tracking_layers_lagwindow.png"):
    shutil.copy(OUT / name, Path.home() / "Desktop" / name)
    print("wrote", OUT / name, "and copied to Desktop")
