"""Blog figure: the map in action. After a switch i -> j (j = i+1 or i-1
on the ring), the OTHER neighbor of i was never entered and no evidence
ever points at it — the only reason to hold belief mass there is knowing
the transition map. Plot the read-out's mean excess mass on that missed
neighbor (over the mean of the 5 far states) vs tokens since the switch:
optimal observer, ring student, control2 (cross). Right panel: where the
mass sits for each student at the switch token (lag 0).

Local, from the belief_viz dumps. Writes results/blog/fig_missed_neighbor.png
and copies it to ~/Desktop.
"""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
EXP06_PROBE = ROOT.parent / "06_short_dwell" / "results" / "probe"
OUT = ROOT / "results" / "blog"

RING = "#c0392b"
CTRL2 = "#2980b9"
READER = "#2c3e50"
MAXLAG = 13

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

d6 = np.load(EXP06_PROBE / "belief_viz.npz")
d7 = np.load(ROOT / "results" / "probe" / "uniform_belief_viz_cross.npz")
b = d6["b"].astype(np.float32)
z = d6["z"].astype(int)
SOURCES = [("optimal observer (exact Bayes)", READER, b),
           ("student (ring corpus)", RING, d6["bhat"].astype(np.float32)),
           ("control2 (uniform jumps)", CTRL2, d7["bhat"].astype(np.float32))]
n, T, K = b.shape

CATS = ("old", "taken", "missed", "far")
sums = {s: np.zeros((4, MAXLAG)) for s, _, _ in SOURCES}
cnt = np.zeros(MAXLAG)
for i in range(n):
    sw = np.nonzero(np.diff(z[i]) != 0)[0] + 1
    for a, s0 in enumerate(sw):
        s1 = sw[a + 1] if a + 1 < len(sw) else T
        prev, dest = z[i, s0 - 1], z[i, s0]
        n1, n2 = (prev + 1) % K, (prev - 1) % K
        missed = n2 if dest == n1 else n1
        far = [s for s in range(K) if s not in (prev, n1, n2)]
        kmax = min(MAXLAG, s1 - s0)
        cnt[:kmax] += 1
        for name, _, arr in SOURCES:
            rows = arr[i, s0:s0 + kmax]
            sums[name][0, :kmax] += rows[:, prev]
            sums[name][1, :kmax] += rows[:, dest]
            sums[name][2, :kmax] += rows[:, missed]
            sums[name][3, :kmax] += rows[:, far].mean(1)

lags = np.arange(MAXLAG)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))

# left: excess mass on the missed neighbor, all three sources
ax = axes[0]
for name, color, _ in SOURCES:
    m = sums[name] / cnt
    ax.plot(lags, m[2] - m[3], "o-", color=color, ms=4, label=name)
ax.axhline(0, color="k", lw=0.8, ls=":")
ax.set_xlabel(r"tokens since the switch  $\tau$")
ax.set_ylabel("belief on missed neighbor $-$ mean far state")
ax.set_title("mass on the never-visited other neighbor\n"
             "(only the transition map points there)", fontsize=10)
ax.legend(frameon=False, fontsize=9)

# right: full mass budget at the switch token (lag 0)
ax = axes[1]
w = 0.26
xs = np.arange(4)
labels = ["old state $i$", "destination $j$", "missed neighbor",
          "far states\n(mean of 5)"]
for off, (name, color, _) in zip((-w, 0, w), SOURCES):
    m = sums[name][:, 0] / cnt[0]
    ax.bar(xs + off, m, width=w, color=color,
           label=name.split(" (")[0])
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("mean read-out mass at the switch token")
ax.set_title(r"where the belief sits at $\tau = 0$", fontsize=10)
ax.legend(frameon=False, fontsize=9)

fig.tight_layout()
fig.savefig(OUT / "fig_missed_neighbor.png", bbox_inches="tight")
plt.close(fig)
shutil.copy(OUT / "fig_missed_neighbor.png",
            Path.home() / "Desktop" / "fig_missed_neighbor.png")
print("wrote", OUT / "fig_missed_neighbor.png", "and copied to Desktop")
