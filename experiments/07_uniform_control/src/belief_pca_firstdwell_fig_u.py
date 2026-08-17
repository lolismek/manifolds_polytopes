"""Blog figure: first-dwell-only version of the three-panel belief-state
PCA (observer b_t, ring student b_hat, control2 cross b_hat). Purely local:
filters the existing belief_viz dumps to tokens before each doc's first
state switch (carryover-free), then reuses exp06's pca_scatter.

Writes results/blog/fig_belief_pca_3_firstdwell.png + copies to ~/Desktop.
"""
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
EXP06_SRC = ROOT.parent / "06_short_dwell" / "src"
sys.path.insert(0, str(EXP06_SRC))
import belief_pca_figure as bpf  # noqa: E402
from firstdwell_probe import first_dwell_mask  # noqa: E402

EXP06_PROBE = ROOT.parent / "06_short_dwell" / "results" / "probe"
OUT = ROOT / "results" / "blog"

d6 = np.load(EXP06_PROBE / "belief_viz.npz")
d7 = np.load(ROOT / "results" / "probe" / "uniform_belief_viz_cross.npz")
z = d6["z"].astype(int)
assert np.array_equal(z, d7["z"].astype(int))

m = (first_dwell_mask(z) >= 0).reshape(-1)
zf = z.reshape(-1)[m]
print(f"first-dwell tokens: {m.sum()} of {m.size}")
bpf.N_SCATTER = min(20000, int(m.sum()))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
bpf.pca_scatter(axes[0], d6["b"].astype(np.float32).reshape(-1, bpf.K)[m],
                zf, r"optimal observer  $b_t$")
bpf.pca_scatter(axes[1], d6["bhat"].astype(np.float32).reshape(-1, bpf.K)[m],
                zf, rf"student probe read-out  $\hat{{b}}_t$"
                rf"  (layer {int(d6['layer'])})")
bpf.pca_scatter(axes[2], d7["bhat"].astype(np.float32).reshape(-1, bpf.K)[m],
                zf, rf"control2 probe read-out  $\hat{{b}}_t$"
                rf"  (layer {int(d7['layer'])})")
axes[0].set_ylabel("PC 2")
fig.suptitle("first dwell only (carryover-free)", fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig_belief_pca_3_firstdwell.png", bbox_inches="tight")
plt.close(fig)
shutil.copy(OUT / "fig_belief_pca_3_firstdwell.png",
            Path.home() / "Desktop" / "fig_belief_pca_3_firstdwell.png")
print("wrote", OUT / "fig_belief_pca_3_firstdwell.png",
      "and copied to Desktop")
