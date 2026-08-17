"""Blog figure: three-panel belief-state PCA — optimal observer b_t, ring
student probe read-out b_hat, and control2's read-out on the SAME ring eval
docs (cross condition). Panels 1-2 reuse exp06's belief_viz.npz; panel 3
needs uniform_belief_viz_cross.npz (dump_belief_viz_u.py on seahorse).

Writes results/blog/fig_belief_pca_3.png and copies it to ~/Desktop.
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
from belief_pca_figure import K, pca_scatter  # noqa: E402

EXP06_PROBE = ROOT.parent / "06_short_dwell" / "results" / "probe"
OUT = ROOT / "results" / "blog"

cross_path = ROOT / "results" / "probe" / "uniform_belief_viz_cross.npz"
if not cross_path.exists():
    sys.exit("missing uniform_belief_viz_cross.npz — run "
             "dump_belief_viz_u.py on seahorse and scp it back first")

d6 = np.load(EXP06_PROBE / "belief_viz.npz")
d7 = np.load(cross_path)
zf = d6["z"].astype(int).reshape(-1)
assert np.array_equal(zf, d7["z"].astype(int).reshape(-1))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
pca_scatter(axes[0], d6["b"].astype(np.float32).reshape(-1, K), zf,
            r"optimal observer  $b_t$")
pca_scatter(axes[1], d6["bhat"].astype(np.float32).reshape(-1, K), zf,
            rf"student probe read-out  $\hat{{b}}_t$"
            rf"  (layer {int(d6['layer'])})")
pca_scatter(axes[2], d7["bhat"].astype(np.float32).reshape(-1, K), zf,
            rf"control2 probe read-out  $\hat{{b}}_t$"
            rf"  (layer {int(d7['layer'])})")
axes[0].set_ylabel("PC 2")
fig.tight_layout()
fig.savefig(OUT / "fig_belief_pca_3.png", bbox_inches="tight")
plt.close(fig)
shutil.copy(OUT / "fig_belief_pca_3.png",
            Path.home() / "Desktop" / "fig_belief_pca_3.png")
print("wrote", OUT / "fig_belief_pca_3.png", "and copied to Desktop")
