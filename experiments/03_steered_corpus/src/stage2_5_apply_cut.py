"""Apply the stage-2.5 causal cuts to the screen results.

Keep a latent iff, at the 5x clamp:
  - KL(steered || clean) in [KL_MIN, KL_MAX]   (causal, but not a bulldozer:
    the state should tilt the text statistically, not announce itself per token)
  - |steered entropy - clean entropy| < ENT_TOL (re-weighting, not degenerate
    concentration or shattering)

Writes results/stage2_5/causal_pool.json with the kept latent ids and stats.
"""

import json
from pathlib import Path

import numpy as np

KL_MIN = 0.05
KL_MAX = 2.0
ENT_TOL = 0.3

res = Path(__file__).parent.parent / "results" / "stage2_5"
d = np.load(res / "causal_screen.npz")
cand, kl, ent = d["candidates"], d["kl"], d["ent_steered"]
ent_clean = float(d["ent_clean"])

mid = kl[:, 1]
keep = (mid >= KL_MIN) & (mid <= KL_MAX) & (np.abs(ent[:, 1] - ent_clean) < ENT_TOL)

pool = {
    "thresholds": {"KL_MIN": KL_MIN, "KL_MAX": KL_MAX, "ENT_TOL": ENT_TOL},
    "ent_clean": ent_clean,
    "n_screened": int(len(cand)),
    "n_kept": int(keep.sum()),
    "cut_low_kl": int((mid < KL_MIN).sum()),
    "cut_high_kl": int((mid > KL_MAX).sum()),
    "cut_entropy": int(((mid >= KL_MIN) & (mid <= KL_MAX)
                        & (np.abs(ent[:, 1] - ent_clean) >= ENT_TOL)).sum()),
    "pool": [{"latent": int(cand[i]), "kl_2x": round(float(kl[i, 0]), 4),
              "kl_5x": round(float(kl[i, 1]), 4), "kl_10x": round(float(kl[i, 2]), 4),
              "ent_5x": round(float(ent[i, 1]), 3)}
             for i in np.where(keep)[0]],
}
(res / "causal_pool.json").write_text(json.dumps(pool, indent=2))
print(json.dumps({k: v for k, v in pool.items() if k != "pool"}, indent=2))
