"""Integrity checks for the exp07 uniform corpus (run after generation).

- doc/token counts (expect 400,000 / 102,400,000)
- switch rate ~12/doc; jump destinations uniform over the 7 non-self
  offsets (chi-square-ish eyeball: per-offset fractions)
- switch TIMES identical to the exp06 ring corpus (same seed + draw order;
  checked on every chunk pair present locally)
- vocab coverage of exp06's vocab32k map on the training chunks

Usage: python corpus_check.py
"""

import glob
from pathlib import Path

import numpy as np

K = 8
ROOT = Path(__file__).parent.parent / "results" / "corpus"
RING_ROOT = (Path(__file__).parents[2] / "06_short_dwell" / "results"
             / "corpus")
VOCAB = RING_ROOT / "vocab32k.npz"


def main():
    files = sorted(glob.glob(str(ROOT / "uniform_shard*_chunk*.npz")))
    print(f"{len(files)} chunks (expect 80)")

    n_docs = n_tokens = n_switch = 0
    offset_counts = np.zeros(K, dtype=np.int64)
    occupancy = np.zeros(K, dtype=np.int64)
    kept = unk = 0
    vmap = np.load(VOCAB)["map"]
    mismatched_times = []

    for f in files:
        d = np.load(f)
        ids, z = d["ids"], d["z"]
        n_docs += ids.shape[0]
        n_tokens += ids.size
        dz = np.diff(z.astype(np.int64), axis=1)
        sw = dz != 0
        n_switch += int(sw.sum())
        offset_counts += np.bincount((dz[sw] % K), minlength=K)
        occupancy += np.bincount(z.ravel(), minlength=K)
        if "chunk009" not in f:
            compact = vmap[ids]
            kept += int((compact > 0).sum())
            unk += int((compact == 0).sum())
        rf = RING_ROOT / Path(f).name.replace("uniform_", "ring_")
        if rf.exists():
            zr = np.load(rf)["z"]
            if not ((np.diff(zr, axis=1) != 0) == sw).all():
                mismatched_times.append(Path(f).name)

    print(f"docs {n_docs}, tokens {n_tokens}")
    print(f"switches/doc {n_switch / max(n_docs, 1):.2f}")
    frac = offset_counts / max(offset_counts.sum(), 1)
    print("jump-offset fractions (1..7, expect ~0.143 each; 0 must be 0):")
    print("  " + " ".join(f"{o}:{frac[o]:.4f}" for o in range(K)))
    occ = occupancy / max(occupancy.sum(), 1)
    print("state occupancy (expect ~0.125 each): "
          + " ".join(f"{p:.4f}" for p in occ))
    print(f"vocab32k coverage on training chunks: {kept / max(kept + unk, 1):.4f}")
    if mismatched_times:
        print(f"SWITCH-TIME MISMATCH vs ring corpus in: {mismatched_times}")
    else:
        print("switch times identical to ring corpus in every "
              "locally-available chunk pair")


if __name__ == "__main__":
    main()
