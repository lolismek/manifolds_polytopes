"""Compact student vocab for exp06 (same recipe as exp03/04/05): count Gemma
token frequencies over the ring training chunks (000-008 x 8 shards = 72
files, 360,000 docs; chunk009 excluded entirely — it contains the eval
split), keep the top 32,768 ids; compact id 0 = <unk>, 1..32768 = kept ids
by rank.

Output: results/corpus/vocab32k.npz with
  map  (gemma_vocab_size,) int32   gemma id -> compact id (0 if dropped)
  kept (32768,) int32              gemma ids by rank
Run after generation: python build_vocab.py
"""

import glob
from pathlib import Path

import numpy as np

TOP = 32768
ROOT = Path(__file__).parent.parent / "results" / "corpus"


def main():
    files = [f for f in sorted(glob.glob(str(ROOT / "ring_shard*_chunk*.npz")))
             if "chunk009" not in f]
    assert len(files) == 72, f"expected 72 training chunks, found {len(files)}"
    counts = None
    for f in files:
        ids = np.load(f)["ids"].ravel()
        c = np.bincount(ids)
        if counts is None:
            counts = c
        elif len(c) > len(counts):
            c[: len(counts)] += counts; counts = c
        else:
            counts[: len(c)] += c
    kept = np.argsort(counts)[::-1][:TOP].astype(np.int32)
    vmap = np.zeros(len(counts), dtype=np.int32)
    vmap[kept] = np.arange(1, TOP + 1, dtype=np.int32)
    cov = counts[kept].sum() / counts.sum()
    print(f"distinct ids {int((counts > 0).sum())}, top {TOP} cover "
          f"{cov:.4f} of training tokens")
    np.savez(ROOT / "vocab32k.npz", map=vmap, kept=kept)


if __name__ == "__main__":
    main()
