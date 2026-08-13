"""Ring-mode Fourier test (user's 'finer than PCA' instrument).

Treat the 8 per-state vectors (first-dwell class means, or one-hot probe
columns) as a function on the ring and decompose the state index into
cyclic frequency components. A circle-in-ring-order is exactly frequency 1;
PCA only sees it if it dominates variance, this measures its energy
fraction directly. Frequencies 3-4 are neighbor-alternating patterns — the
signature of neighbor-separation (the anti-ring).

Significance: permutation test over all 2520 distinct ring orderings of 8
states (rotations/reflections identified). p_f1 = fraction of orderings
whose frequency-1 energy fraction >= the true ring's (small p = the circle
is aligned with the TRUE ring); p_f34 likewise for freq 3+4 energy (small
p = neighbor-alternation aligned with the true ring).

For 8 generic points the expected fractions are f1=f2=f3=2/7~0.29, f4=1/7.

Reads results/probe/firstdwell{,_late}_arrays.npz; writes
results/probe/fourier_ringmode.json. Usage: python fourier_ringmode.py
"""

import itertools
import json
from pathlib import Path

import numpy as np

K = 8
ROOT = Path(__file__).parent.parent / "results" / "probe"


_k = np.arange(K)
_P = [np.exp(-2j * np.pi * f * (_k[:, None] - _k[None, :]) / K)
      * (1 if f == 4 else 2)                   # f and K-f are conjugates
      for f in range(1, 5)]


def spectrum(G):
    """G (K,K) Gram of centered vectors -> energy fraction per frequency
    1..4 (conjugate pairs combined)."""
    E = np.array([float(np.real((P * G).sum())) for P in _P])
    return E / E.sum()


def main():
    orderings = [(0,) + p for p in itertools.permutations(range(1, K))
                 if p[0] < p[-1]]
    assert len(orderings) == 2520

    out = {}
    for variant in ("firstdwell", "firstdwell_late"):
        d = np.load(ROOT / f"{variant}_arrays.npz")
        out[variant] = {}
        for student in ("ring", "ctrl"):
            out[variant][student] = {}
            for arr in ("means", "onehot_cols"):
                M = d[f"{student}_{arr}"]           # (L, K, dim)
                if arr == "onehot_cols":
                    M = M / np.linalg.norm(M, axis=2, keepdims=True)
                recs = []
                for l in range(M.shape[0]):
                    X = M[l] - M[l].mean(0)
                    G = X @ X.T
                    fr = spectrum(G)
                    f1s, f34s = [], []
                    for o in orderings:
                        fo = spectrum(G[np.ix_(o, o)])
                        f1s.append(fo[0])
                        f34s.append(fo[2] + fo[3])
                    f1s, f34s = np.array(f1s), np.array(f34s)
                    recs.append({
                        "layer": l,
                        "frac_f1234": [round(float(x), 4) for x in fr],
                        "p_f1": round(float((f1s >= fr[0]).mean()), 4),
                        "p_f34": round(float(
                            (f34s >= fr[2] + fr[3]).mean()), 4)})
                out[variant][student][arr] = recs
                for r in recs:
                    if r["layer"] in (0, 3, 9, 10, 11, 12):
                        print(f"{variant:>15} {student:>4} {arr:>11} "
                              f"L{r['layer']:>2} f1..4 {r['frac_f1234']} "
                              f"p_f1 {r['p_f1']:>6} p_f34 {r['p_f34']:>6}",
                              flush=True)

    (ROOT / "fourier_ringmode.json").write_text(json.dumps(out, indent=1))
    print("wrote fourier_ringmode.json", flush=True)


if __name__ == "__main__":
    main()
