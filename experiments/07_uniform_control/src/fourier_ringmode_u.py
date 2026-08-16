"""Ring-mode Fourier test for the exp07 uniform student (exp06's
instrument): decompose the 8 per-state vectors (first-dwell class means /
one-hot probe columns) into cyclic frequency components in CAST ring
order, permutation-tested over all 2520 ring orderings.

exp06 ring student: f1 (circle) fraction 0.36-0.38 at L9-12, p 0.0004-0.0016
in the means; f1 suppressed / f3+f4 peaked in the whitened one-hot columns.
Prediction for the uniform student: chance everywhere (like exp03's ctrl).

Reads uniform_firstdwell{,_late}_arrays.npz;
writes uniform_fourier_ringmode.json. Usage: python fourier_ringmode_u.py
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np

EXP06_SRC = Path(__file__).parents[2] / "06_short_dwell" / "src"
sys.path.insert(0, str(EXP06_SRC))
from fourier_ringmode import spectrum  # noqa: E402

K = 8
ROOT = Path(__file__).parent.parent / "results" / "probe"


def main():
    orderings = [(0,) + p for p in itertools.permutations(range(1, K))
                 if p[0] < p[-1]]
    assert len(orderings) == 2520

    out = {}
    for variant in ("uniform_firstdwell", "uniform_firstdwell_late"):
        d = np.load(ROOT / f"{variant}_arrays.npz")
        out[variant] = {}
        for arr in ("means", "onehot_cols"):
            M = d[f"uniform_{arr}"]                 # (L, K, dim)
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
            out[variant][arr] = recs
            for r in recs:
                if r["layer"] in (0, 3, 9, 10, 11, 12):
                    print(f"{variant:>25} {arr:>11} "
                          f"L{r['layer']:>2} f1..4 {r['frac_f1234']} "
                          f"p_f1 {r['p_f1']:>6} p_f34 {r['p_f34']:>6}",
                          flush=True)

    (ROOT / "uniform_fourier_ringmode.json").write_text(
        json.dumps(out, indent=1))
    print("wrote uniform_fourier_ringmode.json", flush=True)


if __name__ == "__main__":
    main()
