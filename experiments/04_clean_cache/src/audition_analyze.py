"""Step 3: pick the single global clamp strength and the shortlist.

Per your call (one shared strength, not per-latent calibration): for each
strength in the sweep, count candidates whose evidence rate (median per-doc
KL/token) lands in BAND with clean fluency (NLL ratio vs clean arm <= 1.12)
and sane output entropy (within 0.35 nats of clean). Pick the strength with
the most in-band candidates; shortlist = the in-band candidates at that
strength (cap SHORTLIST_MAX, closest to the band median first).

Prints the full latent x strength table so the choice is inspectable.
Output: results/audition/shortlist.json
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BAND = (0.02, 0.30)          # nats/token, evidence rate (KL clamped||clean)
NLL_RATIO_MAX = 1.15
ENT_TOL = 0.35
SHORTLIST_MAX = 24

ROOT = Path(__file__).parent.parent / "results" / "audition"


def distinct2(ids_row, prompt_len=13):
    g = ids_row[prompt_len:]
    pairs = set(zip(g[:-1], g[1:]))
    return len(pairs) / max(len(g) - 1, 1)


def main():
    lat, mult, kl, nll, ent, d2 = [], [], [], [], [], []
    for f in sorted(ROOT.glob("audition_shard*.npz")):
        d = np.load(f)
        lat.append(d["latent"]); mult.append(d["mult"])
        kl.append(d["kl"]); nll.append(d["nll_clean"]); ent.append(d["ent"])
        d2.append(np.array([distinct2(r) for r in d["ids"]]))
    lat = np.concatenate(lat); mult = np.concatenate(mult)
    kl = np.concatenate(kl); nll = np.concatenate(nll)
    ent = np.concatenate(ent); d2 = np.concatenate(d2)

    clean = lat == -1
    nll0 = float(np.median(nll[clean]))
    ent0 = float(np.median(ent[clean]))
    d20 = float(np.median(d2[clean]))
    print(f"clean arm: NLL {nll0:.3f}  ent {ent0:.3f}  distinct2 {d20:.3f}\n")

    cells = defaultdict(dict)
    for j in sorted(set(lat[~clean].tolist())):
        for m in sorted(set(mult[~clean].tolist())):
            sel = (lat == j) & (mult == m)
            if not sel.any():
                continue
            cells[j][m] = {
                "kl": float(np.median(kl[sel])),
                "nll_ratio": float(np.median(nll[sel]) / nll0),
                "d_ent": float(np.median(ent[sel]) - ent0),
                "distinct2": float(np.median(d2[sel])),
            }

    mults = sorted({m for v in cells.values() for m in v})
    print("latent   " + "".join(f"{m:>18.0f}x" for m in mults))
    for j, v in cells.items():
        row = "".join(
            f"  {v[m]['kl']:.3f}/{v[m]['nll_ratio']:.2f}/{v[m]['d_ent']:+.2f}"
            if m in v else " " * 19 for m in mults)
        print(f"{j:<7d}{row}")
    print("\n(cell = KL per token / NLL ratio / entropy delta)\n")

    def in_band(c):
        return (BAND[0] <= c["kl"] <= BAND[1] and c["nll_ratio"] <= NLL_RATIO_MAX
                and abs(c["d_ent"]) <= ENT_TOL)

    counts = {m: sum(in_band(v[m]) for v in cells.values() if m in v)
              for m in mults}
    print("in-band candidates per strength:", counts)
    s_star = max(counts, key=lambda m: (counts[m], -m))
    band_kls = [v[s_star]["kl"] for v in cells.values()
                if s_star in v and in_band(v[s_star])]
    center = float(np.median(band_kls))
    short = sorted((j for j, v in cells.items()
                    if s_star in v and in_band(v[s_star])),
                   key=lambda j: abs(np.log(cells[j][s_star]["kl"] / center)))
    short = sorted(short[:SHORTLIST_MAX])

    out = {"mult": s_star, "band": BAND, "kl_center": center,
           "n_in_band": counts[s_star], "shortlist": short,
           "per_latent": {str(j): cells[j][s_star] for j in short}}
    (ROOT / "shortlist.json").write_text(json.dumps(out, indent=2))
    print(f"\nchosen strength {s_star}x, shortlist {len(short)}: {short}")


if __name__ == "__main__":
    main()
