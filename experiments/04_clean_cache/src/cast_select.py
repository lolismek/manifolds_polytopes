"""Step 4b: build the pairwise margin matrix and select the cast of 8.

Margin M[i, j] = mean over docs generated under latent i, tokens, of
log P(token | z=i) - log P(token | z=j): how much per-token likelihood
evidence separates i's text from hypothesis j (nats/token, exact reader).
Evidence E[i] = mean log P(token | z=i) - log P_clean(token).

Selection: brute-force all C(n_short, 8) subsets, maximize the minimum
ordered-pair margin; among the top candidates, prefer matched evidence rates
(lowest spread of E) and uniform margins. Reserves: greedy additions that
keep the highest min-margin to the chosen cast.

Output: results/confusability/cast.json + casting_report.md
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np

K = 8
N_RESERVES = 4
TOP_POOL = 200            # top subsets by min-margin kept for the tie-break

ROOT = Path(__file__).parent.parent / "results" / "confusability"


def main():
    lat, lp, lpc = [], [], []
    hyp = None
    for f in sorted(ROOT.glob("score_shard*.npz")):
        d = np.load(f)
        lat.append(d["doc_latent"]); lp.append(d["logp"])
        lpc.append(d["logp_clean"]); hyp = d["hyp_latents"]
    lat = np.concatenate(lat); lp = np.concatenate(lp)
    lpc = np.concatenate(lpc)
    H = len(hyp)
    pos = {int(j): k for k, j in enumerate(hyp)}

    M = np.zeros((H, H))
    E = np.zeros(H)
    for j in hyp:
        i = pos[int(j)]
        sel = lat == j
        own = lp[sel, i, :]                       # (n_docs, T)
        M[i] = (own[:, None, :] - lp[sel]).mean((0, 2))
        E[i] = (own - lpc[sel]).mean()

    subs = np.array(list(combinations(range(H), K)))
    pairs = [(a, b) for a in range(K) for b in range(K) if a != b]
    mins = np.full(len(subs), np.inf)
    for a, b in pairs:
        mins = np.minimum(mins, M[subs[:, a], subs[:, b]])
    order = np.argsort(-mins)[:TOP_POOL]

    best, best_key = None, None
    for s in order:
        idx = subs[s]
        e_spread = float(E[idx].std())
        sub_m = np.array([M[i, j] for i in idx for j in idx if i != j])
        key = (-mins[s] + 0.5 * e_spread + 0.1 * sub_m.std(),)
        if best_key is None or key < best_key:
            best, best_key = s, key
    cast_idx = subs[best]
    cast = sorted(int(hyp[i]) for i in cast_idx)

    chosen = set(cast_idx.tolist())
    reserves = []
    for _ in range(N_RESERVES):
        rest = [i for i in range(H) if i not in chosen and i not in reserves]
        if not rest:
            break
        score = [min(min(M[i, j], M[j, i]) for j in chosen) for i in rest]
        reserves.append(rest[int(np.argmax(score))])
    reserves_ids = sorted(int(hyp[i]) for i in reserves)

    exp03_cast = [368, 1220, 2404, 2970, 6172, 10615, 10621, 13931]
    rep = {
        "cast": cast, "reserves": reserves_ids,
        "min_pairwise_margin": float(mins[best]),
        "evidence_rates": {int(hyp[i]): round(float(E[i]), 4)
                           for i in cast_idx},
        "overlap_with_exp03_cast": sorted(set(cast) & set(exp03_cast)),
        "margin_matrix_latents": [int(j) for j in hyp],
        "margin_matrix": np.round(M, 4).tolist(),
        "evidence_all": {int(j): round(float(E[pos[int(j)]]), 4) for j in hyp},
    }
    (ROOT / "cast.json").write_text(json.dumps(rep, indent=2))

    lines = ["# exp04 casting report (exact-likelihood confusability)", "",
             f"Cast of 8: {cast}", f"Reserves: {reserves_ids}",
             f"Min pairwise margin: {mins[best]:.4f} nats/token",
             f"Overlap with exp03 cast: {rep['overlap_with_exp03_cast']}", "",
             "Evidence rate (nats/token vs clean) per cast member:"]
    for j in cast:
        lines.append(f"  {j}: {E[pos[j]]:.4f}")
    lines += ["", "Cast pairwise margins M[i,j] (row = generator, col = "
              "hypothesis):", "      " + "".join(f"{j:>8d}" for j in cast)]
    for i in cast:
        lines.append(f"{i:>6d}" + "".join(f"{M[pos[i], pos[j]]:>8.3f}"
                                          for j in cast))
    (ROOT / "casting_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
