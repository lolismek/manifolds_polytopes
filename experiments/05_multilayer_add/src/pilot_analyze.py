"""Pilot readout: per s — fluency (KL/token, NLL ratio, entropy delta,
distinct-2) and the outcome numbers (min/median pairwise margin, worst pair,
per-token noise std, evidence vs clean). Full margin matrix for the best s.

Gate: min pairwise margin >= ~0.3 nats/token with NLL ratio <= 1.15.
"""

from pathlib import Path

import numpy as np

OUT = Path(__file__).parent.parent / "results" / "pilot"
NLL_RATIO_MAX = 1.15


def distinct2(row, prompt_len=13):
    g = row[prompt_len:]
    return len(set(zip(g[:-1], g[1:]))) / max(len(g) - 1, 1)


def main():
    g = {k: [] for k in ("ids", "latent", "mult", "kl", "nll_clean", "ent")}
    for f in sorted(OUT.glob("gen_shard*.npz")):
        d = np.load(f)
        for k in g:
            g[k].append(d[k])
    g = {k: np.concatenate(v) for k, v in g.items()}
    d2 = np.array([distinct2(r) for r in g["ids"]])

    clean = g["latent"] == -1
    nll0 = np.median(g["nll_clean"][clean])
    ent0 = np.median(g["ent"][clean])
    print(f"clean arm: NLL {nll0:.3f}  ent {ent0:.3f}  "
          f"distinct2 {np.median(d2[clean]):.3f}\n")

    s = {k: [] for k in ("doc_latent", "doc_mult", "logp", "logp_clean")}
    for f in sorted(OUT.glob("score_shard*.npz")):
        d = np.load(f)
        for k in s:
            s[k].append(d[k])
        hyp = d["hyp_latents"]
    s = {k: np.concatenate(v) for k, v in s.items()}

    best = None
    for m in sorted(set(g["mult"][~clean])):
        sel = (g["mult"] == m) & ~clean
        kl_m = np.median(g["kl"][sel])
        nllr = np.median(g["nll_clean"][sel]) / nll0
        dent = np.median(g["ent"][sel]) - ent0

        K = len(hyp)
        M = np.zeros((K, K))
        noise, ev = [], []
        for a, j in enumerate(hyp):
            dsel = (s["doc_latent"] == j) & (s["doc_mult"] == m)
            lp = s["logp"][dsel]                      # (n, K, T)
            diff = lp[:, a : a + 1] - lp
            M[a] = diff.mean((0, 2))
            noise.append(diff.std(2)[:, np.arange(K) != a].mean())
            ev.append((lp[:, a] - s["logp_clean"][dsel]).mean())
        off = M[~np.eye(K, dtype=bool)]
        wa, wb = np.unravel_index(
            np.argmin(M + np.eye(K) * 1e9), M.shape)
        print(f"s={m:.1f}x  KL/tok {kl_m:.3f}  NLLr {nllr:.2f}  "
              f"dent {dent:+.2f}  d2 {np.median(d2[sel]):.3f}  |  "
              f"margin min {off.min():.3f} med {np.median(off):.3f} "
              f"(worst {hyp[wa]} vs {hyp[wb]})  noise_std {np.mean(noise):.2f}  "
              f"evid {np.mean(ev):.3f}")
        ok = off.min() >= 0.3 and nllr <= NLL_RATIO_MAX
        if best is None or (ok and not best[2]) or \
                (ok == best[2] and off.min() > best[3]):
            best = (m, M, ok, off.min())

    m, M, ok, _ = best
    verdict = "GATE PASSED" if ok else "GATE NOT PASSED — best cell"
    print(f"\n{verdict} at s={m:.1f}x. Margin matrix (row=true, col=hyp):")
    print("        " + "".join(f"{j:>8d}" for j in hyp))
    for a, j in enumerate(hyp):
        print(f"{j:>7d} " + "".join(f"{M[a, b]:8.3f}" for b in range(len(hyp))))


if __name__ == "__main__":
    main()
