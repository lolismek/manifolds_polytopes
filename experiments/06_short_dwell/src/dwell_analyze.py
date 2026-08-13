"""Reader arithmetic over dwell pilots: how much next-token loss does each
kind of knowledge buy, at each dwell length?

Readers (all consume the exact per-state emission table logp (n, 8, T); a
reader's next-token prediction is its prior-weighted mixture of the 8
state-conditional distributions, so its loss is directly comparable to what
a student LM could gain by tracking the state that way):

  oracle   knows the true state z_t outright (ceiling, not achievable)
  ring     full Bayes: knows switch rate AND ring wiring (the ideal reader)
  nomap    Bayes with the same switch rate but switches spread evenly over
           all 7 other states — knows "states change" but not the map
  forget   decayed evidence sum softmax(sum_t gamma^dt logp_t), best gamma
           from a grid — the "running average" reader the exp05 student
           resembles; no transition knowledge at all
  none     uniform belief always (no tracking)

The money numbers, in nats/token:
  wiring value   = loss(nomap) - loss(ring)    <- what the RING is worth
  memory value   = loss(none)  - loss(forget)  <- what integration is worth
  tracking value = loss(none)  - loss(ring)    <- total state-tracking value

Usage:
  python dwell_analyze.py results/pilot/p95.npz ... \
      --baseline ../05_multilayer_add/results/posterior/eval_shard0.npz
(the baseline is exp05's scored eval split, p_stay 0.98; first --n_docs
docs are used so every row rests on comparable token counts)
"""

import argparse
import json
from pathlib import Path

import numpy as np

K = 8
GAMMAS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]


def ring_T(p):
    T = np.zeros((K, K))
    for i in range(K):
        T[i, i] = p
        T[i, (i + 1) % K] = T[i, (i - 1) % K] = (1 - p) / 2
    return T


def nomap_T(p):
    T = np.full((K, K), (1 - p) / (K - 1))
    np.fill_diagonal(T, p)
    return T


def logsumexp(a, axis):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def bayes_reader(logp, z, T_mat):
    """logp (n, K, T) float64 -> per-token predictive loss, filtered acc,
    mean max filtered belief."""
    n, _, T = logp.shape
    log_prior = np.full((n, K), -np.log(K))
    logT = np.log(T_mat + 1e-300)
    loss = np.empty((n, T))
    hit = np.empty((n, T), dtype=bool)
    maxb = np.empty((n, T))
    for t in range(T):
        joint = log_prior + logp[:, :, t]
        loss[:, t] = -logsumexp(joint, 1)
        la = joint - logsumexp(joint, 1)[:, None]      # filtered posterior
        hit[:, t] = la.argmax(1) == z[:, t]
        maxb[:, t] = np.exp(la.max(1))
        log_prior = logsumexp(la[:, :, None] + logT[None], 1)
    return loss.mean(), hit.mean(), maxb.mean()


def forget_reader(logp, gamma):
    n, _, T = logp.shape
    e = np.zeros((n, K))
    loss = np.empty((n, T))
    for t in range(T):
        prior = e - logsumexp(e, 1)[:, None]
        loss[:, t] = -logsumexp(prior + logp[:, :, t], 1)
        e = gamma * e + logp[:, :, t]
    return loss.mean()


def analyze(path, p_stay, n_docs):
    d = np.load(path)
    logp = d["logp"][:n_docs].astype(np.float64)
    z = d["z"][:n_docs, :logp.shape[2]].astype(np.int64)

    oracle = -logp[np.arange(len(z))[:, None], z,
                   np.arange(logp.shape[2])[None]].mean()
    none = -(logsumexp(logp, 1) - np.log(K)).mean()
    l_ring, acc_ring, maxb = bayes_reader(logp, z, ring_T(p_stay))
    l_nomap, acc_nomap, _ = bayes_reader(logp, z, nomap_T(p_stay))
    fg = {g: forget_reader(logp, g) for g in GAMMAS}
    g_best = min(fg, key=fg.get)

    r = lambda x: round(float(x), 4)  # noqa: E731  (np floats break json)
    return {"p_stay": p_stay, "dwell": r(1 / (1 - p_stay)),
            "n_docs": int(len(z)), "n_tokens": int(z.size),
            "loss_oracle": r(oracle), "loss_ring": r(l_ring),
            "loss_nomap": r(l_nomap),
            "loss_forget": r(fg[g_best]), "forget_gamma": g_best,
            "loss_none": r(none),
            "wiring_value": r(l_nomap - l_ring),
            "memory_value": r(none - fg[g_best]),
            "tracking_value": r(none - l_ring),
            "acc_ring": r(acc_ring), "acc_nomap": r(acc_nomap),
            "mean_max_belief": r(maxb)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pilots", nargs="*")
    ap.add_argument("--baseline", default=None,
                    help="exp05 eval shard npz (p_stay 0.98)")
    ap.add_argument("--n_docs", type=int, default=224)
    args = ap.parse_args()

    rows = []
    if args.baseline:
        rows.append(analyze(args.baseline, 0.98, args.n_docs))
    for p in args.pilots:
        rows.append(analyze(p, float(np.load(p)["p_stay"]), args.n_docs))
    rows.sort(key=lambda r: -r["p_stay"])

    cols = ["p_stay", "dwell", "loss_oracle", "loss_ring", "loss_nomap",
            "loss_forget", "forget_gamma", "loss_none", "wiring_value",
            "memory_value", "tracking_value", "acc_ring", "acc_nomap",
            "mean_max_belief"]
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r[c]) for c in cols))

    out = Path(__file__).parent.parent / "results" / "pilot" / "readers.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
