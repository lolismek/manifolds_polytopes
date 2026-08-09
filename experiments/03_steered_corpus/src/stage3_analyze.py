"""Stage 3 analysis: fluency and separability of the audition generations.

Aggregates gen_shard*.npz and reports, per strength:
  - fluency: median per-seq NLL under the unsteered teacher vs. the clean arm
  - separability (mini-M): 48-way bag-of-words logistic classifier on
    continuations, accuracy as a function of token budget (evidence-rate curve)
Per latent: NLL ratio at each strength and test recall at the top strength
(quiet or broken latents show up here).

Output: results/stage3/audition_report.json

Usage (on tigerfish):
  python stage3_analyze.py --device cuda:0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

MULTS = [1.0, 2.0, 3.0, 5.0]
BUDGETS = [50, 100, 200, 300]
TRAIN_PER_CELL = 15     # of 20 samples per (latent, strength); rest is test
STEPS = 400
LR = 0.05
SEED = 0


def bow(ids, vocab_index, n_vocab, budget):
    """log(1+count) bag-of-words over the first `budget` tokens."""
    n = ids.shape[0]
    x = np.zeros((n, n_vocab), dtype=np.float32)
    for i in range(n):
        for t in ids[i, :budget]:
            k = vocab_index.get(int(t))
            if k is not None:
                x[i, k] += 1
    return np.log1p(x)


def fit_logistic(xtr, ytr, xte, dev):
    torch.manual_seed(SEED)
    xtr = torch.tensor(xtr, device=dev)
    ytr = torch.tensor(ytr, device=dev)
    xte = torch.tensor(xte, device=dev)
    n_cls = int(ytr.max()) + 1
    w = torch.zeros(xtr.shape[1], n_cls, device=dev, requires_grad=True)
    b = torch.zeros(n_cls, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=LR)
    for _ in range(STEPS):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(xtr @ w + b, ytr) \
            + 1e-4 * (w * w).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return (xte @ w + b).argmax(1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)

    out = Path(__file__).parent.parent / "results" / "stage3"
    shards = sorted(out.glob("gen_shard*.npz"))
    ids = np.concatenate([np.load(s)["ids"] for s in shards])
    lat = np.concatenate([np.load(s)["latent"] for s in shards])
    mult = np.concatenate([np.load(s)["mult"] for s in shards])
    nll = np.concatenate([np.load(s)["nll"] for s in shards])

    clean_nll = float(np.median(nll[lat == -1]))
    pool = sorted(set(lat[lat >= 0].tolist()))
    cls = {j: c for c, j in enumerate(pool)}

    # vocab = tokens seen in the data (keeps BoW matrices small)
    uniq = np.unique(ids)
    vocab_index = {int(t): k for k, t in enumerate(uniq)}
    n_vocab = len(uniq)

    report = {"clean_nll_median": clean_nll, "n_pool": len(pool),
              "n_seqs": int((lat >= 0).sum()), "per_strength": {}, "per_latent": {}}

    rng = np.random.default_rng(SEED)
    recall_top = {}
    for m in MULTS:
        sel = np.where((lat >= 0) & (mult == m))[0]
        y = np.array([cls[int(j)] for j in lat[sel]])
        # split within each cell
        tr_idx, te_idx = [], []
        for c in range(len(pool)):
            rows = sel[y == c]
            perm = rng.permutation(len(rows))
            tr_idx += rows[perm[:TRAIN_PER_CELL]].tolist()
            te_idx += rows[perm[TRAIN_PER_CELL:]].tolist()
        tr_idx, te_idx = np.array(tr_idx), np.array(te_idx)
        ytr = np.array([cls[int(j)] for j in lat[tr_idx]])
        yte = np.array([cls[int(j)] for j in lat[te_idx]])

        acc_by_budget = {}
        for budget in BUDGETS:
            xtr = bow(ids[tr_idx], vocab_index, n_vocab, budget)
            xte = bow(ids[te_idx], vocab_index, n_vocab, budget)
            pred = fit_logistic(xtr, ytr, xte, dev)
            acc_by_budget[budget] = round(float((pred == yte).mean()), 4)
            if m == MULTS[-1] and budget == BUDGETS[-1]:
                for c, j in enumerate(pool):
                    recall_top[j] = round(float((pred[yte == c] == c).mean()), 3)

        med_nll = float(np.median(nll[sel]))
        report["per_strength"][str(m)] = {
            "median_nll": round(med_nll, 4),
            "nll_ratio_vs_clean": round(med_nll / clean_nll, 4),
            "acc_by_budget": acc_by_budget,
            "chance": round(1 / len(pool), 4),
        }
        print(f"mult {m}: nll_ratio {med_nll / clean_nll:.3f}, acc {acc_by_budget}", flush=True)

    for j in pool:
        row = {"recall_at_top_strength": recall_top.get(j)}
        for m in MULTS:
            s = nll[(lat == j) & (mult == m)]
            row[f"nll_ratio_{m}x"] = round(float(np.median(s)) / clean_nll, 4)
        report["per_latent"][str(j)] = row

    (out / "audition_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["per_strength"], indent=2))


if __name__ == "__main__":
    main()
