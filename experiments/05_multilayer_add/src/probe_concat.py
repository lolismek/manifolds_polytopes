"""Concatenated-residual probe (Shai et al.'s RRXOR lesson): fit one ridge
probe from the concatenation of all 13 hidden-state layers (13*768 = 9984
dims, per-feature standardized) to the exact posterior. Recovers anything any
layer ever represented linearly, including features overwritten downstream.

Also reports probe argmax accuracy binned by tokens-since-transition (vs the
exact reader's own curve): climbing = evidence integration, flat = static
topic readout. Ring geometry on the concat mid-dwell class means.

Final checkpoints only. Usage:
  python probe_concat.py --student ring --device cuda:2
  python probe_concat.py --student ctrl --device cuda:3
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from probe_sweep import (EXP03_ROOT, K, LAM_SCALE, MICRO, P_LEN, ROOT, SEED,
                         load_eval, middwell_states, ring_metrics)

LAG_BINS = [(0, 5), (5, 10), (10, 20), (20, 35), (35, 50), (50, 10 ** 9)]
FINAL = {"ring": "ckpt_04218.pt", "ctrl": "ckpt_02812.pt"}


def lag_since_transition(z):
    n, T = z.shape
    lag = np.zeros((n, T), dtype=np.int64)
    same = z[:, 1:] == z[:, :-1]
    for t in range(1, T):
        lag[:, t] = np.where(same[:, t - 1], lag[:, t - 1] + 1, 0)
    return lag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", choices=["ring", "ctrl"], required=True)
    ap.add_argument("--device", default="cuda:2")
    args = ap.parse_args()
    dev = torch.device(args.device)

    if args.student == "ring":
        sroot = ROOT / "students" / "ring"
        vmap = np.load(ROOT / "corpus" / "vocab32k.npz")["map"]
    else:
        sroot = EXP03_ROOT / "students" / "ctrl"
        vmap = np.load(EXP03_ROOT / "corpus" / "vocab32k.npz")["map"]
    cfg = LlamaConfig.from_dict(json.loads((sroot / "config.json").read_text()))

    ids, post, z, tr, te = load_eval()
    ids_t = torch.from_numpy(vmap[ids].astype(np.int64))
    ms = middwell_states(z)
    lag = lag_since_transition(z)
    rng = np.random.default_rng(SEED)
    perm_tr = np.arange(len(ids)); perm_te = np.arange(len(ids))
    perm_tr[tr] = tr[rng.permutation(len(tr))]
    perm_te[te] = te[rng.permutation(len(te))]

    model = LlamaForCausalLM(cfg)
    sd = torch.load(sroot / FINAL[args.student], map_location="cpu")
    model.load_state_dict({k: v.float() for k, v in sd.items()})
    model = model.to(dev).eval()
    D = (cfg.num_hidden_layers + 1) * cfg.hidden_size
    print(f"{args.student}: concat dim {D}", flush=True)

    def batches(idx):
        for i in range(0, len(idx), MICRO):
            b = idx[i : i + MICRO]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                hs = model(ids_t[b].to(dev), output_hidden_states=True).hidden_states
            X = torch.cat([h[:, P_LEN:, :].float() for h in hs], -1)
            yield b, X.reshape(-1, D)

    G = torch.zeros(D, D, dtype=torch.float64, device=dev)
    XtY = torch.zeros(D, K, dtype=torch.float64, device=dev)
    XtYs = torch.zeros(D, K, dtype=torch.float64, device=dev)
    sx = torch.zeros(D, dtype=torch.float64, device=dev)
    sy = torch.zeros(K, dtype=torch.float64, device=dev)
    csum = torch.zeros(K, D, dtype=torch.float64, device=dev)
    ccount = torch.zeros(K, dtype=torch.float64, device=dev)
    n = 0
    for b, X in batches(tr):
        Y = torch.from_numpy(post[b].reshape(-1, K)).to(dev)
        Ys = torch.from_numpy(post[perm_tr[b]].reshape(-1, K)).to(dev)
        msb = torch.from_numpy(ms[b].reshape(-1)).to(dev)
        G += (X.T @ X).double()
        XtY += (X.T @ Y).double()
        XtYs += (X.T @ Ys).double()
        sx += X.sum(0).double(); sy += Y.sum(0).double(); n += X.shape[0]
        for k in range(K):
            m = msb == k
            if m.any():
                csum[k] += X[m].sum(0).double(); ccount[k] += m.sum()

    mx, my = sx / n, sy / n
    Gc = G - n * torch.outer(mx, mx)
    # per-feature standardization so one ridge lam treats all layers equally
    s = 1.0 / torch.sqrt(torch.diagonal(Gc) / n + 1e-12)
    A = (Gc * s[:, None]) * s[None, :]
    lam = LAM_SCALE * torch.diagonal(A).mean()
    A += lam * torch.eye(D, dtype=torch.float64, device=dev)
    By = (XtY - n * torch.outer(mx, my)) * s[:, None]
    Bys = (XtYs - n * torch.outer(mx, my)) * s[:, None]
    W = (torch.linalg.solve(A, By) * s[:, None]).float()
    Ws = (torch.linalg.solve(A, Bys) * s[:, None]).float()
    mx, my = mx.float(), my.float()
    del G, XtY, XtYs, A, By, Bys
    torch.cuda.empty_cache()

    nb = len(LAG_BINS)
    ss_res = 0.0; ss_res_s = 0.0
    y_sum = torch.zeros(K, dtype=torch.float64, device=dev)
    y_sq = 0.0; n_te = 0
    acc_lag = np.zeros(nb); acc_lag_reader = np.zeros(nb); n_lag = np.zeros(nb)
    for b, X in batches(te):
        Y = torch.from_numpy(post[b].reshape(-1, K)).to(dev)
        Ys = torch.from_numpy(post[perm_te[b]].reshape(-1, K)).to(dev)
        zt = torch.from_numpy(z[b].reshape(-1)).to(dev)
        lg = lag[b].reshape(-1)
        pred = (X - mx) @ W + my
        ss_res += float(((pred - Y) ** 2).sum())
        ss_res_s += float((((X - mx) @ Ws + my - Ys) ** 2).sum())
        y_sum += Y.sum(0).double(); y_sq += float((Y ** 2).sum()); n_te += Y.shape[0]
        hit = (pred.argmax(1) == zt).cpu().numpy()
        hit_r = (Y.argmax(1) == zt).cpu().numpy()
        for i, (lo, hi) in enumerate(LAG_BINS):
            m = (lg >= lo) & (lg < hi)
            acc_lag[i] += hit[m].sum(); acc_lag_reader[i] += hit_r[m].sum()
            n_lag[i] += m.sum()

    y_mean = y_sum / n_te
    ss_tot = y_sq - n_te * float((y_mean ** 2).sum())
    means = (csum / ccount[:, None]).cpu().numpy()
    rec = {"student": args.student, "ckpt": FINAL[args.student],
           "probe_R2": round(1 - ss_res / ss_tot, 4),
           "probe_R2_shuffled": round(1 - ss_res_s / ss_tot, 4),
           "acc_vs_true_state": round(float(acc_lag.sum() / n_lag.sum()), 4),
           "lag_bins": [f"{lo}-{hi-1}" if hi < 10**9 else f"{lo}+"
                        for lo, hi in LAG_BINS],
           "acc_by_lag": [round(float(a / c), 4) for a, c in zip(acc_lag, n_lag)],
           "reader_acc_by_lag": [round(float(a / c), 4)
                                 for a, c in zip(acc_lag_reader, n_lag)]}
    rec.update(ring_metrics(means))
    print(json.dumps(rec, indent=1), flush=True)
    out = ROOT / "probe" / f"{args.student}_concat.json"
    out.write_text(json.dumps(rec, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
