"""Contamination-free PROBE test (user's design, level 2): train a fresh
ridge probe using ONLY first-dwell tokens (every token before a document's
first state switch). There the inputs carry no previous-dwell residue and
the exact reader's target beliefs never slide along the ring (they start
uniform and sharpen toward one state), so neither the carryover artifact
nor target-inherited ring structure can shape the probe.

Reported per student (ring L12, ctrl L3):
  - test R2 / argmax accuracy on held-out docs' first-dwell tokens
    (probe trained and evaluated on clean prefixes -> the fair numbers)
  - ring metrics on the probe's 8 learned read-out columns, raw and
    unit-normalized (the covariance-rescaled view of state geometry that
    plain class means could miss if a ring hides in low-variance directions)

Caveat found on the first run: the exact-reader posterior keeps a little
"maybe we just switched" mass on the ring NEIGHBORS of the true state even
inside a first dwell (the HMM prior forces it), so a probe trained on the
posterior inherits ring structure from the answer key for ANY model — the
unsteered control came out with a perfect ring order (corr 0.78). Hence the
second probe below, trained on ONE-HOT true-state labels, which carry zero
neighbor structure on first dwells; its columns are the artifact-free test.

Output: results/probe/firstdwell_probe.json + firstdwell_probe.png

Usage: python firstdwell_probe.py --device cuda:2
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from firstdwell_geometry import first_dwell_mask
from pca_shapes import STUDENTS
from probe_sweep import (K, LAM_SCALE, MICRO, P_LEN, ROOT, load_eval,
                         ring_metrics)


def fit_and_test(student, dev, ids, post, ms, tr, te):
    sroot, vroot, ckpt, layer = STUDENTS[student]
    vmap = np.load(vroot / "corpus" / "vocab32k.npz")["map"]
    cfg = LlamaConfig.from_dict(json.loads((sroot / "config.json").read_text()))
    model = LlamaForCausalLM(cfg)
    sd = torch.load(sroot / ckpt, map_location="cpu")
    model.load_state_dict({k: v.float() for k, v in sd.items()})
    model = model.to(dev).eval()
    ids_t = torch.from_numpy(vmap[ids].astype(np.int64))
    d = cfg.hidden_size

    def batches(idx):
        for i in range(0, len(idx), MICRO):
            b = idx[i : i + MICRO]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                hs = model(ids_t[b].to(dev),
                           output_hidden_states=True).hidden_states[layer]
            X = hs[:, P_LEN:, :].float().reshape(-1, d)
            sel = torch.from_numpy(ms[b].reshape(-1) >= 0).to(dev)
            Y = torch.from_numpy(post[b].reshape(-1, K)).to(dev)
            zt = torch.from_numpy(ms[b].reshape(-1)).to(dev)
            yield X[sel], Y[sel], zt[sel]

    # train pass: first-dwell tokens of train docs only
    G = torch.zeros(d, d, dtype=torch.float64, device=dev)
    XtY = torch.zeros(d, K, dtype=torch.float64, device=dev)
    XtZ = torch.zeros(d, K, dtype=torch.float64, device=dev)   # one-hot targets
    sx = torch.zeros(d, dtype=torch.float64, device=dev)
    sy = torch.zeros(K, dtype=torch.float64, device=dev)
    sz = torch.zeros(K, dtype=torch.float64, device=dev)
    n = 0
    for X, Y, zt in batches(tr):
        Z = torch.nn.functional.one_hot(zt, K).float()
        G += (X.T @ X).double()
        XtY += (X.T @ Y).double()
        XtZ += (X.T @ Z).double()
        sx += X.sum(0).double(); sy += Y.sum(0).double(); sz += Z.sum(0).double()
        n += X.shape[0]
    mx, my = (sx / n).float(), (sy / n).float()
    mz = (sz / n).float()
    Gc = G - n * torch.outer(sx / n, sx / n)
    lam = LAM_SCALE * torch.diagonal(Gc).mean()
    A = Gc + lam * torch.eye(d, dtype=torch.float64, device=dev)
    W = torch.linalg.solve(
        A, XtY - n * torch.outer(sx / n, sy / n)).float()
    Wz = torch.linalg.solve(
        A, XtZ - n * torch.outer(sx / n, sz / n)).float()

    # test pass: first-dwell tokens of held-out docs
    ss_res = 0.0; hits = 0; hits_z = 0; n_te = 0
    y_sum = torch.zeros(K, dtype=torch.float64, device=dev)
    y_sq = torch.tensor(0.0, dtype=torch.float64, device=dev)
    for X, Y, zt in batches(te):
        pred = (X - mx) @ W + my
        ss_res += float(((pred - Y) ** 2).sum())
        hits += int((pred.argmax(1) == zt).sum())
        hits_z += int((((X - mx) @ Wz + mz).argmax(1) == zt).sum())
        y_sum += Y.sum(0).double(); y_sq += (Y ** 2).sum().double()
        n_te += X.shape[0]
    y_mean = y_sum / n_te
    ss_tot = float(y_sq - n_te * (y_mean ** 2).sum())
    del model
    torch.cuda.empty_cache()

    cols = W.T.cpu().numpy()                       # (K, d) read-out directions
    cols_n = cols / np.linalg.norm(cols, axis=1, keepdims=True)
    zcols = Wz.T.cpu().numpy()
    zcols_n = zcols / np.linalg.norm(zcols, axis=1, keepdims=True)
    rec = {"layer": layer, "n_train_tokens": n, "n_test_tokens": n_te,
           "probe_R2": round(1 - ss_res / ss_tot, 4),
           "acc_vs_true_state": round(hits / n_te, 4),
           "onehot_acc_vs_true_state": round(hits_z / n_te, 4),
           "weight_cols": ring_metrics(cols),
           "weight_cols_unitnorm": ring_metrics(cols_n),
           "onehot_cols": ring_metrics(zcols),
           "onehot_cols_unitnorm": ring_metrics(zcols_n)}
    return rec, cols_n, zcols_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:2")
    args = ap.parse_args()
    dev = torch.device(args.device)

    ids, post, z, tr, te = load_eval()
    ms = first_dwell_mask(z)

    sel = ms >= 0
    reader_fd_acc = float((post.argmax(-1) == z)[sel].mean())
    print(f"exact reader acc on first-dwell tokens: {reader_fd_acc:.4f}",
          flush=True)

    out = {"reader_firstdwell_acc": round(reader_fd_acc, 4)}
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    cmap = plt.get_cmap("tab10")

    def panel(ax, cols_n, title):
        mc = cols_n - cols_n.mean(0)
        U, S, Vt = np.linalg.svd(mc, full_matrices=False)
        mm = mc @ Vt[:2].T
        poly = np.array([mm[k] for k in list(range(K)) + [0]])
        ax.plot(poly[:, 0], poly[:, 1], "-", color="0.3", lw=1.5, zorder=4)
        for k in range(K):
            ax.scatter(*mm[k], s=220, color=cmap(k), edgecolor="black",
                       zorder=5)
            ax.annotate(str(k), mm[k], ha="center", va="center", zorder=6,
                        fontsize=9, fontweight="bold", color="white")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    for col, student in enumerate(("ring", "ctrl")):
        rec, cols_n, zcols_n = fit_and_test(student, dev, ids, post, ms, tr, te)
        out[student] = rec
        print(f"{student}: {json.dumps(rec)}", flush=True)
        wr = rec["weight_cols_unitnorm"]
        zr = rec["onehot_cols_unitnorm"]
        panel(axes[0, col], cols_n,
              f"{student} L{rec['layer']} — POSTERIOR-target probe cols "
              f"(target-contaminated)\nR2 {rec['probe_R2']}, "
              f"acc {rec['acc_vs_true_state']}, "
              f"ring corr {wr['dist_vs_ringdist_corr']}, "
              f"ring order: {wr['is_ring_order']}")
        panel(axes[1, col], zcols_n,
              f"{student} L{rec['layer']} — ONE-HOT-target probe cols "
              f"(clean)\nacc {rec['onehot_acc_vs_true_state']}, "
              f"ring corr {zr['dist_vs_ringdist_corr']}, "
              f"ring order: {zr['is_ring_order']}")
    fig.suptitle("ridge probes trained on FIRST dwells only — geometry of the "
                 "8 learned read-out directions (unit norm)")
    fig.tight_layout()
    fig.savefig(ROOT / "probe" / "firstdwell_probe.png", dpi=150)
    (ROOT / "probe" / "firstdwell_probe.json").write_text(
        json.dumps(out, indent=1))
    print("wrote firstdwell_probe.json / .png", flush=True)


if __name__ == "__main__":
    main()
