"""Contamination-free probe + geometry test on FIRST dwells (exp05's
firstdwell design, all 13 layers, exp06 data): every token before a
document's first state switch. There the inputs carry no previous-dwell
residue, so the neighbor-carryover artifact cannot operate.

Per student and per layer, using only first-dwell tokens (train docs fit,
held-out docs eval):
  - ridge probe -> exact posterior: test R2, argmax accuracy
  - ridge probe -> ONE-HOT true state: argmax accuracy
  - ring metrics on the 8 first-dwell class means (raw activation view)
  - ring metrics on the probe's 8 read-out columns, unit-normalized, for
    BOTH targets. Caveat inherited from exp05: the posterior target keeps
    "maybe we just switched" mass on ring neighbors even inside a first
    dwell, so posterior-probe columns inherit ring structure from the
    answer key for ANY model; the one-hot columns are the clean test.

Figure: 2D PCA projections (class means / one-hot columns) at ring L10,
ring L12, ctrl L3. Output: results/probe/firstdwell.json + firstdwell.png

Usage: python firstdwell_probe.py --device cuda:0
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from probe_sweep import (EXP03_ROOT, K, LAM_SCALE, MICRO, P_LEN, ROOT,
                         load_eval, ring_metrics)

FINAL = {"ring": (ROOT / "students" / "ring", ROOT, "ckpt_04218.pt"),
         "ctrl": (EXP03_ROOT / "students" / "ctrl", EXP03_ROOT,
                  "ckpt_02812.pt")}
FIG_PANELS = [("ring", 10), ("ring", 12), ("ctrl", 3)]


def first_dwell_mask(z, min_into=0, min_len=0):
    """(n, T) int: state z[i,0] on tokens before the first switch, else -1.
    min_into/min_len restrict to LATE first dwells: only tokens >= min_into
    positions in, and only if the first dwell lasts >= min_len tokens."""
    n, T = z.shape
    ms = np.full((n, T), -1, dtype=np.int64)
    for i in range(n):
        sw = np.nonzero(z[i] != z[i, 0])[0]
        end = sw[0] if len(sw) else T
        if end >= min_len:
            ms[i, min_into:end] = z[i, 0]
    return ms


def analyze(student, dev, ids, post, ms, tr, te):
    sroot, vroot, ckpt = FINAL[student]
    vmap = np.load(vroot / "corpus" / "vocab32k.npz")["map"]
    cfg = LlamaConfig.from_dict(json.loads((sroot / "config.json").read_text()))
    model = LlamaForCausalLM(cfg)
    sd = torch.load(sroot / ckpt, map_location="cpu")
    model.load_state_dict({k: v.float() for k, v in sd.items()})
    model = model.to(dev).eval()
    ids_t = torch.from_numpy(vmap[ids].astype(np.int64))
    d = cfg.hidden_size
    L = cfg.num_hidden_layers + 1

    def batches(idx):
        for i in range(0, len(idx), MICRO):
            b = idx[i : i + MICRO]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                hs = model(ids_t[b].to(dev),
                           output_hidden_states=True).hidden_states
            sel = torch.from_numpy(ms[b].reshape(-1) >= 0).to(dev)
            Xs = [h[:, P_LEN:, :].float().reshape(-1, d)[sel] for h in hs]
            Y = torch.from_numpy(post[b].reshape(-1, K)).to(dev)[sel]
            zt = torch.from_numpy(ms[b].reshape(-1)).to(dev)[sel]
            yield Xs, Y, zt

    G = [torch.zeros(d, d, dtype=torch.float64, device=dev) for _ in range(L)]
    XtY = [torch.zeros(d, K, dtype=torch.float64, device=dev) for _ in range(L)]
    XtZ = [torch.zeros(d, K, dtype=torch.float64, device=dev) for _ in range(L)]
    sx = [torch.zeros(d, dtype=torch.float64, device=dev) for _ in range(L)]
    csum = [torch.zeros(K, d, dtype=torch.float64, device=dev) for _ in range(L)]
    ccount = torch.zeros(K, dtype=torch.float64, device=dev)
    sy = torch.zeros(K, dtype=torch.float64, device=dev)
    sz = torch.zeros(K, dtype=torch.float64, device=dev)
    n = 0
    for Xs, Y, zt in batches(tr):
        Z = torch.nn.functional.one_hot(zt, K).float()
        sy += Y.sum(0).double(); sz += Z.sum(0).double(); n += Y.shape[0]
        ccount += Z.sum(0).double()
        for l in range(L):
            X = Xs[l]
            G[l] += (X.T @ X).double()
            XtY[l] += (X.T @ Y).double()
            XtZ[l] += (X.T @ Z).double()
            sx[l] += X.sum(0).double()
            csum[l] += (Z.T @ X).double()
    my, mz = (sy / n).float(), (sz / n).float()

    solved = []
    for l in range(L):
        mxl = sx[l] / n
        Gc = G[l] - n * torch.outer(mxl, mxl)
        lam = LAM_SCALE * torch.diagonal(Gc).mean()
        A = (Gc + lam * torch.eye(d, dtype=torch.float64, device=dev)).cpu()
        W = torch.linalg.solve(
            A, (XtY[l] - n * torch.outer(mxl, sy / n)).cpu()).to(dev).float()
        Wz = torch.linalg.solve(
            A, (XtZ[l] - n * torch.outer(mxl, sz / n)).cpu()).to(dev).float()
        solved.append((W, Wz, mxl.float()))

    ss_res = np.zeros(L); hits = np.zeros(L); hits_z = np.zeros(L)
    y_sum = torch.zeros(K, dtype=torch.float64, device=dev)
    y_sq = 0.0; n_te = 0
    for Xs, Y, zt in batches(te):
        y_sum += Y.sum(0).double(); y_sq += float((Y ** 2).sum())
        n_te += Y.shape[0]
        for l in range(L):
            W, Wz, mxl = solved[l]
            pred = (Xs[l] - mxl) @ W + my
            ss_res[l] += float(((pred - Y) ** 2).sum())
            hits[l] += int((pred.argmax(1) == zt).sum())
            hits_z[l] += int((((Xs[l] - mxl) @ Wz + mz).argmax(1) == zt).sum())
    y_mean = y_sum / n_te
    ss_tot = float(y_sq - n_te * float((y_mean ** 2).sum()))
    del model
    torch.cuda.empty_cache()

    recs, fig_data = [], {}
    for l in range(L):
        W, Wz, _ = solved[l]
        means = (csum[l] / ccount[:, None]).cpu().numpy()
        cols_n = W.T.cpu().numpy()
        cols_n /= np.linalg.norm(cols_n, axis=1, keepdims=True)
        zcols_n = Wz.T.cpu().numpy()
        zcols_n /= np.linalg.norm(zcols_n, axis=1, keepdims=True)
        recs.append({
            "layer": l,
            "probe_R2": round(1 - ss_res[l] / ss_tot, 4),
            "acc_vs_true_state": round(hits[l] / n_te, 4),
            "onehot_acc_vs_true_state": round(hits_z[l] / n_te, 4),
            "class_means": ring_metrics(means),
            "posterior_cols_unitnorm": ring_metrics(cols_n),
            "onehot_cols_unitnorm": ring_metrics(zcols_n)})
        fig_data[l] = (means, zcols_n)
    arrays = {
        "means": np.stack([(csum[l] / ccount[:, None]).cpu().numpy()
                           for l in range(L)]).astype(np.float32),
        "post_cols": np.stack([solved[l][0].T.cpu().numpy()
                               for l in range(L)]).astype(np.float32),
        "onehot_cols": np.stack([solved[l][1].T.cpu().numpy()
                                 for l in range(L)]).astype(np.float32)}
    return {"ckpt": ckpt, "n_train_tokens": int(n), "n_test_tokens": int(n_te),
            "layers": recs}, fig_data, arrays


def panel(ax, pts, title):
    mc = pts - pts.mean(0)
    U, S, Vt = np.linalg.svd(mc, full_matrices=False)
    mm = mc @ Vt[:2].T
    cmap = plt.get_cmap("tab10")
    poly = np.array([mm[k] for k in list(range(K)) + [0]])
    ax.plot(poly[:, 0], poly[:, 1], "-", color="0.3", lw=1.5, zorder=4)
    for k in range(K):
        ax.scatter(*mm[k], s=220, color=cmap(k), edgecolor="black", zorder=5)
        ax.annotate(str(k), mm[k], ha="center", va="center", zorder=6,
                    fontsize=9, fontweight="bold", color="white")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--min-into", type=int, default=0,
                    help="use only tokens >= this many positions into the "
                         "first dwell (late-first-dwell variant)")
    ap.add_argument("--min-len", type=int, default=0,
                    help="require the first dwell to last >= this many tokens")
    ap.add_argument("--tag", default="",
                    help="output name suffix, e.g. _late")
    args = ap.parse_args()
    dev = torch.device(args.device)

    ids, post, z, tr, te = load_eval()
    ms = first_dwell_mask(z, args.min_into, args.min_len)
    sel = ms >= 0
    reader_fd_acc = float((post.argmax(-1) == z)[sel].mean())
    print(f"first-dwell tokens: {int(sel.sum())} "
          f"(mean {sel.sum()/len(z):.1f}/doc); "
          f"exact reader acc there: {reader_fd_acc:.4f}", flush=True)

    out = {"reader_firstdwell_acc": round(reader_fd_acc, 4)}
    all_fig, npz_out = {}, {}
    for student in ("ring", "ctrl"):
        rec, fig_data, arrays = analyze(student, dev, ids, post, ms, tr, te)
        out[student] = rec
        all_fig[student] = fig_data
        for k, v in arrays.items():
            npz_out[f"{student}_{k}"] = v
        best = max(rec["layers"], key=lambda r: r["probe_R2"])
        print(f"{student}: best L{best['layer']} R2 {best['probe_R2']} "
              f"acc {best['acc_vs_true_state']} "
              f"(one-hot acc {best['onehot_acc_vs_true_state']})", flush=True)
        for r in rec["layers"]:
            cm, oc = r["class_means"], r["onehot_cols_unitnorm"]
            print(f"  L{r['layer']:>2} R2 {r['probe_R2']:>7} "
                  f"means: ring={cm['is_ring_order']} "
                  f"corr {cm['dist_vs_ringdist_corr']:>7} | "
                  f"onehot cols: ring={oc['is_ring_order']} "
                  f"corr {oc['dist_vs_ringdist_corr']:>7}", flush=True)

    fig, axes = plt.subplots(2, len(FIG_PANELS), figsize=(6 * len(FIG_PANELS), 11))
    for c, (student, l) in enumerate(FIG_PANELS):
        means, zcols_n = all_fig[student][l]
        r = next(x for x in out[student]["layers"] if x["layer"] == l)
        cm, oc = r["class_means"], r["onehot_cols_unitnorm"]
        panel(axes[0, c], means,
              f"{student} L{l} — FIRST-DWELL class means\n"
              f"ring order: {cm['is_ring_order']}, "
              f"corr {cm['dist_vs_ringdist_corr']}, "
              f"pc12 var {cm['pc12_var_frac']}")
        panel(axes[1, c], zcols_n,
              f"{student} L{l} — ONE-HOT probe columns (unit norm)\n"
              f"ring order: {oc['is_ring_order']}, "
              f"corr {oc['dist_vs_ringdist_corr']}, "
              f"acc {r['onehot_acc_vs_true_state']}")
    fig.suptitle("first-dwell-only probes: state geometry without "
                 "neighbor-carryover (top: raw activation means, "
                 "bottom: probe read-out directions)")
    fig.tight_layout()
    fig.savefig(ROOT / "probe" / f"firstdwell{args.tag}.png", dpi=150)
    (ROOT / "probe" / f"firstdwell{args.tag}.json").write_text(
        json.dumps(out, indent=1))
    np.savez(ROOT / "probe" / f"firstdwell{args.tag}_arrays.npz", **npz_out)
    print(f"wrote firstdwell{args.tag}.json / .png / _arrays.npz", flush=True)


if __name__ == "__main__":
    main()
