"""Per-state probes on first-dwell prefixes with the whitening dial exposed
(user's request).

A per-class ridge probe equals the joint ridge column-for-column, but its
regularization strength lambda interpolates the geometry continuously:
lambda -> 0 fully whitens by the activation covariance (the anti-ring
view), lambda -> inf converges to the raw class-correlation directions,
i.e. the centroids (the circle view). We sweep lambda and watch the 8
read-out directions morph. As a genuinely different per-class estimator we
also fit one-vs-rest LOGISTIC probes (class-balanced, no global whitening).

Scored per direction-set: ring-distance corr of the unit-normed directions,
circle (f1) and neighbor-alternation (f3+f4) Fourier energy fractions with
permutation p over the 2520 ring orderings (at PERM_LAYERS), and held-out
first-dwell argmax accuracy (at PERM_LAYERS).

Usage: python perclass_probe.py --device cuda:0 --min-into 10 --min-len 15 \
           --tag _late
Writes results/probe/perclass{tag}.json + .png
"""

import argparse
import itertools
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from firstdwell_probe import FINAL, first_dwell_mask
from fourier_ringmode import spectrum
from probe_sweep import K, MICRO, P_LEN, ROOT, load_eval, ring_metrics

LAM_SCALES = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
PERM_LAYERS = [3, 9, 10, 11, 12]
LOGIT_STEPS = 400

ORDERINGS = [(0,) + p for p in itertools.permutations(range(1, K))
             if p[0] < p[-1]]


def perm_p(vecs):
    """vecs (K, d) -> f1/f34 fractions + permutation p-values."""
    X = vecs - vecs.mean(0)
    G = X @ X.T
    fr = spectrum(G)
    f1s, f34s = [], []
    for o in ORDERINGS:
        fo = spectrum(G[np.ix_(o, o)])
        f1s.append(fo[0]); f34s.append(fo[2] + fo[3])
    return {"frac_f1": round(float(fr[0]), 4),
            "frac_f34": round(float(fr[2] + fr[3]), 4),
            "p_f1": round(float((np.array(f1s) >= fr[0]).mean()), 4),
            "p_f34": round(float((np.array(f34s) >= fr[2] + fr[3]).mean()), 4)}


def geometry(vecs, with_p):
    vn = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    rec = {"dist_vs_ringdist_corr": ring_metrics(vn)["dist_vs_ringdist_corr"],
           "is_ring_order": ring_metrics(vn)["is_ring_order"]}
    if with_p:
        rec.update(perm_p(vn))
    return rec


def collect(student, dev, ids, ms, tr, te):
    """Forward the eval docs; return per-layer Gram stats (train) and raw
    first-dwell activations at PERM_LAYERS (train+test) with labels."""
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
            zt = torch.from_numpy(ms[b].reshape(-1)).to(dev)[sel]
            yield Xs, zt

    G = [torch.zeros(d, d, dtype=torch.float64, device=dev) for _ in range(L)]
    XtZ = [torch.zeros(d, K, dtype=torch.float64, device=dev) for _ in range(L)]
    sx = [torch.zeros(d, dtype=torch.float64, device=dev) for _ in range(L)]
    sz = torch.zeros(K, dtype=torch.float64, device=dev)
    n = 0
    Xtr = {l: [] for l in PERM_LAYERS}; ztr = []
    Xte = {l: [] for l in PERM_LAYERS}; zte = []
    for Xs, zt in batches(tr):
        Z = torch.nn.functional.one_hot(zt, K).float()
        sz += Z.sum(0).double(); n += zt.shape[0]; ztr.append(zt)
        for l in range(L):
            G[l] += (Xs[l].T @ Xs[l]).double()
            XtZ[l] += (Xs[l].T @ Z).double()
            sx[l] += Xs[l].sum(0).double()
        for l in PERM_LAYERS:
            Xtr[l].append(Xs[l])
    for Xs, zt in batches(te):
        zte.append(zt)
        for l in PERM_LAYERS:
            Xte[l].append(Xs[l])
    del model
    torch.cuda.empty_cache()
    return {"G": G, "XtZ": XtZ, "sx": sx, "sz": sz, "n": n, "L": L, "d": d,
            "Xtr": {l: torch.cat(v) for l, v in Xtr.items()},
            "ztr": torch.cat(ztr),
            "Xte": {l: torch.cat(v) for l, v in Xte.items()},
            "zte": torch.cat(zte)}


def ridge_dirs(st, l, lam_scale, dev):
    mx = (st["sx"][l] / st["n"])
    mz = (st["sz"] / st["n"])
    Gc = st["G"][l] - st["n"] * torch.outer(mx, mx)
    lam = lam_scale * torch.diagonal(Gc).mean()
    A = (Gc + lam * torch.eye(st["d"], dtype=torch.float64, device=dev)).cpu()
    B = (st["XtZ"][l] - st["n"] * torch.outer(mx, mz)).cpu()
    W = torch.linalg.solve(A, B).to(dev).float()
    return W, mx.float(), mz.float()


def logistic_dirs(st, l, dev):
    X, z = st["Xtr"][l], st["ztr"]
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    W = torch.zeros(st["d"], K, device=dev, requires_grad=True)
    b = torch.zeros(K, device=dev, requires_grad=True)
    Y = torch.nn.functional.one_hot(z, K).float()
    pos_w = (Y.shape[0] - Y.sum(0)) / Y.sum(0)
    opt = torch.optim.Adam([W, b], lr=0.05)
    for _ in range(LOGIT_STEPS):     # 8 independent one-vs-rest problems
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            Xn @ W + b, Y, pos_weight=pos_w)
        loss.backward()
        opt.step()
    Wd = (W.detach() / sd[:, None])          # undo standardization
    return Wd, mu, b.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--min-into", type=int, default=10)
    ap.add_argument("--min-len", type=int, default=15)
    ap.add_argument("--tag", default="_late")
    args = ap.parse_args()
    dev = torch.device(args.device)

    ids, post, z, tr, te = load_eval()
    ms = first_dwell_mask(z, args.min_into, args.min_len)
    out = {}
    dirs_out = {"lam_scales": np.array(LAM_SCALES)}
    for student in ("ring", "ctrl"):
        st = collect(student, dev, ids, ms, tr, te)
        L = st["L"]
        srec = {"ridge": [], "logistic": []}
        ridge_saved = {l: [] for l in PERM_LAYERS}
        for lam_scale in LAM_SCALES:
            for l in range(L):
                W, mx, mz = ridge_dirs(st, l, lam_scale, dev)
                rec = {"layer": l, "lam_scale": lam_scale}
                rec.update(geometry(W.T.cpu().numpy(), l in PERM_LAYERS))
                if l in PERM_LAYERS:
                    pred = (st["Xte"][l] - mx) @ W + mz
                    rec["acc"] = round(float(
                        (pred.argmax(1) == st["zte"]).float().mean()), 4)
                    ridge_saved[l].append(W.T.cpu().numpy())
                srec["ridge"].append(rec)
        for l in PERM_LAYERS:
            dirs_out[f"{student}_ridge_L{l}"] = np.stack(
                ridge_saved[l]).astype(np.float32)      # (n_lam, K, d)
            W, mu, b = logistic_dirs(st, l, dev)
            dirs_out[f"{student}_logistic_L{l}"] = \
                W.T.cpu().numpy().astype(np.float32)     # (K, d)
            rec = {"layer": l}
            rec.update(geometry(W.T.cpu().numpy(), True))
            pred = (st["Xte"][l] - mu) @ W + b
            rec["acc"] = round(float(
                (pred.argmax(1) == st["zte"]).float().mean()), 4)
            srec["logistic"].append(rec)
            print(f"{student} logistic L{l}: {json.dumps(rec)}", flush=True)
        out[student] = srec
    np.savez(ROOT / "probe" / f"perclass{args.tag}_dirs.npz", **dirs_out)
        for rec in srec["ridge"]:
            if rec["layer"] == 10:
                print(f"{student} ridge L10 lam {rec['lam_scale']}: "
                      f"corr {rec['dist_vs_ringdist_corr']} "
                      f"ring={rec['is_ring_order']} "
                      f"f1 {rec.get('frac_f1')} p_f1 {rec.get('p_f1')} "
                      f"acc {rec.get('acc')}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for student, ls in (("ring", "-"), ("ctrl", ":")):
        for l in (9, 10, 11, 12) if student == "ring" else (3,):
            rs = [r for r in out[student]["ridge"] if r["layer"] == l]
            xs = [r["lam_scale"] for r in rs]
            axes[0].plot(xs, [r["dist_vs_ringdist_corr"] for r in rs], ls,
                         marker="o", label=f"{student} L{l}")
            if l in PERM_LAYERS:
                axes[1].plot(xs, [r["frac_f1"] for r in rs], ls, marker="o",
                             label=f"{student} L{l}")
    for ax, ttl in ((axes[0], "ring-distance corr of unit-norm read-out "
                     "directions"),
                    (axes[1], "circle (f1) energy fraction (chance 0.286)")):
        ax.set_xscale("log"); ax.set_xlabel("ridge lam scale "
                                            "(small = whitened)")
        ax.axhline(0 if ax is axes[0] else 2 / 7, color="0.7", lw=0.8)
        ax.set_title(ttl, fontsize=10); ax.legend(fontsize=8)
    fig.suptitle("per-state probe directions vs whitening strength "
                 "(first-dwell prefixes)")
    fig.tight_layout()
    fig.savefig(ROOT / "probe" / f"perclass{args.tag}.png", dpi=150)
    (ROOT / "probe" / f"perclass{args.tag}.json").write_text(
        json.dumps(out, indent=1))
    print(f"wrote perclass{args.tag}.json / .png", flush=True)


if __name__ == "__main__":
    main()
