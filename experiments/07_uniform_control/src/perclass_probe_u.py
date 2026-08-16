"""Per-state probes with the whitening dial for the exp07 uniform student
(exp06's perclass design, machinery imported): sweep ridge lambda from
fully-whitened to centroid-like on late-first-dwell tokens, plus
one-vs-rest logistic probes. exp06's ring student swings -0.81 -> +0.69
across the dial with exact ring order at lam=10; prediction here: no
significant ring at ANY lambda (like exp03's ctrl).

Usage: python perclass_probe_u.py --device cuda:0 --min-into 10 \
           --min-len 15 --tag _late
Writes results/probe/uniform_perclass{tag}.json + .png + _dirs.npz
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

EXP06_SRC = Path(__file__).parents[2] / "06_short_dwell" / "src"
sys.path.insert(0, str(EXP06_SRC))
import firstdwell_probe as fd  # noqa: E402
import perclass_probe as pc  # noqa: E402

from probe_sweep_u import ROOT, STUDENT_ROOT, load_eval  # noqa: E402

fd.FINAL["uniform"] = (STUDENT_ROOT, fd.ROOT, "ckpt_04218.pt")
K = pc.K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--min-into", type=int, default=10)
    ap.add_argument("--min-len", type=int, default=15)
    ap.add_argument("--tag", default="_late")
    args = ap.parse_args()
    dev = torch.device(args.device)

    ids, post, z, tr, te = load_eval(ROOT)
    ms = fd.first_dwell_mask(z, args.min_into, args.min_len)
    out = {}
    dirs_out = {"lam_scales": np.array(pc.LAM_SCALES)}
    st = pc.collect("uniform", dev, ids, ms, tr, te)
    L = st["L"]
    srec = {"ridge": [], "logistic": []}
    ridge_saved = {l: [] for l in pc.PERM_LAYERS}
    for lam_scale in pc.LAM_SCALES:
        for l in range(L):
            W, mx, mz = pc.ridge_dirs(st, l, lam_scale, dev)
            rec = {"layer": l, "lam_scale": lam_scale}
            rec.update(pc.geometry(W.T.cpu().numpy(), l in pc.PERM_LAYERS))
            if l in pc.PERM_LAYERS:
                pred = (st["Xte"][l] - mx) @ W + mz
                rec["acc"] = round(float(
                    (pred.argmax(1) == st["zte"]).float().mean()), 4)
                ridge_saved[l].append(W.T.cpu().numpy())
            srec["ridge"].append(rec)
    for l in pc.PERM_LAYERS:
        dirs_out[f"uniform_ridge_L{l}"] = np.stack(
            ridge_saved[l]).astype(np.float32)      # (n_lam, K, d)
        W, mu, b = pc.logistic_dirs(st, l, dev)
        dirs_out[f"uniform_logistic_L{l}"] = \
            W.T.cpu().numpy().astype(np.float32)     # (K, d)
        rec = {"layer": l}
        rec.update(pc.geometry(W.T.cpu().numpy(), True))
        pred = (st["Xte"][l] - mu) @ W + b
        rec["acc"] = round(float(
            (pred.argmax(1) == st["zte"]).float().mean()), 4)
        srec["logistic"].append(rec)
        print(f"uniform logistic L{l}: {json.dumps(rec)}", flush=True)
    out["uniform"] = srec
    for rec in srec["ridge"]:
        if rec["layer"] == 10:
            print(f"uniform ridge L10 lam {rec['lam_scale']}: "
                  f"corr {rec['dist_vs_ringdist_corr']} "
                  f"ring={rec['is_ring_order']} "
                  f"f1 {rec.get('frac_f1')} p_f1 {rec.get('p_f1')} "
                  f"acc {rec.get('acc')}", flush=True)
    np.savez(ROOT / "probe" / f"uniform_perclass{args.tag}_dirs.npz",
             **dirs_out)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for l in (9, 10, 11, 12):
        rs = [r for r in srec["ridge"] if r["layer"] == l]
        xs = [r["lam_scale"] for r in rs]
        axes[0].plot(xs, [r["dist_vs_ringdist_corr"] for r in rs], "-",
                     marker="o", label=f"uniform L{l}")
        axes[1].plot(xs, [r["frac_f1"] for r in rs], "-", marker="o",
                     label=f"uniform L{l}")
    for ax, ttl in ((axes[0], "ring-distance corr of unit-norm read-out "
                     "directions"),
                    (axes[1], "circle (f1) energy fraction (chance 0.286)")):
        ax.set_xscale("log"); ax.set_xlabel("ridge lam scale "
                                            "(small = whitened)")
        ax.axhline(0 if ax is axes[0] else 2 / 7, color="0.7", lw=0.8)
        ax.set_title(ttl, fontsize=10); ax.legend(fontsize=8)
    fig.suptitle("exp07 uniform student: per-state probe directions vs "
                 "whitening strength (first-dwell prefixes, cast ring order)")
    fig.tight_layout()
    fig.savefig(ROOT / "probe" / f"uniform_perclass{args.tag}.png", dpi=150)
    (ROOT / "probe" / f"uniform_perclass{args.tag}.json").write_text(
        json.dumps(out, indent=1))
    print(f"wrote uniform_perclass{args.tag}.json / .png", flush=True)


if __name__ == "__main__":
    main()
