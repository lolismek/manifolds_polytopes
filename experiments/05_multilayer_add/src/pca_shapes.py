"""2D PCA picture of the 8 states in each student's residual stream.

For each student (best linear-probe layer: ring L12, ctrl L3), collect
mid-dwell hidden states from the eval docs, compute the 8 class means, PCA
the means, and plot: token cloud (colored by true state) projected onto the
class-mean PC1-2 plane, means as numbered markers, ring-neighbor edges drawn
as the polygon 0-1-...-7-0. A clean ring = convex non-crossing polygon in
ring order. Output: results/probe/pca_shapes.png

Usage: python pca_shapes.py --device cuda:2
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from probe_sweep import (EXP03_ROOT, K, MICRO, P_LEN, ROOT, load_eval,
                         middwell_states)

STUDENTS = {"ring": (ROOT / "students" / "ring", ROOT, "ckpt_04218.pt", 12),
            "ctrl": (EXP03_ROOT / "students" / "ctrl", EXP03_ROOT,
                     "ckpt_02812.pt", 3)}
PER_STATE = 1500          # scatter subsample per state


def collect(student, dev, ids, ms):
    sroot, vroot, ckpt, layer = STUDENTS[student]
    vmap = np.load(vroot / "corpus" / "vocab32k.npz")["map"]
    cfg = LlamaConfig.from_dict(json.loads((sroot / "config.json").read_text()))
    model = LlamaForCausalLM(cfg)
    sd = torch.load(sroot / ckpt, map_location="cpu")
    model.load_state_dict({k: v.float() for k, v in sd.items()})
    model = model.to(dev).eval()
    ids_t = torch.from_numpy(vmap[ids].astype(np.int64))

    feats = [[] for _ in range(K)]
    csum = np.zeros((K, cfg.hidden_size)); ccnt = np.zeros(K)
    rng = np.random.default_rng(0)
    for i in range(0, ids_t.shape[0], MICRO):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hs = model(ids_t[i:i + MICRO].to(dev),
                       output_hidden_states=True).hidden_states[layer]
        X = hs[:, P_LEN:, :].float().reshape(-1, cfg.hidden_size).cpu().numpy()
        m = ms[i:i + MICRO].reshape(-1)
        for k in range(K):
            sel = np.nonzero(m == k)[0]
            if len(sel):
                csum[k] += X[sel].sum(0); ccnt[k] += len(sel)
                take = sel[rng.random(len(sel)) < 0.1]
                if len(take):
                    feats[k].append(X[take])
    del model
    torch.cuda.empty_cache()
    means = csum / ccnt[:, None]
    clouds = [np.concatenate(f)[:PER_STATE] for f in feats]
    return means, clouds, layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:2")
    args = ap.parse_args()
    dev = torch.device(args.device)

    ids, post, z, tr, te = load_eval()
    ms = middwell_states(z)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    cmap = plt.get_cmap("tab10")
    for ax, student in zip(axes, ("ring", "ctrl")):
        means, clouds, layer = collect(student, dev, ids, ms)
        mc = means - means.mean(0)
        U, S, Vt = np.linalg.svd(mc, full_matrices=False)
        P = Vt[:2].T                       # (d, 2) class-mean PC plane
        mm = mc @ P
        for k in range(K):
            pts = (clouds[k] - means.mean(0)) @ P
            ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.15,
                       color=cmap(k), linewidths=0)
        poly = np.array([mm[k] for k in list(range(K)) + [0]])
        ax.plot(poly[:, 0], poly[:, 1], "-", color="0.3", lw=1.5, zorder=4)
        for k in range(K):
            ax.scatter(*mm[k], s=220, color=cmap(k), edgecolor="black",
                       zorder=5)
            ax.annotate(str(k), mm[k], ha="center", va="center", zorder=6,
                        fontsize=9, fontweight="bold", color="white")
        var2 = (S[:2] ** 2).sum() / (S ** 2).sum()
        ax.set_title(f"{student} student — layer {layer}\n"
                     f"class-mean PC1-2 ({var2:.0%} of mean variance); "
                     f"edges = ring neighbors")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        print(f"{student}: done", flush=True)
    fig.suptitle("mid-dwell hidden states projected on class-mean PC plane "
                 "(clean ring = non-crossing 0-1-...-7 polygon)")
    fig.tight_layout()
    out = ROOT / "probe" / "pca_shapes.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
