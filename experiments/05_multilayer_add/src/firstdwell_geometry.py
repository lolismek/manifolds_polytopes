"""Contamination-free geometry test (user's design): compute the 8 class
means using ONLY first-dwell tokens — every token before the document's
first state switch. No previous dwell exists in the context there, so the
neighbor-residue artifact cannot operate. Ring metrics on those means for
both students. Output: results/probe/firstdwell.json + firstdwell.png

Usage: python firstdwell_geometry.py --device cuda:2
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from pca_shapes import STUDENTS
from probe_sweep import K, MICRO, P_LEN, ROOT, load_eval, ring_metrics


def first_dwell_mask(z):
    """(n, T) int: state z[i,0] on tokens before the first switch, else -1."""
    n, T = z.shape
    ms = np.full((n, T), -1, dtype=np.int64)
    for i in range(n):
        sw = np.nonzero(z[i] != z[i, 0])[0]
        end = sw[0] if len(sw) else T
        ms[i, :end] = z[i, 0]
    return ms


def class_means(student, dev, ids, ms):
    sroot, vroot, ckpt, layer = STUDENTS[student]
    vmap = np.load(vroot / "corpus" / "vocab32k.npz")["map"]
    cfg = LlamaConfig.from_dict(json.loads((sroot / "config.json").read_text()))
    model = LlamaForCausalLM(cfg)
    sd = torch.load(sroot / ckpt, map_location="cpu")
    model.load_state_dict({k: v.float() for k, v in sd.items()})
    model = model.to(dev).eval()
    ids_t = torch.from_numpy(vmap[ids].astype(np.int64))

    csum = np.zeros((K, cfg.hidden_size)); ccnt = np.zeros(K)
    for i in range(0, ids_t.shape[0], MICRO):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hs = model(ids_t[i:i + MICRO].to(dev),
                       output_hidden_states=True).hidden_states[layer]
        X = hs[:, P_LEN:, :].float().reshape(-1, cfg.hidden_size).cpu().numpy()
        m = ms[i:i + MICRO].reshape(-1)
        for k in range(K):
            sel = m == k
            if sel.any():
                csum[k] += X[sel].sum(0); ccnt[k] += sel.sum()
    del model
    torch.cuda.empty_cache()
    return csum / ccnt[:, None], ccnt, layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:2")
    args = ap.parse_args()
    dev = torch.device(args.device)

    ids, post, z, tr, te = load_eval()
    ms = first_dwell_mask(z)
    print(f"first-dwell tokens per state: "
          f"{[int((ms == k).sum()) for k in range(K)]}", flush=True)

    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    cmap = plt.get_cmap("tab10")
    for ax, student in zip(axes, ("ring", "ctrl")):
        means, ccnt, layer = class_means(student, dev, ids, ms)
        rec = ring_metrics(means)
        rec["layer"] = layer
        rec["tokens_per_state"] = [int(c) for c in ccnt]
        out[student] = rec
        print(f"{student}: {json.dumps(rec)}", flush=True)
        mc = means - means.mean(0)
        U, S, Vt = np.linalg.svd(mc, full_matrices=False)
        mm = mc @ Vt[:2].T
        poly = np.array([mm[k] for k in list(range(K)) + [0]])
        ax.plot(poly[:, 0], poly[:, 1], "-", color="0.3", lw=1.5, zorder=4)
        for k in range(K):
            ax.scatter(*mm[k], s=220, color=cmap(k), edgecolor="black",
                       zorder=5)
            ax.annotate(str(k), mm[k], ha="center", va="center", zorder=6,
                        fontsize=9, fontweight="bold", color="white")
        ax.set_title(f"{student} student — layer {layer} (first-dwell means)\n"
                     f"ring corr {rec['dist_vs_ringdist_corr']}, "
                     f"ring order: {rec['is_ring_order']}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.suptitle("class means from FIRST dwells only (no previous state in "
                 "context -> no neighbor contamination)")
    fig.tight_layout()
    fig.savefig(ROOT / "probe" / "firstdwell.png", dpi=150)
    (ROOT / "probe" / "firstdwell.json").write_text(json.dumps(out, indent=1))
    print("wrote firstdwell.json / .png", flush=True)


if __name__ == "__main__":
    main()
