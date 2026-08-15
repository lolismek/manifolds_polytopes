"""Dump the optimal observer's posteriors b_t and the ridge probe's
predictions b_hat at one layer, for the eval test docs (blog figure:
belief-state PCA). Same eval split, probe formula, and final ring checkpoint
as probe_sweep.py.

Writes results/probe/belief_viz.npz: b (1000, T, 8), bhat (1000, T, 8),
z (1000, T), all test docs, plus layer.

Usage: python dump_belief_viz.py --device cuda:0 [--layer 12]
"""
import argparse
import json

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from probe_sweep import K, LAM_SCALE, MICRO, P_LEN, ROOT, load_eval

FINAL = "ckpt_04218.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layer", type=int, default=12)
    args = ap.parse_args()
    dev = torch.device(args.device)
    L = args.layer

    ids, post, z, tr, te = load_eval()
    vmap = np.load(ROOT / "corpus" / "vocab32k.npz")["map"]
    ids_t = torch.from_numpy(vmap[ids].astype(np.int64))

    sroot = ROOT / "students" / "ring"
    cfg = LlamaConfig.from_dict(json.loads((sroot / "config.json").read_text()))
    model = LlamaForCausalLM(cfg)
    sd = torch.load(sroot / FINAL, map_location="cpu")
    model.load_state_dict({k: v.float() for k, v in sd.items()})
    model = model.to(dev).eval()
    d = cfg.hidden_size

    def batches(idx):
        for i in range(0, len(idx), MICRO):
            b = idx[i : i + MICRO]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                hs = model(ids_t[b].to(dev), output_hidden_states=True).hidden_states
            yield b, hs[L][:, P_LEN:, :].float()

    # fit the ridge on training docs (single layer, same formula as LayerStats)
    G = torch.zeros(d, d, dtype=torch.float64, device=dev)
    XtY = torch.zeros(d, K, dtype=torch.float64, device=dev)
    sx = torch.zeros(d, dtype=torch.float64, device=dev)
    sy = torch.zeros(K, dtype=torch.float64, device=dev)
    n = 0
    for b, h in batches(tr):
        X = h.reshape(-1, d)
        Y = torch.from_numpy(post[b].reshape(-1, K)).to(dev)
        G += (X.T @ X).double()
        XtY += (X.T @ Y).double()
        sx += X.sum(0).double()
        sy += Y.sum(0).double()
        n += X.shape[0]
    mx, my = sx / n, sy / n
    Gc = G - n * torch.outer(mx, mx)
    lam = LAM_SCALE * torch.diagonal(Gc).mean()
    A = (Gc + lam * torch.eye(d, dtype=torch.float64, device=dev)).cpu()
    W = torch.linalg.solve(A, (XtY - n * torch.outer(mx, my)).cpu()).to(dev)
    mx, my, W = mx.float(), my.float(), W.float()
    print(f"probe fit on {n} train tokens at layer {L}", flush=True)

    # predict on test docs
    preds = []
    for b, h in batches(te):
        X = h.reshape(h.shape[0], -1, d)
        preds.append(((X - mx) @ W + my).cpu().numpy())
    bhat = np.concatenate(preds)

    np.savez(ROOT / "probe" / "belief_viz.npz",
             b=post[te].astype(np.float16), bhat=bhat.astype(np.float16),
             z=z[te].astype(np.int8), layer=L)
    print("wrote", ROOT / "probe" / "belief_viz.npz", flush=True)


if __name__ == "__main__":
    main()
