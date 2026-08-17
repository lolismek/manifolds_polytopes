"""Dump the uniform student's ridge-probe read-out b_hat on the RING eval
docs (cross condition) at one layer — control2's panel for the belief-PCA
blog figure. Same eval split, probe formula and layer as exp06's
dump_belief_viz.py, so the panels are directly comparable.

Writes results/probe/uniform_belief_viz_cross.npz: bhat (1000, T, 8),
z (1000, T), layer. (b_t is already in exp06's belief_viz.npz.)

Usage (seahorse): python dump_belief_viz_u.py --device cuda:0 [--layer 12]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

EXP06_SRC = Path(__file__).parents[2] / "06_short_dwell" / "src"
sys.path.insert(0, str(EXP06_SRC))
from probe_sweep import K, LAM_SCALE, MICRO, P_LEN, ROOT, load_eval  # noqa: E402

STUDENT_ROOT = Path(__file__).parents[1] / "results" / "students" / "uniform"
OUT = Path(__file__).parents[1] / "results" / "probe"
FINAL = "ckpt_04218.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layer", type=int, default=12)
    args = ap.parse_args()
    dev = torch.device(args.device)
    L = args.layer

    ids, post, z, tr, te = load_eval()          # exp06 ring eval docs
    vmap = np.load(ROOT / "corpus" / "vocab32k.npz")["map"]
    ids_t = torch.from_numpy(vmap[ids].astype(np.int64))

    cfg = LlamaConfig.from_dict(
        json.loads((STUDENT_ROOT / "config.json").read_text()))
    model = LlamaForCausalLM(cfg)
    sd = torch.load(STUDENT_ROOT / FINAL, map_location="cpu")
    model.load_state_dict({k: v.float() for k, v in sd.items()})
    model = model.to(dev).eval()
    d = cfg.hidden_size

    def batches(idx):
        for i in range(0, len(idx), MICRO):
            b = idx[i : i + MICRO]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                hs = model(ids_t[b].to(dev),
                           output_hidden_states=True).hidden_states
            yield b, hs[L][:, P_LEN:, :].float()

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

    preds = []
    for b, h in batches(te):
        X = h.reshape(h.shape[0], -1, d)
        preds.append(((X - mx) @ W + my).cpu().numpy())
    bhat = np.concatenate(preds)

    np.savez(OUT / "uniform_belief_viz_cross.npz",
             bhat=bhat.astype(np.float16), z=z[te].astype(np.int8), layer=L)
    print("wrote", OUT / "uniform_belief_viz_cross.npz", flush=True)


if __name__ == "__main__":
    main()
