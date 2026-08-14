"""Dump the 8 mid-dwell class means per layer for the final checkpoints of
both students (blog figure: beat-1 PCA). Same eval docs, same mid-dwell
definition as probe_sweep.py; no probes, just per-state activation means.

Writes results/probe/middwell_means.npz: {ring,ctrl}_means (13, 8, 768).

Usage: python dump_middwell_means.py --device cuda:0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from probe_sweep import (K, MICRO, P_LEN, ROOT, EXP03_ROOT, load_eval,
                         middwell_states)

FINAL = {"ring": "ckpt_04218.pt", "ctrl": "ckpt_02812.pt"}


def class_means(model, dev, ids_t, ms):
    n_layers = model.config.num_hidden_layers + 1
    d = model.config.hidden_size
    csum = torch.zeros(n_layers, K, d, dtype=torch.float64, device=dev)
    ccount = torch.zeros(K, dtype=torch.float64, device=dev)
    idx = np.arange(ids_t.shape[0])
    for i in range(0, len(idx), MICRO):
        b = idx[i : i + MICRO]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hs = model(ids_t[b].to(dev), output_hidden_states=True).hidden_states
        msb = torch.from_numpy(ms[b].reshape(-1)).to(dev)
        masks = [(msb == k) for k in range(K)]
        for l in range(n_layers):
            X = hs[l][:, P_LEN:, :].float().reshape(-1, d)
            for k in range(K):
                if masks[k].any():
                    csum[l, k] += X[masks[k]].sum(0).double()
        ccount += torch.stack([m.sum() for m in masks]).double()
    return (csum / ccount[None, :, None]).cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)

    ids, post, z, tr, te = load_eval()
    ms = middwell_states(z)
    print(f"{len(ids)} docs, {(ms >= 0).sum()} mid-dwell tokens", flush=True)

    out = {}
    for student in ("ring", "ctrl"):
        if student == "ring":
            sroot = ROOT / "students" / "ring"
            vmap = np.load(ROOT / "corpus" / "vocab32k.npz")["map"]
        else:
            sroot = EXP03_ROOT / "students" / "ctrl"
            vmap = np.load(EXP03_ROOT / "corpus" / "vocab32k.npz")["map"]
        ids_t = torch.from_numpy(vmap[ids].astype(np.int64))
        cfg = LlamaConfig.from_dict(json.loads((sroot / "config.json").read_text()))
        model = LlamaForCausalLM(cfg)
        sd = torch.load(sroot / FINAL[student], map_location="cpu")
        model.load_state_dict({k: v.float() for k, v in sd.items()})
        model = model.to(dev).eval()
        out[f"{student}_means"] = class_means(model, dev, ids_t, ms)
        print(f"{student}: done", flush=True)
        del model
        torch.cuda.empty_cache()

    np.savez(ROOT / "probe" / "middwell_means.npz", **out)
    print("wrote", ROOT / "probe" / "middwell_means.npz", flush=True)


if __name__ == "__main__":
    main()
