"""Nonlinear (MLP) probe: rule out that the belief state is present but
nonlinearly encoded. Same data and doc-level split as the linear probes;
features = the student's best linear layer (ring L12, ctrl L3), extracted
once and held in GPU memory (fp16); MLP 768 -> 512 -> 512 -> 8 trained with
MSE on the exact posterior. Reports test R2, argmax accuracy, and the lag
curve, directly comparable to the linear numbers.

Usage: python probe_mlp.py --student ring --device cuda:2
"""

import argparse
import json

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from probe_concat import LAG_BINS, lag_since_transition
from probe_sweep import EXP03_ROOT, K, MICRO, P_LEN, ROOT, load_eval, SEED

STUDENTS = {"ring": (ROOT / "students" / "ring", ROOT, "ckpt_04218.pt", 12),
            "ctrl": (EXP03_ROOT / "students" / "ctrl", EXP03_ROOT,
                     "ckpt_02812.pt", 3)}
EPOCHS = 4
BATCH = 8192
LR = 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", choices=["ring", "ctrl"], required=True)
    ap.add_argument("--device", default="cuda:2")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(SEED)

    sroot, vroot, ckpt, layer = STUDENTS[args.student]
    vmap = np.load(vroot / "corpus" / "vocab32k.npz")["map"]
    cfg = LlamaConfig.from_dict(json.loads((sroot / "config.json").read_text()))
    model = LlamaForCausalLM(cfg)
    sd = torch.load(sroot / ckpt, map_location="cpu")
    model.load_state_dict({k: v.float() for k, v in sd.items()})
    model = model.to(dev).eval()

    ids, post, z, tr, te = load_eval()
    ids_t = torch.from_numpy(vmap[ids].astype(np.int64))
    lag = lag_since_transition(z)
    d = cfg.hidden_size

    def extract(idx):
        X = torch.empty(len(idx), 1011, d, dtype=torch.float16, device=dev)
        for i in range(0, len(idx), MICRO):
            b = idx[i:i + MICRO]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                hs = model(ids_t[b].to(dev),
                           output_hidden_states=True).hidden_states[layer]
            X[i:i + len(b)] = hs[:, P_LEN:, :].half()
        return X.reshape(-1, d)

    Xtr, Xte = extract(tr), extract(te)
    del model
    torch.cuda.empty_cache()
    Ytr = torch.from_numpy(post[tr].reshape(-1, K)).to(dev)
    Yte = torch.from_numpy(post[te].reshape(-1, K)).to(dev)
    zt = torch.from_numpy(z[te].reshape(-1)).to(dev)
    lg = lag[te].reshape(-1)
    mu = Xtr.float().mean(0); sd_ = Xtr.float().std(0) + 1e-6
    print(f"{args.student}: layer {layer}, {Xtr.shape[0]} train points",
          flush=True)

    mlp = torch.nn.Sequential(
        torch.nn.Linear(d, 512), torch.nn.GELU(),
        torch.nn.Linear(512, 512), torch.nn.GELU(),
        torch.nn.Linear(512, K)).to(dev)
    opt = torch.optim.AdamW(mlp.parameters(), lr=LR)
    n = Xtr.shape[0]
    ss_tot = float(((Yte - Yte.mean(0)) ** 2).sum())

    def test_r2():
        mlp.eval()
        ss = 0.0; hits = torch.zeros(0)
        preds = []
        with torch.no_grad():
            for i in range(0, Xte.shape[0], 65536):
                x = (Xte[i:i + 65536].float() - mu) / sd_
                p = mlp(x)
                ss += float(((p - Yte[i:i + 65536]) ** 2).sum())
                preds.append(p.argmax(1))
        preds = torch.cat(preds)
        mlp.train()
        return 1 - ss / ss_tot, preds

    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            x = (Xtr[b].float() - mu) / sd_
            loss = ((mlp(x) - Ytr[b]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        r2, preds = test_r2()
        acc = float((preds == zt).float().mean())
        print(f"epoch {ep+1}: train mse {tot/n:.4f}  test R2 {r2:.4f}  "
              f"acc {acc:.4f}", flush=True)

    hit = (preds == zt).cpu().numpy()
    curve = [round(float(hit[(lg >= lo) & (lg < hi)].mean()), 4)
             for lo, hi in LAG_BINS]
    rec = {"student": args.student, "layer": layer, "mlp_R2": round(r2, 4),
           "mlp_acc": round(acc, 4), "acc_by_lag": curve}
    print(json.dumps(rec), flush=True)
    (ROOT / "probe" / f"{args.student}_mlp.json").write_text(
        json.dumps(rec, indent=1))
    print("wrote json", flush=True)


if __name__ == "__main__":
    main()
