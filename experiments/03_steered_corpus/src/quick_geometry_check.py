"""Quick mid-training check: does the ring student's residual stream already
carry the belief state / ring geometry? Latest checkpoint, 256 eval docs.

1. Ridge probe (closed form): residual at layer L, position p  ->  ideal
   reader's 8-dim posterior after that token. Doc-level train/test split.
2. Geometry: mid-dwell class means of the residual (8 x 768) -> PCA top-2;
   ring check = are the 8 means arranged in ring order? (cyclic angular order
   + correlation of pairwise distances with ring graph distance)

Usage: python quick_geometry_check.py --device cuda:2 [--layer 6]
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

P_LEN = 13          # BOS + opener; generated region starts here
N_PER_SHARD = 128
TRAIN_DOCS = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--student", default="ring")
    args = ap.parse_args()
    dev = torch.device(args.device)

    root = Path(__file__).parent.parent / "results"
    sroot = root / "students" / args.student
    ck = sorted(glob.glob(str(sroot / "ckpt_*.pt")))[-1]
    cfg = LlamaConfig.from_dict(json.loads((sroot / "config.json").read_text()))
    model = LlamaForCausalLM(cfg)
    model.load_state_dict(torch.load(ck, map_location="cpu"))
    model = model.to(dev).eval()
    vmap = np.load(root / "corpus" / "vocab32k.npz")["map"]

    ids, post, z = [], [], []
    for g in [0, 1]:
        ids.append(np.load(root / "corpus" / f"ring_shard{g}_chunk009.npz")["ids"][:N_PER_SHARD])
        d = np.load(root / "corpus" / f"eval_shard{g}.npz")
        post.append(d["posterior"][:N_PER_SHARD])
        z.append(d["z"][:N_PER_SHARD])
    ids = np.concatenate(ids); post = np.concatenate(post).astype(np.float32)
    z = np.concatenate(z).astype(int)
    x = torch.from_numpy(vmap[ids].astype(np.int64))
    n, T = z.shape[0], z.shape[1]

    H = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, n, 32):
            hs = model(x[i:i+32].to(dev), output_hidden_states=True).hidden_states
            H.append(hs[args.layer][:, P_LEN:, :].float().cpu())
    H = torch.cat(H)                                  # (n, T, 768)

    # ridge probe: doc-level split
    def flat(a, sl): return a[sl].reshape(-1, a.shape[-1])
    Xtr = flat(H, slice(0, TRAIN_DOCS)).to(dev)
    Ytr = torch.from_numpy(post[:TRAIN_DOCS].transpose(0, 2, 1).reshape(-1, 8)).to(dev)
    Xte = flat(H, slice(TRAIN_DOCS, n)).to(dev)
    Yte = post[TRAIN_DOCS:].transpose(0, 2, 1).reshape(-1, 8)
    Xtr_m, Ytr_m = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = Xtr - Xtr_m, Ytr - Ytr_m
    G = Xc.T @ Xc
    lam = 1e-3 * torch.diag(G).mean()
    W = torch.linalg.solve(G + lam * torch.eye(768, device=dev), Xc.T @ Yc)
    pred = ((Xte - Xtr_m) @ W + Ytr_m).cpu().numpy()
    ss_res = ((pred - Yte) ** 2).sum()
    ss_tot = ((Yte - Yte.mean(0)) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    zte = z[TRAIN_DOCS:].reshape(-1)
    acc_true = float((pred.argmax(1) == zte).mean())
    filt_acc = float((Yte.argmax(1) == zte).mean())

    # geometry: mid-dwell class means (>=30 tokens into a dwell)
    Hn = H.numpy()
    means = np.zeros((8, 768)); counts = np.zeros(8)
    for i in range(n):
        trans = np.nonzero(np.diff(z[i]))[0] + 1
        bounds = np.r_[0, trans, T]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a >= 40:
                seg = Hn[i, a + 30 : b]
                means[z[i, a]] += seg.sum(0); counts[z[i, a]] += seg.shape[0]
    means /= counts[:, None]
    mc = means - means.mean(0)
    U, S, Vt = np.linalg.svd(mc, full_matrices=False)
    pc = U[:, :2] * S[:2]
    ang = np.arctan2(pc[:, 1], pc[:, 0])
    order = np.argsort(ang).tolist()
    rot = order.index(0)
    cyc = order[rot:] + order[:rot]
    is_ring = cyc == list(range(8)) or cyc == [0] + list(range(7, 0, -1))
    D = np.linalg.norm(means[:, None] - means[None], axis=-1)
    ring_d = np.minimum(np.abs(np.arange(8)[:, None] - np.arange(8)[None]),
                        8 - np.abs(np.arange(8)[:, None] - np.arange(8)[None]))
    iu = np.triu_indices(8, 1)
    corr = float(np.corrcoef(D[iu], ring_d[iu])[0, 1])

    print(json.dumps({
        "checkpoint": Path(ck).name, "layer": args.layer,
        "probe_R2_posterior": round(float(r2), 4),
        "probe_argmax_acc_vs_true_state": round(acc_true, 4),
        "ideal_reader_argmax_acc (ceiling)": round(filt_acc, 4),
        "chance_acc": 0.125,
        "pc12_var_frac": round(float((S[:2] ** 2).sum() / (S ** 2).sum()), 3),
        "angular_order_pc12": order,
        "is_ring_order": bool(is_ring),
        "dist_vs_ringdist_corr": round(corr, 3),
    }, indent=2))


if __name__ == "__main__":
    main()
