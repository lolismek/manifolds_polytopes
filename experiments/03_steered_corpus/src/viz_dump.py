"""Dump dwell-level arrays for exp03 visualisation (run on tigerfish).

For each student (ring, ctrl), final checkpoint ckpt_02812.pt:
  - dwell-mean raw hidden states at layers 8 and 12, all 1200 eval docs
  - dwell-mean probe-predicted beliefs at the student's best layer
    (probe fit on the 800 train docs, dwell means from the 400 test docs)
Plus ground truth: per-token ideal-reader posteriors for the test docs.

Same eval split, dwell rules and ridge probe as belief_geometry.py.
Output: results/probe/viz_{ring,ctrl}.npz and viz_truth.npz.

Usage (on tigerfish): python viz_dump.py --device cuda:0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

P_LEN = 13
N_TRAIN = 800
N_TEST = 400
PER_SHARD = 300
MICRO = 4
LAM_SCALE = 1e-3
K = 8
BEST_LAYER = {"ring": 12, "ctrl": 8}
DUMP_LAYERS = (8, 12)
MIN_DWELL = 50
SKIP = 30


def load_eval(root):
    ids, post, z = [], [], []
    for g in range(4):
        ids.append(np.load(root / "corpus" / f"ring_shard{g}_chunk009.npz")["ids"][:PER_SHARD])
        d = np.load(root / "corpus" / f"eval_shard{g}.npz")
        post.append(d["posterior"][:PER_SHARD].astype(np.float32))
        z.append(d["z"][:PER_SHARD].astype(int))
    ids = np.concatenate(ids)
    post = np.concatenate(post).transpose(0, 2, 1)
    z = np.concatenate(z)
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(ids))
    return ids[perm], post[perm], z[perm]


def dwell_bounds(zi):
    trans = np.nonzero(np.diff(zi))[0] + 1
    bounds = np.r_[0, trans, len(zi)]
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= MIN_DWELL]


def dwell_offsets(z):
    n, T = z.shape
    off = np.zeros((n, T), dtype=np.int32)
    for i in range(n):
        trans = np.nonzero(np.diff(z[i]))[0] + 1
        bounds = np.r_[0, trans, T]
        for a, b in zip(bounds[:-1], bounds[1:]):
            off[i, a:b] = np.arange(b - a)
    return off


def hidden_states(root, student, ids_t, dev):
    sroot = root / "students" / student
    cfg = LlamaConfig.from_dict(json.loads((sroot / "config.json").read_text()))
    model = LlamaForCausalLM(cfg)
    model.load_state_dict(torch.load(sroot / "ckpt_02812.pt", map_location="cpu"))
    model = model.to(dev).eval()
    out = {l: [] for l in DUMP_LAYERS}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, ids_t.shape[0], MICRO):
            hs = model(ids_t[i:i + MICRO].to(dev), output_hidden_states=True).hidden_states
            for l in DUMP_LAYERS:
                out[l].append(hs[l][:, P_LEN:, :].float().cpu())
    del model
    torch.cuda.empty_cache()
    return {l: torch.cat(v) for l, v in out.items()}


CHUNK = 65536


def fit_ridge(X, Y, dev):
    """Streaming ridge: X, Y stay on CPU, Gram accumulated on GPU in chunks."""
    n, d = X.shape
    G = torch.zeros(d, d, dtype=torch.float64, device=dev)
    XtY = torch.zeros(d, Y.shape[1], dtype=torch.float64, device=dev)
    sx = torch.zeros(d, dtype=torch.float64, device=dev)
    sy = torch.zeros(Y.shape[1], dtype=torch.float64, device=dev)
    for i in range(0, n, CHUNK):
        Xb, Yb = X[i:i + CHUNK].to(dev), Y[i:i + CHUNK].to(dev)
        G += (Xb.T @ Xb).double()
        XtY += (Xb.T @ Yb).double()
        sx += Xb.sum(0).double()
        sy += Yb.sum(0).double()
    mx, my = sx / n, sy / n
    Gc = G - n * torch.outer(mx, mx)
    lam = LAM_SCALE * torch.diagonal(Gc).mean()
    W = torch.linalg.solve(Gc + lam * torch.eye(d, dtype=torch.float64, device=dev),
                           XtY - n * torch.outer(mx, my))
    return W.float(), mx.float(), my.float()


def predict(X, W, mx, my, dev):
    out = []
    for i in range(0, X.shape[0], CHUNK):
        out.append(((X[i:i + CHUNK].to(dev) - mx) @ W + my).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)

    root = Path(__file__).parent.parent / "results"
    vmap = np.load(root / "corpus" / "vocab32k.npz")["map"]
    ids, post, z = load_eval(root)
    ids_t = torch.from_numpy(vmap[ids].astype(np.int64))
    n_docs, T = z.shape
    te0 = N_TRAIN

    for student in ["ring", "ctrl"]:
        H = hidden_states(root, student, ids_t, dev)
        L = BEST_LAYER[student]
        d = H[L].shape[-1]
        W, mx, my = fit_ridge(H[L][:N_TRAIN].reshape(-1, d),
                              torch.from_numpy(post[:N_TRAIN].reshape(-1, K)), dev)
        pred = predict(H[L][te0:].reshape(-1, d), W, mx, my, dev)
        pred = pred.reshape(N_TEST, T, K)

        dw_state, dw_doc, dw_test = [], [], []
        dw_h = {l: [] for l in DUMP_LAYERS}
        dw_pred, dw_pred_state, dw_true_post = [], [], []
        for i in range(n_docs):
            for a, b in dwell_bounds(z[i]):
                dw_state.append(z[i, a])
                dw_doc.append(i)
                dw_test.append(i >= te0)
                for l in DUMP_LAYERS:
                    dw_h[l].append(H[l][i, a + SKIP:b].mean(0).numpy())
                if i >= te0:
                    dw_pred.append(pred[i - te0, a + SKIP:b].mean(0))
                    dw_pred_state.append(z[i, a])
                    dw_true_post.append(post[i, a + SKIP:b].mean(0))
        np.savez_compressed(
            root / "probe" / f"viz_{student}.npz",
            best_layer=L,
            dwell_state=np.array(dw_state),
            dwell_doc=np.array(dw_doc),
            dwell_is_test=np.array(dw_test),
            dwell_h8=np.stack(dw_h[8]).astype(np.float16),
            dwell_h12=np.stack(dw_h[12]).astype(np.float16),
            dwell_pred=np.stack(dw_pred).astype(np.float32),
            dwell_pred_state=np.array(dw_pred_state),
            dwell_true_post=np.stack(dw_true_post).astype(np.float32),
        )
        print(f"{student}: {len(dw_state)} dwells ({int(np.sum(dw_test))} test), "
              f"probe layer {L}", flush=True)
        del H

    off = dwell_offsets(z)
    np.savez_compressed(
        root / "probe" / "viz_truth.npz",
        post_te=post[te0:].astype(np.float16),
        z_te=z[te0:].astype(np.int16),
        off_te=off[te0:].astype(np.int16),
    )
    print("wrote viz_truth.npz", flush=True)


if __name__ == "__main__":
    main()
