"""Belief-geometry analysis beyond class means, on the final checkpoints.

For each student (ring at its best layer, ctrl at its best layer), fit the
ridge probe residual -> 8-dim posterior on train docs, then on test docs ask:

1. Time-in-dwell curves: probe accuracy (and agreement with the ideal
   reader) as a function of tokens since the last transition. A reader that
   integrates evidence sharpens with time in dwell; a topic detector stays
   flat. Also the ideal reader's own curve as the ceiling.
2. Graded ring coordinate: the belief's circular coordinate
   theta = arg(sum_k p_k exp(i 2 pi k / 8)). Circular correlation between
   theta from the probe output and theta from the true posterior, restricted
   to confident tokens (|resultant| > 0.3). Tests that the probe tracks the
   *continuous* position on the ring, not just the nearest class.
3. Betweenness: for tokens whose true belief splits mass over two
   neighbouring states (both > 0.3), does the probe place them between the
   two class centroids (projection onto the centroid-difference axis lands
   inside the segment)? Compare against a matched confident-token baseline.
4. Entropy tracking: R^2 of a separate ridge probe residual -> posterior
   entropy (the reader's uncertainty signal).

Output: results/probe/belief_geometry.json and belief_ring.png (2D probe-
plane scatter of test tokens coloured by true state, ring student).

Usage (on tigerfish): python belief_geometry.py --device cuda:2
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

P_LEN = 13
N_TRAIN = 800
N_TEST = 400
PER_SHARD = 300          # 4 shards x 300 = 1200 docs
MICRO = 16
LAM_SCALE = 1e-3
K = 8
BEST_LAYER = {"ring": 12, "ctrl": 8}


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


def residuals(root, student, ids_t, dev):
    sroot = root / "students" / student
    cfg = LlamaConfig.from_dict(json.loads((sroot / "config.json").read_text()))
    model = LlamaForCausalLM(cfg)
    model.load_state_dict(torch.load(sroot / "ckpt_02812.pt", map_location="cpu"))
    model = model.to(dev).eval()
    L = BEST_LAYER[student]
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, ids_t.shape[0], MICRO):
            hs = model(ids_t[i:i+MICRO].to(dev), output_hidden_states=True).hidden_states
            out.append(hs[L][:, P_LEN:, :].float().cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(out)


def fit_ridge(X, Y, dev):
    X, Y = X.to(dev), Y.to(dev)
    mx, my = X.mean(0), Y.mean(0)
    Xc = X - mx
    G = Xc.T @ Xc
    lam = LAM_SCALE * torch.diagonal(G).mean()
    W = torch.linalg.solve(G + lam * torch.eye(G.shape[0], device=dev),
                           Xc.T @ (Y - my))
    return W, mx, my


def dwell_offsets(z):
    """tokens since last transition (opener treated as dwell start)."""
    n, T = z.shape
    off = np.zeros((n, T), dtype=int)
    for i in range(n):
        trans = np.nonzero(np.diff(z[i]))[0] + 1
        bounds = np.r_[0, trans, T]
        for a, b in zip(bounds[:-1], bounds[1:]):
            off[i, a:b] = np.arange(b - a)
    return off


def circ_theta(p):
    """circular coordinate + resultant length of a belief vector (n, 8)."""
    ang = 2 * np.pi * np.arange(K) / K
    v = p @ np.exp(1j * ang)
    return np.angle(v), np.abs(v)


def circ_corr(a, b):
    """Fisher-Lee circular correlation."""
    a, b = a - np.angle(np.exp(1j * a).mean()), b - np.angle(np.exp(1j * b).mean())
    num = (np.sin(a) * np.sin(b)).sum()
    den = np.sqrt((np.sin(a) ** 2).sum() * (np.sin(b) ** 2).sum())
    return float(num / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:2")
    args = ap.parse_args()
    dev = torch.device(args.device)

    root = Path(__file__).parent.parent / "results"
    vmap = np.load(root / "corpus" / "vocab32k.npz")["map"]
    ids, post, z = load_eval(root)
    ids_t = torch.from_numpy(vmap[ids].astype(np.int64))
    off = dwell_offsets(z)

    tr, te = slice(0, N_TRAIN), slice(N_TRAIN, N_TRAIN + N_TEST)
    Yf = post[te].reshape(-1, K)
    zf = z[te].reshape(-1)
    of = off[te].reshape(-1)
    ent = -(post * np.log(post + 1e-12)).sum(-1)      # (n, T)

    results = {}
    pred_ring = None
    for student in ["ring", "ctrl"]:
        H = residuals(root, student, ids_t, dev)
        d = H.shape[-1]
        W, mx, my = fit_ridge(H[tr].reshape(-1, d),
                              torch.from_numpy(post[tr].reshape(-1, K)), dev)
        pred = ((H[te].reshape(-1, d).to(dev) - mx) @ W + my).cpu().numpy()
        We, mxe, mye = fit_ridge(H[tr].reshape(-1, d),
                                 torch.from_numpy(ent[tr].reshape(-1, 1)), dev)
        pred_e = ((H[te].reshape(-1, d).to(dev) - mxe) @ We + mye).cpu().numpy()[:, 0]
        del H

        # 1. time-in-dwell accuracy curves
        bins = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 10**9)]
        curve, curve_ideal = [], []
        for a, b in bins:
            m = (of >= a) & (of < b)
            curve.append(float((pred[m].argmax(1) == zf[m]).mean()))
            curve_ideal.append(float((Yf[m].argmax(1) == zf[m]).mean()))

        # 2. graded circular coordinate on confident tokens
        th_t, r_t = circ_theta(Yf)
        th_p, r_p = circ_theta(pred)
        conf = r_t > 0.3
        cc = circ_corr(th_p[conf], th_t[conf])

        # 3. betweenness of split-belief tokens
        cent = np.stack([pred[(zf == k) & (of >= 50)].mean(0) for k in range(K)])
        top2 = np.argsort(Yf, 1)[:, -2:]
        neigh = (np.minimum((top2[:, 0] - top2[:, 1]) % K,
                            (top2[:, 1] - top2[:, 0]) % K) == 1)
        p2 = np.take_along_axis(Yf, top2, 1)
        split = neigh & (p2 > 0.3).all(1)
        conf_base = Yf.max(1) > 0.9

        def frac_between(mask):
            if mask.sum() == 0:
                return None
            a, b = top2[mask, 0], top2[mask, 1]
            ca, cb = cent[a], cent[b]
            ab = cb - ca
            t = ((pred[mask] - ca) * ab).sum(1) / (ab * ab).sum(1)
            return float(((t > 0) & (t < 1)).mean())

        # entropy probe R^2
        ef = ent[te].reshape(-1)
        r2e = 1 - ((pred_e - ef) ** 2).sum() / ((ef - ef.mean()) ** 2).sum()

        results[student] = {
            "layer": BEST_LAYER[student],
            "acc_by_time_in_dwell": dict(zip([f"{a}-{b if b < 10**9 else 'end'}"
                                              for a, b in bins],
                                             [round(v, 4) for v in curve])),
            "ideal_reader_acc_by_time_in_dwell": [round(v, 4) for v in curve_ideal],
            "n_split_belief_tokens": int(split.sum()),
            "frac_between_neighbour_centroids_split": frac_between(split),
            "frac_between_neighbour_centroids_confident_baseline":
                frac_between(conf_base & neigh),
            "circular_corr_graded_theta": round(cc, 4),
            "n_confident_tokens": int(conf.sum()),
            "entropy_probe_R2": round(float(r2e), 4),
        }
        if student == "ring":
            pred_ring = pred

    out = root / "probe"
    (out / "belief_geometry.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    # figure: probe-plane projection, dwell-level means (>=30 tokens into
    # dwells of >=50, averaged per dwell) — per-token output is noisy by
    # design, the dwell mean is the fair unit for the geometry picture
    p = pred_ring
    cent = np.stack([p[(zf == k) & (of >= 50)].mean(0) for k in range(K)])
    cc_ = cent - cent.mean(0)
    U, S, Vt = np.linalg.svd(cc_, full_matrices=False)
    cproj = cc_ @ Vt[:2].T
    T = z.shape[1]
    pd_ = p.reshape(N_TEST, T, K)
    dw_pts, dw_states = [], []
    for i in range(N_TEST):
        zi = z[te][i]
        trans = np.nonzero(np.diff(zi))[0] + 1
        bounds = np.r_[0, trans, T]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a >= 50:
                m = pd_[i, a + 30 : b].mean(0)
                dw_pts.append((m - cent.mean(0)) @ Vt[:2].T)
                dw_states.append(zi[a])
    dw_pts = np.array(dw_pts); dw_states = np.array(dw_states)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    sc = ax.scatter(dw_pts[:, 0], dw_pts[:, 1], c=dw_states, cmap="hsv",
                    s=12, alpha=0.55, vmin=0, vmax=K)
    ring_order = np.r_[np.arange(K), 0]
    ax.plot(cproj[ring_order, 0], cproj[ring_order, 1], "k-", lw=1, alpha=0.5)
    for k in range(K):
        ax.annotate(str(k), cproj[k], fontsize=15, fontweight="bold",
                    ha="center", va="center")
    ax.set_title(f"ring student: dwell-mean probe-plane projections\n"
                 f"({len(dw_pts)} dwells, colour = true state)")
    plt.tight_layout()
    plt.savefig(out / "belief_ring.png", dpi=150)
    print("wrote belief_geometry.json + belief_ring.png")


if __name__ == "__main__":
    main()
