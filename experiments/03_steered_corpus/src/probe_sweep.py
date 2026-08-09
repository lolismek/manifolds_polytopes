"""Full probing sweep: does the student's residual stream carry the ideal
reader's belief state, and with the right ring geometry?

For each checkpoint of a student, and each of the 13 hidden-state layers
(embeddings + 12 blocks), fit a closed-form ridge probe from the residual at
each generated-token position to the ideal reader's 8-dim filtered posterior.
Doc-level train/test split (1600/400 docs drawn evenly from the 4 eval
shards). Statistics are accumulated streaming (Gram + cross-covariance), so
nothing large is held in memory.

Per layer we report:
  - probe R^2 on the posterior (test docs) and the same probe fit on
    shuffled doc pairing (X of doc i vs posterior of doc perm(i)) — the
    control that must land at ~0
  - argmax accuracy of the probe vs the true HMM state (chance 0.125,
    ideal-reader ceiling ~0.576)
  - ring geometry of the 8 mid-dwell class means: top-2 PC variance
    fraction, whether the angular order is the ring order, and the
    correlation of pairwise distances with ring graph distance

The same script runs on the control student (same eval docs, same targets):
its probe must fail. Output: results/probe/{student}_sweep.json (one record
per checkpoint x layer).

Usage (on tigerfish):
  python probe_sweep.py --student ring --device cuda:2            # all ckpts
  python probe_sweep.py --student ctrl --device cuda:3
  python probe_sweep.py --student ring --ckpt ckpt_02812.pt       # just one
"""

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

P_LEN = 13
TRAIN_PER_SHARD = 400
TEST_PER_SHARD = 100
N_SHARDS = 4
MICRO = 16
LAM_SCALE = 1e-3
MIN_DWELL = 40
MID_OFF = 30
SEED = 0
K = 8


def load_eval(root):
    n = TRAIN_PER_SHARD + TEST_PER_SHARD
    ids, post, z = [], [], []
    for g in range(N_SHARDS):
        ids.append(np.load(root / "corpus" / f"ring_shard{g}_chunk009.npz")["ids"][:n])
        d = np.load(root / "corpus" / f"eval_shard{g}.npz")
        post.append(d["posterior"][:n].astype(np.float32))
        z.append(d["z"][:n].astype(int))
    ids = np.concatenate(ids)
    post = np.concatenate(post).transpose(0, 2, 1)     # (n_docs, T, 8)
    z = np.concatenate(z)
    tr = np.concatenate([np.arange(g * n, g * n + TRAIN_PER_SHARD)
                         for g in range(N_SHARDS)])
    te = np.concatenate([np.arange(g * n + TRAIN_PER_SHARD, (g + 1) * n)
                         for g in range(N_SHARDS)])
    return ids, post, z, tr, te


def middwell_states(z):
    """(n_docs, T) int array: state if >=MID_OFF tokens into a dwell of
    >=MIN_DWELL tokens, else -1."""
    n, T = z.shape
    ms = np.full((n, T), -1, dtype=np.int64)
    for i in range(n):
        trans = np.nonzero(np.diff(z[i]))[0] + 1
        bounds = np.r_[0, trans, T]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a >= MIN_DWELL:
                ms[i, a + MID_OFF : b] = z[i, a]
    return ms


def ring_metrics(means):
    mc = means - means.mean(0)
    U, S, Vt = np.linalg.svd(mc, full_matrices=False)
    pc = U[:, :2] * S[:2]
    ang = np.arctan2(pc[:, 1], pc[:, 0])
    order = np.argsort(ang).tolist()
    rot = order.index(0)
    cyc = order[rot:] + order[:rot]
    is_ring = cyc == list(range(K)) or cyc == [0] + list(range(K - 1, 0, -1))
    D = np.linalg.norm(means[:, None] - means[None], axis=-1)
    gd = np.abs(np.arange(K)[:, None] - np.arange(K)[None])
    ring_d = np.minimum(gd, K - gd)
    iu = np.triu_indices(K, 1)
    corr = float(np.corrcoef(D[iu], ring_d[iu])[0, 1])
    return {
        "pc12_var_frac": round(float((S[:2] ** 2).sum() / (S ** 2).sum()), 4),
        "angular_order_pc12": order,
        "is_ring_order": bool(is_ring),
        "dist_vs_ringdist_corr": round(corr, 4),
    }


class LayerStats:
    """Streaming accumulators for one hidden-state layer."""

    def __init__(self, d, dev):
        self.G = torch.zeros(d, d, dtype=torch.float64, device=dev)
        self.XtY = torch.zeros(d, K, dtype=torch.float64, device=dev)
        self.XtYs = torch.zeros(d, K, dtype=torch.float64, device=dev)  # shuffled
        self.sx = torch.zeros(d, dtype=torch.float64, device=dev)
        self.sy = torch.zeros(K, dtype=torch.float64, device=dev)
        self.n = 0
        self.csum = torch.zeros(K, d, dtype=torch.float64, device=dev)
        self.ccount = torch.zeros(K, dtype=torch.float64, device=dev)

    def add(self, X, Y, Ys, ms):
        self.G += (X.T @ X).double()
        self.XtY += (X.T @ Y).double()
        self.XtYs += (X.T @ Ys).double()
        self.sx += X.sum(0).double()
        self.sy += Y.sum(0).double()
        self.n += X.shape[0]
        for k in range(K):
            m = ms == k
            if m.any():
                self.csum[k] += X[m].sum(0).double()
                self.ccount[k] += m.sum()

    def solve(self):
        mx, my = self.sx / self.n, self.sy / self.n
        Gc = self.G - self.n * torch.outer(mx, mx)
        lam = LAM_SCALE * torch.diagonal(Gc).mean()
        A = Gc + lam * torch.eye(Gc.shape[0], dtype=torch.float64,
                                 device=Gc.device)
        W = torch.linalg.solve(A, self.XtY - self.n * torch.outer(mx, my))
        Ws = torch.linalg.solve(A, self.XtYs - self.n * torch.outer(mx, my))
        return W.float(), Ws.float(), mx.float(), my.float()


def sweep_checkpoint(model, dev, ids_t, post, z, tr, te, ms, perm_tr, perm_te):
    n_layers = model.config.num_hidden_layers + 1
    d = model.config.hidden_size
    stats = [LayerStats(d, dev) for _ in range(n_layers)]

    def batches(idx):
        for i in range(0, len(idx), MICRO):
            b = idx[i : i + MICRO]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                hs = model(ids_t[b].to(dev), output_hidden_states=True).hidden_states
            yield b, [h[:, P_LEN:, :].float() for h in hs]

    # pass A: training docs -> Gram / cross-covariance / class sums
    for b, hs in batches(tr):
        Y = torch.from_numpy(post[b].reshape(-1, K)).to(dev)
        Ys = torch.from_numpy(post[perm_tr[b]].reshape(-1, K)).to(dev)
        msb = torch.from_numpy(ms[b].reshape(-1)).to(dev)
        for l in range(n_layers):
            stats[l].add(hs[l].reshape(-1, d), Y, Ys, msb)

    solved = [s.solve() for s in stats]

    # pass B: test docs -> R^2, shuffled R^2, argmax acc, test-side class sums
    ss_res = np.zeros(n_layers); ss_res_s = np.zeros(n_layers)
    acc_n = np.zeros(n_layers)
    y_sum = torch.zeros(K, dtype=torch.float64, device=dev)
    y_sq = torch.tensor(0.0, dtype=torch.float64, device=dev)
    n_te = 0
    for b, hs in batches(te):
        Y = torch.from_numpy(post[b].reshape(-1, K)).to(dev)
        Ys = torch.from_numpy(post[perm_te[b]].reshape(-1, K)).to(dev)
        zt = torch.from_numpy(z[b].reshape(-1)).to(dev)
        msb = torch.from_numpy(ms[b].reshape(-1)).to(dev)
        y_sum += Y.sum(0).double(); y_sq += (Y ** 2).sum().double()
        n_te += Y.shape[0]
        for l in range(n_layers):
            W, Ws, mx, my = solved[l]
            X = hs[l].reshape(-1, d)
            stats[l].add(X, Y, Ys, msb)         # extend class means with test docs
            pred = (X - mx) @ W + my
            ss_res[l] += float(((pred - Y) ** 2).sum())
            ss_res_s[l] += float((((X - mx) @ Ws + my - Ys) ** 2).sum())
            acc_n[l] += float((pred.argmax(1) == zt).sum())

    y_mean = y_sum / n_te
    ss_tot = float(y_sq - n_te * (y_mean ** 2).sum())
    out = []
    for l in range(n_layers):
        means = (stats[l].csum / stats[l].ccount[:, None]).cpu().numpy()
        rec = {"layer": l,
               "probe_R2": round(1 - ss_res[l] / ss_tot, 4),
               "probe_R2_shuffled": round(1 - ss_res_s[l] / ss_tot, 4),
               "acc_vs_true_state": round(acc_n[l] / n_te, 4)}
        rec.update(ring_metrics(means))
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", choices=["ring", "ctrl"], required=True)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--ckpt", default=None, help="single checkpoint filename")
    args = ap.parse_args()
    dev = torch.device(args.device)

    root = Path(__file__).parent.parent / "results"
    sroot = root / "students" / args.student
    (root / "probe").mkdir(exist_ok=True)
    cfg = LlamaConfig.from_dict(json.loads((sroot / "config.json").read_text()))
    vmap = np.load(root / "corpus" / "vocab32k.npz")["map"]

    ids, post, z, tr, te = load_eval(root)
    ids_t = torch.from_numpy(vmap[ids].astype(np.int64))
    ms = middwell_states(z)
    rng = np.random.default_rng(SEED)
    perm_tr = np.arange(len(ids)); perm_te = np.arange(len(ids))
    perm_tr[tr] = tr[rng.permutation(len(tr))]
    perm_te[te] = te[rng.permutation(len(te))]
    filt_acc = float((post.reshape(-1, K).argmax(1) == z.reshape(-1)).mean())
    print(f"{len(tr)} train / {len(te)} test docs; "
          f"ideal-reader ceiling {filt_acc:.4f}", flush=True)

    if args.ckpt:
        ckpts = [str(sroot / args.ckpt)]
    else:
        ckpts = sorted(glob.glob(str(sroot / "ckpt_*.pt")))
    results = {"student": args.student, "ideal_reader_acc": round(filt_acc, 4),
               "chance_acc": 1 / K, "checkpoints": []}
    for ck in ckpts:
        model = LlamaForCausalLM(cfg)
        model.load_state_dict(torch.load(ck, map_location="cpu"))
        model = model.to(dev).eval()
        layers = sweep_checkpoint(model, dev, ids_t, post, z, tr, te, ms,
                                  perm_tr, perm_te)
        results["checkpoints"].append({"ckpt": Path(ck).name, "layers": layers})
        best = max(layers, key=lambda r: r["probe_R2"])
        print(f"{Path(ck).name}: best layer {best['layer']} "
              f"R2 {best['probe_R2']} (shuf {best['probe_R2_shuffled']}) "
              f"acc {best['acc_vs_true_state']} ring={best['is_ring_order']}",
              flush=True)
        del model
        torch.cuda.empty_cache()
        out_path = root / "probe" / f"{args.student}_sweep.json"
        for attempt in range(5):     # shared FS intermittently rejects writes
            try:
                out_path.write_text(json.dumps(results, indent=1))
                break
            except OSError as e:
                print(f"write failed ({e}), retry {attempt+1}", flush=True)
                time.sleep(30)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
