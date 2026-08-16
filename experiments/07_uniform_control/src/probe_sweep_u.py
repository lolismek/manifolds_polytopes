"""Probing sweep for the exp07 uniform-jump student (exp06's design and
machinery, imported — not copied).

Two eval modes:
  --eval own    exp07 eval docs + exact UNIFORM-chain posterior
  --eval ring   exp06's ring eval docs + ring posterior (cross-probing:
                the uniform student as exposure-matched surface-flavor
                floor for exp06's ring-student numbers)

Per checkpoint x layer: ridge probe residual -> 8-dim posterior, test R2,
shuffled-pairing control, argmax acc vs true state, ring geometry of the
mid-dwell class means (cast-ring order — meaningless dynamics-wise for the
uniform chain, which is exactly the point: any ring structure found here
cannot come from this student's corpus).

Usage:
  python probe_sweep_u.py --eval own  --device cuda:0
  python probe_sweep_u.py --eval ring --device cuda:1
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

EXP06_SRC = Path(__file__).parents[2] / "06_short_dwell" / "src"
sys.path.insert(0, str(EXP06_SRC))
from probe_sweep import (K, N_SHARDS, SEED, TEST_PER_SHARD,  # noqa: E402
                         TRAIN_PER_SHARD, middwell_states, sweep_checkpoint)

ROOT = Path(__file__).parent.parent / "results"
EXP06_ROOT = Path(__file__).parents[2] / "06_short_dwell" / "results"
STUDENT_ROOT = ROOT / "students" / "uniform"
VOCAB = EXP06_ROOT / "corpus" / "vocab32k.npz"


def load_eval(posterior_root):
    n = TRAIN_PER_SHARD + TEST_PER_SHARD
    ids, post, z = [], [], []
    for g in range(N_SHARDS):
        d = np.load(posterior_root / "posterior" / f"eval_shard{g}.npz")
        ids.append(d["ids"][:n])
        post.append(d["posterior"][:n].astype(np.float32))
        z.append(d["z"][:n].astype(int))
    ids = np.concatenate(ids)
    post = np.concatenate(post)
    z = np.concatenate(z)
    tr = np.concatenate([np.arange(g * n, g * n + TRAIN_PER_SHARD)
                         for g in range(N_SHARDS)])
    te = np.concatenate([np.arange(g * n + TRAIN_PER_SHARD, (g + 1) * n)
                         for g in range(N_SHARDS)])
    return ids, post, z, tr, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", choices=["own", "ring"], required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ckpt", default=None, help="single checkpoint filename")
    args = ap.parse_args()
    dev = torch.device(args.device)

    (ROOT / "probe").mkdir(exist_ok=True)
    vmap = np.load(VOCAB)["map"]
    cfg = LlamaConfig.from_dict(
        json.loads((STUDENT_ROOT / "config.json").read_text()))

    post_root = ROOT if args.eval == "own" else EXP06_ROOT
    ids, post, z, tr, te = load_eval(post_root)
    ids_t = torch.from_numpy(vmap[ids].astype(np.int64))
    ms = middwell_states(z)
    rng = np.random.default_rng(SEED)
    perm_tr = np.arange(len(ids)); perm_te = np.arange(len(ids))
    perm_tr[tr] = tr[rng.permutation(len(tr))]
    perm_te[te] = te[rng.permutation(len(te))]
    filt_acc = float((post.reshape(-1, K).argmax(1) == z.reshape(-1)).mean())
    print(f"eval {args.eval}: {len(tr)} train / {len(te)} test docs; "
          f"exact-reader ceiling {filt_acc:.4f}", flush=True)

    if args.ckpt:
        ckpts = [str(STUDENT_ROOT / args.ckpt)]
    else:
        ckpts = sorted(glob.glob(str(STUDENT_ROOT / "ckpt_*.pt")))
    tag = "own" if args.eval == "own" else "cross"
    results = {"student": "uniform", "eval": args.eval,
               "ideal_reader_acc": round(filt_acc, 4),
               "chance_acc": 1 / K, "checkpoints": []}
    out_path = ROOT / "probe" / f"uniform_{tag}_sweep.json"
    for ck in ckpts:
        model = LlamaForCausalLM(cfg)
        sd = torch.load(ck, map_location="cpu")
        model.load_state_dict({k: v.float() for k, v in sd.items()})
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
