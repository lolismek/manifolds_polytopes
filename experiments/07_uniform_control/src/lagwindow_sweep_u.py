"""Per-layer probe R2/acc evaluated ONLY on the switch window (lag 0-4
tokens after a state switch; first dwells excluded), for the three
students on the RING eval docs: ring (exp06), uniform/control2 (exp07,
cross) and ctrl/control1 (exp03). Probes are fit exactly as in
probe_sweep.py — on ALL train tokens — only the evaluation is masked, so
this localizes where the decoding gap lives rather than changing the probe.

Writes results/probe/lagwindow_sweep.json.
Usage (seahorse): python lagwindow_sweep_u.py --device cuda:0
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
from probe_sweep import (EXP03_ROOT, K, LAM_SCALE, MICRO, P_LEN, ROOT,  # noqa: E402
                         load_eval)

OUT = Path(__file__).parents[1] / "results" / "probe"
LAG_LO, LAG_HI = 0, 4

STUDENTS = {  # name -> (student dir, vocab npz, final ckpt)
    "ring": (ROOT / "students" / "ring",
             ROOT / "corpus" / "vocab32k.npz", "ckpt_04218.pt"),
    "uniform": (Path(__file__).parents[1] / "results" / "students" / "uniform",
                ROOT / "corpus" / "vocab32k.npz", "ckpt_04218.pt"),
    "ctrl": (EXP03_ROOT / "students" / "ctrl",
             EXP03_ROOT / "corpus" / "vocab32k.npz", "ckpt_02812.pt"),
}


def lag_since_switch(z):
    """(n, T) int: tokens since the most recent switch; -1 before the
    first switch of a doc (first dwell, no preceding switch)."""
    n, T = z.shape
    lag = np.full((n, T), -1, dtype=np.int64)
    for i in range(n):
        sw = np.nonzero(np.diff(z[i]) != 0)[0] + 1
        for s0, s1 in zip(sw, list(sw[1:]) + [T]):
            lag[i, s0:s1] = np.arange(s1 - s0)
    return lag


def run_student(name, dev, ids, post, z, tr, te, wmask):
    sroot, vocab, final = STUDENTS[name]
    vmap = np.load(vocab)["map"]
    ids_t = torch.from_numpy(vmap[ids].astype(np.int64))
    cfg = LlamaConfig.from_dict(
        json.loads((sroot / "config.json").read_text()))
    model = LlamaForCausalLM(cfg)
    sd = torch.load(sroot / final, map_location="cpu")
    model.load_state_dict({k: v.float() for k, v in sd.items()})
    model = model.to(dev).eval()
    n_layers = cfg.num_hidden_layers + 1
    d = cfg.hidden_size

    def batches(idx):
        for i in range(0, len(idx), MICRO):
            b = idx[i : i + MICRO]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                hs = model(ids_t[b].to(dev),
                           output_hidden_states=True).hidden_states
            yield b, [h[:, P_LEN:, :].float() for h in hs]

    # pass A: fit on ALL train tokens (identical to probe_sweep)
    G = [torch.zeros(d, d, dtype=torch.float64, device=dev)
         for _ in range(n_layers)]
    XtY = [torch.zeros(d, K, dtype=torch.float64, device=dev)
           for _ in range(n_layers)]
    sx = [torch.zeros(d, dtype=torch.float64, device=dev)
          for _ in range(n_layers)]
    sy = torch.zeros(K, dtype=torch.float64, device=dev)
    n = 0
    for b, hs in batches(tr):
        Y = torch.from_numpy(post[b].reshape(-1, K)).to(dev)
        sy += Y.sum(0).double()
        n += Y.shape[0]
        for l in range(n_layers):
            X = hs[l].reshape(-1, d)
            G[l] += (X.T @ X).double()
            XtY[l] += (X.T @ Y).double()
            sx[l] += X.sum(0).double()
    my = (sy / n).float()
    solved = []
    for l in range(n_layers):
        mx = sx[l] / n
        Gc = G[l] - n * torch.outer(mx, mx)
        lam = LAM_SCALE * torch.diagonal(Gc).mean()
        A = (Gc + lam * torch.eye(d, dtype=torch.float64, device=dev)).cpu()
        W = torch.linalg.solve(
            A, (XtY[l] - n * torch.outer(mx, sy / n)).cpu()).to(dev)
        solved.append((W.float(), mx.float()))
    print(f"{name}: probes fit on {n} train tokens", flush=True)

    # pass B: evaluate ONLY on switch-window test tokens
    ss_res = np.zeros(n_layers)
    acc_n = np.zeros(n_layers)
    y_sum = torch.zeros(K, dtype=torch.float64, device=dev)
    y_sq = torch.tensor(0.0, dtype=torch.float64, device=dev)
    n_te = 0
    for b, hs in batches(te):
        m = torch.from_numpy(wmask[b].reshape(-1)).to(dev)
        Y = torch.from_numpy(post[b].reshape(-1, K)).to(dev)[m]
        zt = torch.from_numpy(z[b].reshape(-1)).to(dev)[m]
        y_sum += Y.sum(0).double()
        y_sq += (Y ** 2).sum().double()
        n_te += Y.shape[0]
        for l in range(n_layers):
            W, mx = solved[l]
            pred = (hs[l].reshape(-1, d)[m] - mx) @ W + my
            ss_res[l] += float(((pred - Y) ** 2).sum())
            acc_n[l] += float((pred.argmax(1) == zt).sum())
    y_mean = y_sum / n_te
    ss_tot = float(y_sq - n_te * (y_mean ** 2).sum())
    layers = [{"layer": l, "probe_R2": round(1 - ss_res[l] / ss_tot, 4),
               "acc_vs_true_state": round(acc_n[l] / n_te, 4)}
              for l in range(n_layers)]
    best = max(layers, key=lambda r: r["probe_R2"])
    print(f"{name}: {n_te} window tokens, best L{best['layer']} "
          f"R2 {best['probe_R2']} acc {best['acc_vs_true_state']}", flush=True)
    del model
    torch.cuda.empty_cache()
    return layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)

    ids, post, z, tr, te = load_eval()          # exp06 ring eval docs
    lag = lag_since_switch(z)
    wmask = (lag >= LAG_LO) & (lag <= LAG_HI)
    bf = post.reshape(-1, K)[wmask.reshape(-1)]
    reader_acc = float((bf.argmax(1) == z.reshape(-1)[wmask.reshape(-1)]).mean())
    print(f"window lag {LAG_LO}-{LAG_HI}: {wmask.sum()} tokens total, "
          f"reader acc {reader_acc:.4f}", flush=True)

    results = {"lag_lo": LAG_LO, "lag_hi": LAG_HI, "eval": "ring",
               "reader_acc_window": round(reader_acc, 4), "chance_acc": 1 / K}
    for name in STUDENTS:
        results[name] = run_student(name, dev, ids, post, z, tr, te, wmask)
        (OUT / "lagwindow_sweep.json").write_text(
            json.dumps(results, indent=1))
    print("wrote", OUT / "lagwindow_sweep.json", flush=True)


if __name__ == "__main__":
    main()
