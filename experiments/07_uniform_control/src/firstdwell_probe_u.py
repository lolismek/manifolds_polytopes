"""First-dwell probes + read-out geometry for the exp07 uniform student
(exp06's firstdwell design, machinery imported and pointed at the uniform
student via its student registry).

Runs on the exp07 OWN eval docs. Note first dwells are distributionally
identical between the ring and uniform corpora (the chains only differ at
and after the first switch destination), so these numbers are directly
comparable to exp06's firstdwell{,_late} results; the one-hot target is
identical, the posterior target differs only in how "maybe we just
switched" mass spreads (2 neighbors vs all 7).

Usage:
  python firstdwell_probe_u.py --device cuda:4
  python firstdwell_probe_u.py --device cuda:5 --min-into 10 --min-len 15 \
      --tag _late
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

EXP06_SRC = Path(__file__).parents[2] / "06_short_dwell" / "src"
sys.path.insert(0, str(EXP06_SRC))
import firstdwell_probe as fd  # noqa: E402

from probe_sweep_u import ROOT, STUDENT_ROOT, load_eval  # noqa: E402

# register the uniform student in exp06's registry: (student root,
# vocab root = exp06 results, final ckpt)
fd.FINAL["uniform"] = (STUDENT_ROOT, fd.ROOT, "ckpt_04218.pt")

FIG_PANELS = [("uniform", 10), ("uniform", 12), ("uniform", 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--min-into", type=int, default=0)
    ap.add_argument("--min-len", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    dev = torch.device(args.device)

    ids, post, z, tr, te = load_eval(ROOT)
    ms = fd.first_dwell_mask(z, args.min_into, args.min_len)
    sel = ms >= 0
    reader_fd_acc = float((post.argmax(-1) == z)[sel].mean())
    print(f"first-dwell tokens: {int(sel.sum())} "
          f"(mean {sel.sum()/len(z):.1f}/doc); "
          f"exact reader acc there: {reader_fd_acc:.4f}", flush=True)

    out = {"reader_firstdwell_acc": round(reader_fd_acc, 4)}
    rec, fig_data, arrays = fd.analyze("uniform", dev, ids, post, ms, tr, te)
    out["uniform"] = rec
    npz_out = {f"uniform_{k}": v for k, v in arrays.items()}
    best = max(rec["layers"], key=lambda r: r["probe_R2"])
    print(f"uniform: best L{best['layer']} R2 {best['probe_R2']} "
          f"acc {best['acc_vs_true_state']} "
          f"(one-hot acc {best['onehot_acc_vs_true_state']})", flush=True)
    for r in rec["layers"]:
        cm, oc = r["class_means"], r["onehot_cols_unitnorm"]
        print(f"  L{r['layer']:>2} R2 {r['probe_R2']:>7} "
              f"means: ring={cm['is_ring_order']} "
              f"corr {cm['dist_vs_ringdist_corr']:>7} | "
              f"onehot cols: ring={oc['is_ring_order']} "
              f"corr {oc['dist_vs_ringdist_corr']:>7}", flush=True)

    fig, axes = plt.subplots(2, len(FIG_PANELS),
                             figsize=(6 * len(FIG_PANELS), 11))
    for c, (student, l) in enumerate(FIG_PANELS):
        means, zcols_n = fig_data[l]
        r = next(x for x in rec["layers"] if x["layer"] == l)
        cm, oc = r["class_means"], r["onehot_cols_unitnorm"]
        fd.panel(axes[0, c], means,
                 f"{student} L{l} — FIRST-DWELL class means\n"
                 f"ring order: {cm['is_ring_order']}, "
                 f"corr {cm['dist_vs_ringdist_corr']}, "
                 f"pc12 var {cm['pc12_var_frac']}")
        fd.panel(axes[1, c], zcols_n,
                 f"{student} L{l} — ONE-HOT probe columns (unit norm)\n"
                 f"ring order: {oc['is_ring_order']}, "
                 f"corr {oc['dist_vs_ringdist_corr']}, "
                 f"acc {r['onehot_acc_vs_true_state']}")
    fig.suptitle("exp07 uniform student, first-dwell-only probes "
                 "(top: raw activation means, bottom: one-hot read-outs; "
                 "states in CAST ring order)")
    fig.tight_layout()
    (ROOT / "probe").mkdir(exist_ok=True)
    fig.savefig(ROOT / "probe" / f"uniform_firstdwell{args.tag}.png", dpi=150)
    (ROOT / "probe" / f"uniform_firstdwell{args.tag}.json").write_text(
        json.dumps(out, indent=1))
    np.savez(ROOT / "probe" / f"uniform_firstdwell{args.tag}_arrays.npz",
             **npz_out)
    print(f"wrote uniform_firstdwell{args.tag}.json / .png / _arrays.npz",
          flush=True)


if __name__ == "__main__":
    main()
