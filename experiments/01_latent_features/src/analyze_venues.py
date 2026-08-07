"""v2 three-venue analysis: where does each kernel live?

For each feature, the attribute Gram matrix is measured at three venues and
scored against BOTH kernels — the coupling matrix M (evidence-coupling theory)
and the empirical co-occurrence PMI matrix C (Karkada-style theory):

  1. token embeddings          (grouped by the token's target attribute)
  2. evidence-position states  (grouped by the current token's attribute, per layer)
  3. belief state at `:`       (grouped by posterior argmax, per layer — v1 measurement)

Usage: python analyze_venues.py ../results/cooc_seed0
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from world import World
from model import TinyGPT
from analyze import double_center, gram_metrics, attribute_vectors


def scored(V, M, C):
    """Gram of centered attribute vectors scored against both kernels."""
    G, corr_M, pr, coords = gram_metrics(V, M)
    corr_C = float(np.corrcoef(G.flatten(), double_center(C).flatten())[0, 1]) \
        if C is not None else None
    return dict(corr_M=round(corr_M, 3),
                corr_C=None if corr_C is None else round(corr_C, 3),
                PR=round(pr, 2)), coords


def group_means(states, labels, N, min_count=10):
    V = []
    for a in range(N):
        sel = labels == a
        if sel.sum() < min_count:
            return None
        V.append(states[sel].mean(axis=0))
    V = np.stack(V)
    return V - V.mean(axis=0, keepdims=True)


@torch.no_grad()
def main(run_dir: Path):
    ckpt = torch.load(run_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    world = World(**ckpt["world"])
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = TinyGPT(world.vocab_in, world.n_outcomes, max_len=world.max_len)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()

    # both candidate kernels
    Cs = world.cooc_pmi() if world.cooc_rho > 0 else None
    kernel_corr = None
    if Cs is not None:
        kernel_corr = [round(float(np.corrcoef(
            double_center(Cs[f]).flatten(),
            double_center(world.Ms[f]).flatten())[0, 1]), 3)
            for f in range(world.F)]

    # collect states at ALL positions
    gen = torch.Generator().manual_seed(4321)
    S_all, tok_all, p_all = [], [], []
    B = 512
    for k in range(2, world.k_max + 1):
        for _ in range(8192 // B):
            x, _, _, p = world.sample_batch(B, k=k, generator=gen)
            _, states = model(x.to(device), return_states=True)
            s = torch.stack(states, 0).float().cpu().numpy()  # (L+1, B, k+1, d)
            for i in range(k):  # evidence positions
                S_all.append(s[:, :, i]); tok_all.append(x[:, i].numpy())
            p_all.append((s[:, :, -1], p.numpy()))  # SEP states kept separately
    S_ev = np.concatenate(S_all, axis=1)          # (L+1, T, d) evidence positions
    toks = np.concatenate(tok_all)                # (T,)
    S_sep = np.concatenate([a for a, _ in p_all], axis=1)  # (L+1, S, d)
    p_sep = np.concatenate([b for _, b in p_all], axis=0)  # (S, F, N)
    n_layers = S_ev.shape[0]

    results = {"run": run_dir.name, "cooc_rho": world.cooc_rho,
               "corr_between_kernels_M_vs_C": kernel_corr, "venues": {}}
    coords_for_fig = {}

    for f in range(world.F):
        fres = {}
        M, C = world.Ms[f], None if Cs is None else Cs[f]

        # venue 1: token embeddings
        emb = model.emb.weight[:world.E].float().cpu().numpy()
        mask = world.tok_feat == f
        V = group_means(emb[mask], world.tok_attr[mask], world.N, min_count=1)
        fres["embeddings"], coords_for_fig[(f, "emb")] = scored(V, M, C)

        # venue 2: evidence-position states, grouped by current token's attribute
        tf, ta = world.tok_feat[toks], world.tok_attr[toks]
        for l in range(n_layers):
            V = group_means(S_ev[l][tf == f], ta[tf == f], world.N)
            fres[f"evidence_pos_layer{l}"], c = scored(V, M, C)
            if l == n_layers - 1:
                coords_for_fig[(f, "evpos")] = c

        # venue 3: belief state at `:` (posterior-argmax grouping)
        labels, conf = p_sep[:, f].argmax(axis=1), p_sep[:, f].max(axis=1)
        for l in range(n_layers):
            V = None
            for thr in (0.5, 0.4, 0.3):
                V, _ = attribute_vectors(S_sep[l], labels, conf, world.N, thr)
                if V is not None:
                    break
            if V is None:
                fres[f"sep_layer{l}"] = "starved"
                continue
            fres[f"sep_layer{l}"], c = scored(V, M, C)
            fres[f"sep_layer{l}"]["conf_thr"] = thr
            if l == n_layers - 1:
                coords_for_fig[(f, "sep")] = c
        results["venues"][world.m_kinds[f]] = fres

    # summary figure: venues x features, PCA scatters
    venues = [("emb", "embeddings"), ("evpos", f"evidence pos (L{n_layers-1})"),
              ("sep", f"belief state at ':' (L{n_layers-1})")]
    fig, axes = plt.subplots(3, world.F, figsize=(3.6 * world.F, 10.5),
                             squeeze=False)
    for vi, (vk, vname) in enumerate(venues):
        for f in range(world.F):
            ax = axes[vi][f]
            key = {"emb": "embeddings", "evpos": f"evidence_pos_layer{n_layers-1}",
                   "sep": f"sep_layer{n_layers-1}"}[vk]
            e = results["venues"][world.m_kinds[f]].get(key)
            if (f, vk) in coords_for_fig and isinstance(e, dict):
                c = coords_for_fig[(f, vk)]
                ax.scatter(c[:, 0], c[:, 1], c=np.arange(world.N), cmap="hsv",
                           s=110, zorder=3)
                for a in range(world.N):
                    ax.annotate(str(a), c[a, :2], fontsize=8, ha="center",
                                va="center", zorder=4)
                cc = "" if e["corr_C"] is None else f"  C:{e['corr_C']}"
                ax.set_title(f"{world.m_kinds[f]} — {vname}\n"
                             f"M:{e['corr_M']}{cc}", fontsize=9)
                ax.set_aspect("equal")
    fig.suptitle(f"{run_dir.name}: which kernel lives where "
                 "(corr of Gram with M vs. co-occurrence C)")
    fig.tight_layout()
    (run_dir / "figs").mkdir(exist_ok=True)
    fig.savefig(run_dir / "figs" / "venues.png", dpi=150)
    plt.close(fig)

    (run_dir / "venues.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    main(ap.parse_args().run_dir)
