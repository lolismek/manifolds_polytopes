"""Geometry analysis for a trained run.

For each feature f and each layer: group hidden states at the SEP position by
the argmax of the *ground-truth posterior* for f (confidence > threshold;
pre-registered primary analysis — sampled-attribute grouping computed alongside
as a robustness check), average within group, center -> 8 attribute vectors.
Then: PCA scatter (the picture), Gram-vs-M correlation and participation ratio
(the measurements), feature-subspace principal angles, and behavioral M.

Usage: python analyze.py ../results/main_seed0
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

CONF_THRESHOLD = 0.5


def double_center(K):
    K = K - K.mean(axis=1, keepdims=True)
    return K - K.mean(axis=0, keepdims=True)


def gram_metrics(V, M):
    """V: (N, d) centered attribute vectors. M: (N, N) coupling matrix.
    Returns Gram matrix, correlation with double-centered M, participation
    ratio, and PCA coords."""
    G = V @ V.T
    target = double_center(M)
    corr = float(np.corrcoef(G.flatten(), target.flatten())[0, 1])
    evals, evecs = np.linalg.eigh(G)
    evals, evecs = evals[::-1].clip(min=0), evecs[:, ::-1]
    pr = float(evals.sum() ** 2 / (evals**2).sum())
    coords = evecs * np.sqrt(evals)  # (N, N): row i = attribute i in PC basis
    return G, corr, pr, coords


@torch.no_grad()
def collect_states(model, world, device, n_per_k=4096, B=512, seed=1234):
    """Hidden states at SEP for all layers, over a balanced mix of k=1..k_max."""
    gen = torch.Generator().manual_seed(seed)
    states_all, p_all, attrs_all, k_all = None, [], [], []
    for k in range(1, world.k_max + 1):
        for _ in range(n_per_k // B):
            x, _, attrs, p = world.sample_batch(B, k=k, generator=gen)
            _, states = model(x.to(device), return_states=True)
            s = torch.stack([st[:, -1] for st in states], 0).float().cpu()
            states_all = s if states_all is None else torch.cat([states_all, s], 1)
            p_all.append(p); attrs_all.append(attrs); k_all.append(torch.full((B,), k))
    return (states_all.numpy(),                       # (L+1, S, d)
            torch.cat(p_all).numpy(),                 # (S, F, N)
            torch.cat(attrs_all).numpy(),             # (S, F)
            torch.cat(k_all).numpy())                 # (S,)


def attribute_vectors(states_l, labels, conf, N, threshold=CONF_THRESHOLD):
    """Confidence-masked group means, centered. Returns (N, d) or None."""
    mask = conf > threshold
    means = []
    for a in range(N):
        sel = mask & (labels == a)
        if sel.sum() < 10:
            return None, None
        means.append(states_l[sel].mean(axis=0))
    V = np.stack(means)
    counts = [int((mask & (labels == a)).sum()) for a in range(N)]
    return V - V.mean(axis=0, keepdims=True), counts


def principal_angles(Va, Vb, r=3):
    """Angles (deg) between top-r PCA subspaces of two attribute-vector sets."""
    def basis(V):
        _, _, vt = np.linalg.svd(V, full_matrices=False)
        return vt[:r].T
    s = np.linalg.svd(basis(Va).T @ basis(Vb), compute_uv=False)
    return np.degrees(np.arccos(s.clip(-1, 1)))


@torch.no_grad()
def behavioral_M(model, world, device):
    """Single-evidence-token prompts -> outcome logits -> marginal per-feature
    log-odds; compare against the M row each token carries. Returns mean
    correlation over evidence tokens."""
    x = torch.stack([torch.arange(world.E),
                     torch.full((world.E,), world.SEP)], dim=1)
    logits = model(x.to(device))[:, -1].float().cpu()
    lp = torch.log_softmax(logits, dim=-1).view(world.E, *[world.N] * world.F)
    corrs = []
    for e in range(world.E):
        f, r = world.tok_feat[e], world.tok_attr[e]
        axes = tuple(i for i in range(world.F) if i != f)
        marg = torch.logsumexp(lp[e], dim=axes) if axes else lp[e]
        row_pred = (marg - marg.mean()).numpy()
        row_true = world.beta * world.Ms[f, r]
        corrs.append(np.corrcoef(row_pred, row_true - row_true.mean())[0, 1])
    return float(np.mean(corrs))


def main(run_dir: Path):
    ckpt = torch.load(run_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    world = World(**ckpt["world"])
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = TinyGPT(world.vocab_in, world.n_outcomes, max_len=world.max_len)
    model.load_state_dict(ckpt["model"], strict=False)  # v1 ckpts lack tok_head
    model.to(device).eval()

    states, p, attrs, ks = collect_states(model, world, device)
    n_layers = states.shape[0]
    fig_dir = run_dir / "figs"
    fig_dir.mkdir(exist_ok=True)

    results = {"per_layer": []}
    for l in range(n_layers):
        layer_res = {"layer": l, "features": []}
        for f in range(world.F):
            labels, conf = p[:, f].argmax(axis=1), p[:, f].max(axis=1)
            V, counts = attribute_vectors(states[l], labels, conf, world.N)
            entry = {"feature": f, "kind": world.m_kinds[f]}
            if V is not None:
                G, corr, pr, coords = gram_metrics(V, world.Ms[f])
                # robustness: group by *sampled* attribute instead
                V2, _ = attribute_vectors(states[l], attrs[:, f], conf, world.N)
                _, corr_sampled, _, _ = gram_metrics(V2, world.Ms[f])
                entry.update(gram_corr=corr, gram_corr_sampled_labels=corr_sampled,
                             participation_ratio=pr, group_counts=counts)
                entry["_coords"] = coords
                entry["_gram"] = G
            layer_res["features"].append(entry)
        results["per_layer"].append(layer_res)

    # figures: one row per feature; per layer a PCA scatter + Gram/M heatmaps
    for l in range(n_layers):
        fig, axes = plt.subplots(world.F, 3, figsize=(11, 3.4 * world.F),
                                 squeeze=False)
        for f in range(world.F):
            e = results["per_layer"][l]["features"][f]
            ax_p, ax_g, ax_m = axes[f]
            if "_coords" in e:
                c = e["_coords"]
                ax_p.scatter(c[:, 0], c[:, 1], c=np.arange(world.N),
                             cmap="hsv", s=120, zorder=3)
                order = np.arange(world.N)
                ax_p.plot(c[order, 0], c[order, 1], "-", c="gray", lw=0.8)
                if world.m_kinds[f] == "ring":
                    ax_p.plot(c[[world.N - 1, 0], 0], c[[world.N - 1, 0], 1],
                              "-", c="gray", lw=0.8)
                for a in range(world.N):
                    ax_p.annotate(str(a), c[a, :2], fontsize=8,
                                  ha="center", va="center", zorder=4)
                ax_p.set_title(f"{e['kind']}  PR={e['participation_ratio']:.2f}")
                ax_p.set_aspect("equal")
                ax_g.imshow(e["_gram"], cmap="RdBu_r")
                ax_g.set_title(f"Gram (corr w/ M = {e['gram_corr']:.3f})")
            ax_m.imshow(double_center(world.Ms[f]), cmap="RdBu_r")
            ax_m.set_title("M (double-centered)")
        fig.suptitle(f"{run_dir.name} — layer {l} (0 = embeddings)")
        fig.tight_layout()
        fig.savefig(fig_dir / f"geometry_layer{l}.png", dpi=150)
        plt.close(fig)

    # feature-subspace angles at the last layer with full data
    if world.F > 1:
        last = results["per_layer"][-1]["features"]
        Vs = []
        for f in range(world.F):
            labels, conf = p[:, f].argmax(axis=1), p[:, f].max(axis=1)
            V, _ = attribute_vectors(states[-1], labels, conf, world.N)
            Vs.append(V)
        results["subspace_angles_deg"] = {
            f"{world.m_kinds[a]}-{world.m_kinds[b]}":
                [round(float(x), 1) for x in principal_angles(Vs[a], Vs[b])]
            for a in range(world.F) for b in range(a + 1, world.F)}

    results["behavioral_M_corr"] = behavioral_M(model, world, device)

    # strip numpy internals before saving
    for lr in results["per_layer"]:
        for e in lr["features"]:
            e.pop("_coords", None), e.pop("_gram", None)
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "per_layer"},
                     indent=2))
    for lr in results["per_layer"]:
        line = f"layer {lr['layer']}: "
        line += "  ".join(
            f"{e['kind']}: corr={e.get('gram_corr', float('nan')):.3f} "
            f"PR={e.get('participation_ratio', float('nan')):.2f}"
            for e in lr["features"])
        print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    main(ap.parse_args().run_dir)
