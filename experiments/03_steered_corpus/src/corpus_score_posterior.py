"""Ideal-reader ground truth for the eval subset of the ring corpus.

For each held-out document (first EVAL_PER_SHARD docs of the last ring chunk
per shard), replay the token sequence teacher-forced under:
  - each of the 8 static hypotheses (whole-sequence one-hot clamp at state k,
    the stage-4 instrument), giving per-token log-likelihoods ll (8, GEN_LEN);
  - the true generative clamp path (per-position targets: position s carries
    the clamp of the token it predicts), giving ll_true (GEN_LEN,) — this
    reproduces the generation-time computation exactly and sanity-checks the
    static-hypothesis approximation.

The ideal reader is then the HMM forward algorithm over the static-hypothesis
emissions with the known ring transition matrix (stay P_STAY, else uniform
neighbour): alpha_t(k) proportional to e_t(k) * sum_j T[j,k] alpha_{t-1}(j),
alpha_0 uniform. This is the operational ground truth the student is compared
against. Exact Bayesian filtering is intractable because emissions condition
on the whole past clamp path through the KV cache, not just the current state;
the mismatch is largest just after transitions and ll_true quantifies it.

Outputs (results/corpus/): eval_shard{g}.npz with
  doc_idx (n,), z (n, GEN_LEN) int8, ll (n, 8, GEN_LEN) float16,
  ll_true (n, GEN_LEN) float16, posterior (n, 8, GEN_LEN) float16

Usage (on tigerfish, after generation finishes):
  python corpus_score_posterior.py --shard 0 --device cuda:0
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from corpus_generate import (MODEL, SAE_RELEASE, SAE_ID, LAYER, CAST, K, MULT,
                             P_STAY, PROMPT_LEN, DOC_LEN, GEN_LEN)

EVAL_CHUNK = 9              # last ring chunk of each shard is held out
EVAL_PER_SHARD = 1250       # 4 shards x 1250 = 5000 eval docs
BATCH = 4


def ring_T():
    T = np.full((K, K), 0.0)
    for i in range(K):
        T[i, i] = P_STAY
        T[i, (i + 1) % K] = (1 - P_STAY) / 2
        T[i, (i - 1) % K] = (1 - P_STAY) / 2
    return T


def forward_algorithm(ll, T):
    """ll (8, GEN_LEN) log-emissions -> filtered posterior (8, GEN_LEN)."""
    post = np.empty_like(ll, dtype=np.float64)
    alpha = np.full(K, 1.0 / K)
    for t in range(ll.shape[1]):
        e = np.exp(ll[:, t] - ll[:, t].max())
        alpha = (T.T @ alpha) * e
        alpha /= alpha.sum()
        post[:, t] = alpha
    return post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(__file__).parent.parent / "results" / "corpus"
    dev = torch.device(args.device)
    d = np.load(out / f"ring_shard{args.shard}_chunk{EVAL_CHUNK:03d}.npz")
    ids = torch.from_numpy(d["ids"][:EVAL_PER_SHARD].astype(np.int64))
    z = d["z"][:EVAL_PER_SHARD]                     # (n, GEN_LEN)
    n = ids.shape[0]

    from sae_lens import SAE
    sae = SAE.from_pretrained(SAE_RELEASE, SAE_ID, device=str(dev))[0].to(torch.float32)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev)
    model.eval()
    mean_act = np.load(Path(__file__).parent.parent / "results" / "stage1"
                       / "latent_stats.npz")["mean_act"]
    cast_t = torch.tensor(CAST, device=dev)
    W_cast = sae.W_dec[cast_t]
    clamp_vals = torch.tensor([MULT * float(mean_act[c]) for c in CAST], device=dev)

    steer = {"t": None}                              # (B, P, 8) per-position target

    def hook(_, __, output):
        if steer["t"] is None:
            return
        is_tuple = isinstance(output, tuple)
        x = output[0] if is_tuple else output
        xf = x.float()
        B, P, dm = xf.shape
        a = sae.encode(xf.reshape(-1, dm))[:, cast_t].reshape(B, P, K)
        delta = ((steer["t"] - a).reshape(-1, K) @ W_cast).reshape(B, P, dm)
        x = (xf + delta).to(x.dtype)
        return (x,) + output[1:] if is_tuple else x

    model.model.layers[LAYER].register_forward_hook(hook)

    def score(chunk_ids, targets):
        """chunk_ids (B, DOC_LEN); targets (B, DOC_LEN, 8) -> ll (B, GEN_LEN)."""
        with torch.no_grad():
            steer["t"] = targets
            logits = model(chunk_ids.to(dev)).logits[:, :-1]
            steer["t"] = None
            ll = -F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                chunk_ids[:, 1:].reshape(-1).to(dev), reduction="none"
            ).reshape(chunk_ids.shape[0], -1)
        return ll[:, PROMPT_LEN:].cpu().numpy()      # predictions of generated tokens

    def onehot_grid(states):                         # (B, DOC_LEN) long -> (B, DOC_LEN, 8)
        t = torch.zeros(*states.shape, K, device=dev)
        t.scatter_(2, states.unsqueeze(-1).to(dev), clamp_vals[states.to(dev)].unsqueeze(-1))
        return t

    # per-position clamp path: position s carries the state of the token it
    # predicts — z[0] for BOS+opener (s <= PROMPT_LEN), z[s - PROMPT_LEN] after
    z_t = torch.from_numpy(z.astype(np.int64))
    pos_states = torch.cat([z_t[:, :1].expand(-1, PROMPT_LEN + 1),
                            z_t[:, 1:], z_t[:, -1:]], dim=1)   # (n, DOC_LEN)

    ll = np.empty((n, K, GEN_LEN), dtype=np.float16)
    ll_true = np.empty((n, GEN_LEN), dtype=np.float16)
    for b0 in range(0, n, BATCH):
        b1 = min(b0 + BATCH, n)
        chunk = ids[b0:b1]
        for k in range(K):
            const = torch.full((b1 - b0, DOC_LEN), k, dtype=torch.long)
            ll[b0:b1, k] = score(chunk, onehot_grid(const))
        ll_true[b0:b1] = score(chunk, onehot_grid(pos_states[b0:b1]))
        if (b0 // BATCH) % 25 == 0:
            print(f"shard {args.shard}: {b1}/{n} docs scored", flush=True)

    T = ring_T()
    posterior = np.empty((n, K, GEN_LEN), dtype=np.float16)
    for i in range(n):
        posterior[i] = forward_algorithm(ll[i].astype(np.float64), T)

    np.savez(out / f"eval_shard{args.shard}.npz",
             doc_idx=d["opener_idx"][:EVAL_PER_SHARD], z=z,
             ll=ll, ll_true=ll_true, posterior=posterior)
    acc = float((posterior.argmax(1) == z).mean())
    print(f"shard {args.shard} done: per-token filter accuracy {acc:.3f}", flush=True)


if __name__ == "__main__":
    main()
