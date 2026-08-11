"""Step 0: verify the exact scorer reproduces the generator's distributions.

Generates a few static-state docs with the clean-cache generator, logging the
clamped log-prob of every sampled token. Then rescores the same token ids with
score_emissions and compares:

  A. H=1 (true hypothesis only, same batch shape as generation): op sequences
     align, so the emission log-probs must match the generation-time sampling
     log-probs essentially bit-for-bit (bf16).
  B. H=3 (true + 2 distractors, expanded batch): same numbers modulo bf16
     matmul tiling differences from the different batch shape — must agree to
     small tolerance.

Also prints a first physics number: the mean per-token log-likelihood margin
of the true latent over the distractors at the chosen strength.

Usage (tigerfish): python verify_identity.py --device cuda:0 --mult 8
"""

import argparse
import json
from pathlib import Path

import torch

from steering import (Steerer, audition_pool, generate_clean_cache,
                      get_prompts, load_teacher, mean_act, score_emissions)

N_DOCS = 8
GEN_LEN = 128
SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mult", type=float, default=8.0)
    args = ap.parse_args()
    dev = torch.device(args.device)

    out = Path(__file__).parent.parent / "results" / "verify"
    out.mkdir(parents=True, exist_ok=True)

    model, tok, sae = load_teacher(dev)
    pool = audition_pool()
    steer = Steerer(model, sae, pool, dev)
    ma = mean_act()

    lat_true, lat_d1, lat_d2 = pool[0], pool[1], pool[2]
    val = lambda j: args.mult * float(ma[j])

    openers = get_prompts(tok, N_DOCS)
    torch.manual_seed(SEED)
    targets = steer.rows([lat_true] * N_DOCS,
                         torch.tensor([val(lat_true)] * N_DOCS, device=dev))
    ids, stats = generate_clean_cache(model, steer, openers, targets, GEN_LEN,
                                      record_logp=True)
    print(f"generated {N_DOCS} docs, latent {lat_true} @ {args.mult}x; "
          f"mean KL/token {stats['kl'].mean():.4f}", flush=True)

    P = openers.shape[1]
    hyp1 = steer.rows([lat_true], torch.tensor([val(lat_true)], device=dev))
    logp1, _ = score_emissions(model, steer, ids, hyp1, P)
    dA = (logp1[:, 0, :] - stats["logp_gen"]).abs()

    hyp3 = steer.rows([lat_true, lat_d1, lat_d2],
                      torch.tensor([val(lat_true), val(lat_d1), val(lat_d2)],
                                   device=dev))
    logp3, logp_clean = score_emissions(model, steer, ids, hyp3, P)
    dB = (logp3[:, 0, :] - stats["logp_gen"]).abs()

    margin1 = (logp3[:, 0] - logp3[:, 1]).mean()
    margin2 = (logp3[:, 0] - logp3[:, 2]).mean()
    ev = (logp3[:, 0] - logp_clean).mean()

    rep = {
        "latents": [lat_true, lat_d1, lat_d2], "mult": args.mult,
        "A_same_shape_max_absdiff": float(dA.max()),
        "A_same_shape_median_absdiff": float(dA.median()),
        "B_expanded_max_absdiff": float(dB.max()),
        "B_expanded_median_absdiff": float(dB.median()),
        "margin_true_vs_d1_nats_per_tok": float(margin1),
        "margin_true_vs_d2_nats_per_tok": float(margin2),
        "evidence_vs_clean_nats_per_tok": float(ev),
        "gen_kl_per_tok": float(stats["kl"].mean()),
        "pass_A": bool(dA.max() < 2e-2), "pass_B": bool(dB.max() < 5e-2),
    }
    (out / "identity_check.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2), flush=True)


if __name__ == "__main__":
    main()
