"""Step 0: exp04's generator/scorer identity check under the multi-layer
additive hook (same structure as exp04 verify_identity.py; the property must
be re-verified because the hook changed).

Usage: python verify_identity.py --device cuda:0 --mult 2
"""

import argparse
import json
from pathlib import Path

import torch

from addsteer import (AddSteerer, cast, generate_clean_cache, get_prompts,
                      load_teacher, mean_act, score_emissions)

N_DOCS = 8
GEN_LEN = 128
SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mult", type=float, default=2.0)
    args = ap.parse_args()
    dev = torch.device(args.device)

    out = Path(__file__).parent.parent / "results" / "verify"
    out.mkdir(parents=True, exist_ok=True)

    model, tok, sae = load_teacher(dev)
    pool = cast()
    steer = AddSteerer(model, sae, pool, dev)
    ma = mean_act()

    lat_true, lat_d1, lat_d2 = pool[0], pool[1], pool[2]
    val = lambda j: args.mult * float(ma[j])

    openers = get_prompts(tok, N_DOCS)
    torch.manual_seed(SEED)
    targets = steer.rows([lat_true] * N_DOCS,
                         torch.tensor([val(lat_true)] * N_DOCS, device=dev))
    ids, stats = generate_clean_cache(model, steer, openers, targets, GEN_LEN,
                                      record_logp=True)
    print(f"generated {N_DOCS} docs, latent {lat_true} @ {args.mult}x window; "
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

    rep = {
        "latents": [lat_true, lat_d1, lat_d2], "mult": args.mult,
        "A_same_shape_max_absdiff": float(dA.max()),
        "A_same_shape_median_absdiff": float(dA.median()),
        "B_expanded_max_absdiff": float(dB.max()),
        "B_expanded_median_absdiff": float(dB.median()),
        "margin_true_vs_d1_nats_per_tok": float((logp3[:, 0] - logp3[:, 1]).mean()),
        "margin_true_vs_d2_nats_per_tok": float((logp3[:, 0] - logp3[:, 2]).mean()),
        "evidence_vs_clean_nats_per_tok": float((logp3[:, 0] - logp_clean).mean()),
        "gen_kl_per_tok": float(stats["kl"].mean()),
        "pass_A": bool(dA.max() < 2e-2), "pass_B": bool(dB.max() < 5e-2),
    }
    (out / "identity_check.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2), flush=True)


if __name__ == "__main__":
    main()
