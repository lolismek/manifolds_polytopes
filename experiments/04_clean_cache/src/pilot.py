"""Step 5 pilot: ring-dynamic docs at candidate operating points, scored with
the exact reader.

The cast is placed on the ring in the order that maximizes the minimum
neighbor margin (brute-forced over all orders from the step-4 margin matrix;
symmetrized min-direction margins): min neighbor margin 0.093 vs 0.081 global.

Per doc: sample a state path from the ring HMM (uniform start; stay w.p.
p_stay, else hop to a uniform neighbor), generate with the clean-cache
generator using per-token clamp targets, then (--phase score) teacher-force
every doc under all 8 cast hypotheses to get the exact emission table the
forward algorithm needs.

Sharded over GPUs by global doc index:
  python pilot.py --phase gen   --shard i --n_shards 4 --device cuda:i
  python pilot.py --phase score --shard i --n_shards 4 --device cuda:i
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from steering import (Steerer, generate_clean_cache, get_prompts,
                      load_teacher, mean_act, score_emissions)

RING = [1309, 2970, 14600, 10573, 6455, 15026, 3188, 5615]
P_STAYS = [0.975, 0.985, 0.99]
N_DOCS = 24                 # per operating point
GEN_LEN = 1024
MULT = 8.0
N_PROMPTS = 24
SCORE_BATCH = 4
SEED = 0

OUT = Path(__file__).parent.parent / "results" / "pilot"


def sample_path(rng, p_stay, n):
    z = np.empty(n, dtype=np.int64)
    z[0] = rng.integers(8)
    stay = rng.random(n - 1) < p_stay
    hop = rng.integers(2, size=n - 1) * 2 - 1        # -1 or +1 on the ring
    for t in range(1, n):
        z[t] = z[t - 1] if stay[t - 1] else (z[t - 1] + hop[t - 1]) % 8
    return z


def doc_table():
    """Global doc list: (g, p_stay). g indexes the path RNG seed."""
    return [(k * N_DOCS + d, p) for k, p in enumerate(P_STAYS)
            for d in range(N_DOCS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["gen", "score"], required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n_shards", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)

    model, tok, sae = load_teacher(dev)
    steer = Steerer(model, sae, RING, dev)
    ma = mean_act()
    vals = torch.tensor([MULT * float(ma[j]) for j in RING], device=dev)

    docs = doc_table()[args.shard :: args.n_shards]
    print(f"shard {args.shard}: {len(docs)} docs", flush=True)

    if args.phase == "gen":
        prompts = get_prompts(tok, N_PROMPTS)
        B = len(docs)
        openers = prompts[torch.tensor([g % N_PROMPTS for g, _ in docs])]
        states = np.stack([
            sample_path(np.random.default_rng(SEED + 300_000 + g), p, GEN_LEN)
            for g, p in docs])
        st = torch.tensor(states, device=dev)               # (B, GEN_LEN)
        targets_all = torch.zeros(GEN_LEN, B, 8, device=dev)
        targets_all.scatter_(2, st.T.unsqueeze(2),
                             vals[st.T].unsqueeze(2))
        torch.manual_seed(SEED + 400_000 + args.shard)
        ids, stats = generate_clean_cache(model, steer, openers,
                                          lambda t: targets_all[t], GEN_LEN)
        np.savez(OUT / f"pilot_gen_shard{args.shard}.npz",
                 g=np.array([g for g, _ in docs], dtype=np.int32),
                 p_stay=np.array([p for _, p in docs], dtype=np.float32),
                 states=states.astype(np.int8),
                 ids=ids.numpy().astype(np.int32),
                 **{k: v.numpy() for k, v in stats.items()})
        with open(OUT / f"pilot_texts_shard{args.shard}.jsonl", "w") as f:
            for r in range(0, B, 8):
                f.write(json.dumps({"g": docs[r][0], "p_stay": docs[r][1],
                                    "text": tok.decode(ids[r, 1:])}) + "\n")
        print(f"shard {args.shard} gen done", flush=True)

    else:
        d = np.load(OUT / f"pilot_gen_shard{args.shard}.npz")
        ids = torch.tensor(d["ids"], dtype=torch.long)
        hyp_rows = steer.rows(RING, vals)
        prompt_len = ids.shape[1] - GEN_LEN
        lps, lcs = [], []
        for b0 in range(0, ids.shape[0], SCORE_BATCH):
            lp, lc = score_emissions(model, steer, ids[b0:b0 + SCORE_BATCH],
                                     hyp_rows, prompt_len)
            lps.append(lp.numpy())
            lcs.append(lc.numpy())
            print(f"shard {args.shard}: {b0 + lp.shape[0]}/{ids.shape[0]} "
                  "docs scored", flush=True)
        np.savez(OUT / f"pilot_score_shard{args.shard}.npz",
                 g=d["g"], p_stay=d["p_stay"], states=d["states"],
                 logp=np.concatenate(lps).astype(np.float32),
                 logp_clean=np.concatenate(lcs).astype(np.float32),
                 ring=np.array(RING, dtype=np.int32))
        print(f"shard {args.shard} score done", flush=True)


if __name__ == "__main__":
    main()
