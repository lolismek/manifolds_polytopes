"""Corpus generation for exp07: the uniform-jump structural control.

exp06 recipe UNCHANGED (same cast, ring-ordered state labels, s = 1.5
multi-layer add on layers 12-23, clean-cache physics, Pile openers,
p_stay = 0.95, 256-token docs) except the transition on a move: instead of
+-1 on the ring, the state jumps uniformly to one of the OTHER 7 states.
Same dwell distribution (geometric, mean 20), no transition map to learn —
a student trained here has the same exposure to the 8 flavors and the same
incentive to TRACK the current state, but no wiring. Ring signatures that
survive in the exp06 student and vanish here are attributable to the map.

Seed kept at 6 and the stay/move uniforms are drawn BEFORE the jump
destinations (same order as exp06's sample_paths), so every doc's switch
TIMES are token-for-token identical to the ring corpus; only the
destinations differ. Openers are deterministic per shard-residue, so they
match too.

Note: this file shadows exp05's corpus_generate by name; the sys.path
insert below makes `import corpus_generate` resolve to exp05's module
(this file only ever runs as __main__).

Usage (seahorse, one shard per free GPU):
  python corpus_generate.py --shard i --n_shards 8 --device cuda:d \
      --n_docs 50000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EXP05_SRC = Path(__file__).parents[2] / "05_multilayer_add" / "src"
sys.path.insert(0, str(EXP05_SRC))
from addsteer import AddSteerer, load_teacher, mean_act  # noqa: E402
from corpus_generate import (K, PROMPT_LEN, RING,  # noqa: E402
                             harvest_openers)

MULT = 1.5
P_STAY = 0.95
DOC_LEN = 256
GEN_LEN = DOC_LEN - PROMPT_LEN - 1      # 243 generated tokens
SEED = 6

OUT = Path(__file__).parent.parent / "results" / "corpus"


def sample_paths(rng, n, length):
    """Uniform-jump chain: stay w.p. P_STAY, else jump to one of the other
    7 states uniformly. Draw order (z0, u, jump) matches exp06's
    (z0, u, step) so the switch times are identical to the ring corpus
    under the same (SEED, shard, chunk) rng key."""
    z = np.empty((n, length), dtype=np.int8)
    z[:, 0] = rng.integers(0, K, n)
    u = rng.random((n, length - 1))
    jump = rng.integers(1, K, (n, length - 1)).astype(np.int8)
    for t in range(1, length):
        move = u[:, t - 1] >= P_STAY
        z[:, t] = np.where(move, (z[:, t - 1] + jump[:, t - 1]) % K,
                           z[:, t - 1])
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n_shards", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_docs", type=int, required=True)
    ap.add_argument("--batch", type=int, default=768)
    ap.add_argument("--chunk", type=int, default=5000)
    ap.add_argument("--tag", default="uniform")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)

    model, tokenizer, sae = load_teacher(dev)
    print(f"shard {args.shard}: harvesting {args.n_docs} openers "
          f"(residue {args.shard} mod {args.n_shards})", flush=True)
    openers = harvest_openers(tokenizer, args.n_docs, args.shard,
                              args.n_shards)

    steer = AddSteerer(model, sae, RING, dev)
    ma = mean_act()
    # add-row table: row s = additive coefficient for state s
    R = torch.zeros(K, K, device=dev)
    for s in range(K):
        R[s, s] = MULT * float(ma[RING[s]])

    meta = {"tag": args.tag, "shard": args.shard,
            "scheme": "clean_cache_multilayer_add", "window": [12, 23],
            "cast_ring_order": RING, "mult": MULT, "p_stay": P_STAY,
            "transitions": "uniform over the 7 other states",
            "doc_len": DOC_LEN, "prompt_len": PROMPT_LEN, "gen_len": GEN_LEN,
            "opener_residue": args.shard, "opener_mod": args.n_shards,
            "n_docs": args.n_docs, "seed": SEED,
            "note": "z[t] is the state added (all window layers, single "
                    "position, clean cache) when sampling generated token "
                    "t = ids position 1+PROMPT_LEN+t; opener prefill clean; "
                    "switch times identical to exp06 ring corpus (same seed, "
                    "same draw order), only jump destinations differ"}
    (OUT / f"{args.tag}_shard{args.shard}_meta.json").write_text(
        json.dumps(meta, indent=2))

    n_chunks = (args.n_docs + args.chunk - 1) // args.chunk
    samples_path = OUT / f"{args.tag}_shard{args.shard}_samples.jsonl"

    for c in range(n_chunks):
        fout = OUT / f"{args.tag}_shard{args.shard}_chunk{c:03d}.npz"
        if fout.exists():
            print(f"shard {args.shard}: chunk {c} exists, skipping",
                  flush=True)
            continue
        lo, hi = c * args.chunk, min((c + 1) * args.chunk, args.n_docs)
        n_c = hi - lo
        rng = np.random.default_rng([SEED, args.shard, c])
        torch.manual_seed(SEED * 1_000_000 + args.shard * 10_000 + c)
        z = sample_paths(rng, n_c, GEN_LEN)
        ids_out = np.empty((n_c, DOC_LEN), dtype=np.int32)
        t0 = time.time()

        with torch.no_grad():
            for b0 in range(0, n_c, args.batch):
                b1 = min(b0 + args.batch, n_c)
                prom = openers[lo + b0 : lo + b1].to(dev)
                zb = torch.from_numpy(z[b0:b1]).long().to(dev)
                steer.target = None
                o = model(prom[:, :-1], use_cache=True)
                past = o.past_key_values
                base = PROMPT_LEN
                cur = prom[:, -1:]
                gen = []
                for t in range(GEN_LEN):
                    steer.target = R[zb[:, t]]
                    o = model(cur, past_key_values=past, use_cache=True)
                    past = o.past_key_values
                    nxt = torch.multinomial(
                        F.softmax(o.logits[:, -1].float(), -1), 1)
                    past.crop(base + t)
                    steer.target = None
                    o = model(cur, past_key_values=past, use_cache=True)
                    past = o.past_key_values
                    gen.append(nxt)
                    cur = nxt
                steer.target = None
                seq = torch.cat([prom.cpu(), torch.cat(gen, 1).cpu()], 1)
                ids_out[b0:b1] = seq.numpy()
                rate = (b1 - b0) * GEN_LEN / max(time.time() - t0, 1e-9)
                t0 = time.time()
                print(f"shard {args.shard}: {lo + b1}/{args.n_docs} docs "
                      f"({rate:.0f} tok/s)", flush=True)

        np.savez(fout, ids=ids_out, z=z,
                 opener_idx=np.arange(lo, hi, dtype=np.int32))
        if c == 0:
            with open(samples_path, "w") as f:
                for i in range(min(2, n_c)):
                    trans = (np.diff(z[i]) != 0).nonzero()[0] + 1
                    f.write(json.dumps({
                        "doc": i, "z0": int(z[i, 0]),
                        "transitions": trans.tolist(),
                        "states": z[i][np.r_[0, trans]].tolist(),
                        "text": tokenizer.decode(ids_out[i, 1:])}) + "\n")

    print(f"shard {args.shard} finished: {args.n_docs} docs", flush=True)


if __name__ == "__main__":
    main()
