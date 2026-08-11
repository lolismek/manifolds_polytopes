"""Corpus generation for exp04: ring-HMM steered documents under clean-cache
steering.

Same corpus design as exp03 (100k docs of BOS + 12-token Pile opener + 1011
generated tokens; hidden path pre-sampled from the ring chain, p_stay 0.99)
with the exp04 physics: cast in margin-optimal ring order, 8x one-hot clamp
applied only at the position being sampled, KV cache always clean (clamped
pass -> crop -> clean pass per token, the sequence verified in
verify_identity.py). Opener prefill is clean; z[t] is the state that produced
generated token t (ids position 1 + PROMPT_LEN + t).

Openers use exp03's harvest (residue = shard, mod 8), so ring documents get
the identical openers as exp03's ring corpus, doc for doc. The unsteered
control corpus is NOT regenerated: with no clamp, clean-cache generation is
ordinary generation, so exp03's control corpus is reused as-is.

Lean loop (no stats): ~2 forwards/token, so expect roughly half of exp03's
4.4k tok/s/GPU at batch 224.

Usage (on tigerfish, one shard per GPU):
  python corpus_generate.py --shard i --n_shards 4 --device cuda:i \
      --n_docs 25000
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

from steering import Steerer, load_teacher, mean_act

RING = [1309, 2970, 14600, 10573, 6455, 15026, 3188, 5615]
K = len(RING)
MULT = 8.0
P_STAY = 0.99
PROMPT_LEN = 12
DOC_LEN = 1024
GEN_LEN = DOC_LEN - PROMPT_LEN - 1      # 1011 generated tokens
SEED = 0

OUT = Path(__file__).parent.parent / "results" / "corpus"


def harvest_openers(tokenizer, n, residue, mod):
    """First PROMPT_LEN tokens of every (residue mod mod)-th usable Pile doc
    (identical recipe to exp03 -> identical ring openers)."""
    ds = load_dataset("monology/pile-uncopyrighted", split="train",
                      streaming=True)
    bos = tokenizer.bos_token_id
    out, idx = [], 0
    for ex in ds:
        toks = tokenizer(ex["text"][:400], add_special_tokens=False)["input_ids"]
        if len(toks) >= PROMPT_LEN:
            if idx % mod == residue:
                out.append([bos] + toks[:PROMPT_LEN])
                if len(out) == n:
                    return torch.tensor(out)
            idx += 1


def sample_paths(rng, n, length):
    z = np.empty((n, length), dtype=np.int8)
    z[:, 0] = rng.integers(0, K, n)
    u = rng.random((n, length - 1))
    step = rng.integers(0, 2, (n, length - 1)).astype(np.int8) * 2 - 1
    for t in range(1, length):
        move = u[:, t - 1] >= P_STAY
        z[:, t] = np.where(move, (z[:, t - 1] + step[:, t - 1]) % K,
                           z[:, t - 1])
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n_shards", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_docs", type=int, required=True)
    ap.add_argument("--opener_mod", type=int, default=8)
    ap.add_argument("--batch", type=int, default=224)
    ap.add_argument("--chunk", type=int, default=2500)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)

    model, tokenizer, sae = load_teacher(dev)
    print(f"shard {args.shard}: harvesting {args.n_docs} openers "
          f"(residue {args.shard} mod {args.opener_mod})", flush=True)
    openers = harvest_openers(tokenizer, args.n_docs, args.shard,
                              args.opener_mod)

    steer = Steerer(model, sae, RING, dev)
    ma = mean_act()
    # precomputed clamp-row table: row s = one-hot clamp for ring state s
    R = torch.zeros(K, K, device=dev)
    for s in range(K):
        R[s, s] = MULT * float(ma[RING[s]])

    meta = {"tag": "ring", "shard": args.shard, "scheme": "clean_cache",
            "cast_ring_order": RING, "mult": MULT, "p_stay": P_STAY,
            "doc_len": DOC_LEN, "prompt_len": PROMPT_LEN, "gen_len": GEN_LEN,
            "opener_residue": args.shard, "opener_mod": args.opener_mod,
            "n_docs": args.n_docs, "seed": SEED,
            "note": "z[t] is the state clamped (single position, clean cache)"
                    " when sampling generated token t = ids position "
                    "1+PROMPT_LEN+t; opener prefill clean"}
    (OUT / f"ring_shard{args.shard}_meta.json").write_text(
        json.dumps(meta, indent=2))

    n_chunks = (args.n_docs + args.chunk - 1) // args.chunk
    samples_path = OUT / f"ring_shard{args.shard}_samples.jsonl"

    for c in range(n_chunks):
        fout = OUT / f"ring_shard{args.shard}_chunk{c:03d}.npz"
        if fout.exists():
            print(f"shard {args.shard}: chunk {c} exists, skipping", flush=True)
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

    print(f"shard {args.shard} finished: {args.n_docs} ring docs", flush=True)


if __name__ == "__main__":
    main()
