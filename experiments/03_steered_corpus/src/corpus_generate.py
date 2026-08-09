"""Corpus generation: ring-HMM steered documents (exp03 variant A).

Each document is DOC_LEN tokens: BOS + a 12-token natural opener (a fresh Pile
document per corpus document) + GEN_LEN generated tokens. Before generation, a
hidden state path z_1..z_GEN_LEN is pre-sampled from the ring chain over the 8
cast states (stay w.p. P_STAY, else step to a uniform ring neighbour). During
generation, token t is produced with state z_t's latent clamped one-hot at
MULT x its mean firing activation and the other 7 cast latents ablated to 0 —
the intervention validated in stage 4. The path never looks at the text.

A custom decode loop (KV cache) is used because each document in a batch is in
its own state at each step, so the steering target is a per-document row that
updates every token. The opener prefill is clamped with z_1 (whole-sequence
clamping, matching the stage-4 calibration).

--control generates the unsteered control corpus with the same pipeline
(paths are still sampled and saved, but no clamp is applied).

Output (results/corpus/): {tag}_shard{g}_chunk{c:03d}.npz with
  ids (n, DOC_LEN) int32, z (n, GEN_LEN) int8, opener_idx (n,) int32
plus {tag}_shard{g}_meta.json and a few sample texts per shard. Chunks are
seeded independently, so a killed run resumes by skipping existing files.

Usage (on tigerfish):
  python corpus_generate.py --shard 0 --n_shards 4 --device cuda:0 \
      --n_docs 25000 --tag ring --opener_residue 0
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "google/gemma-2-2b"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_12/width_16k/canonical"
LAYER = 12

CAST = [368, 1220, 2404, 2970, 6172, 10615, 10621, 13931]
K = len(CAST)
MULT = 5.0
P_STAY = 0.99
PROMPT_LEN = 12
DOC_LEN = 1024
GEN_LEN = DOC_LEN - PROMPT_LEN - 1      # 1011 generated tokens
SEED = 0


def harvest_openers(tokenizer, n, residue, mod):
    """First PROMPT_LEN tokens of every (residue mod mod)-th usable Pile doc."""
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
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
        z[:, t] = np.where(move, (z[:, t - 1] + step[:, t - 1]) % K, z[:, t - 1])
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n_shards", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_docs", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--opener_residue", type=int, required=True)
    ap.add_argument("--opener_mod", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=2500)
    ap.add_argument("--control", action="store_true")
    args = ap.parse_args()

    out = Path(__file__).parent.parent / "results" / "corpus"
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    print(f"shard {args.shard}: harvesting {args.n_docs} openers "
          f"(residue {args.opener_residue} mod {args.opener_mod})", flush=True)
    openers = harvest_openers(tokenizer, args.n_docs,
                              args.opener_residue, args.opener_mod)

    from sae_lens import SAE
    sae = SAE.from_pretrained(SAE_RELEASE, SAE_ID, device=str(dev))[0].to(torch.float32)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev)
    model.eval()

    mean_act = np.load(Path(__file__).parent.parent / "results" / "stage1"
                       / "latent_stats.npz")["mean_act"]
    cast_t = torch.tensor(CAST, device=dev)
    W_cast = sae.W_dec[cast_t]                          # (8, d)
    clamp_vals = torch.tensor([MULT * float(mean_act[c]) for c in CAST],
                              device=dev)               # (8,)

    steer = {"t": None}                                 # (B, 8) target or None

    def hook(_, __, output):
        if steer["t"] is None:
            return
        is_tuple = isinstance(output, tuple)
        x = output[0] if is_tuple else output
        xf = x.float()
        B, P, d = xf.shape
        a = sae.encode(xf.reshape(-1, d))[:, cast_t].reshape(B, P, K)
        delta = (steer["t"].unsqueeze(1) - a) @ W_cast
        x = (xf + delta).to(x.dtype)
        return (x,) + output[1:] if is_tuple else x

    model.model.layers[LAYER].register_forward_hook(hook)

    def onehot_rows(states):                            # (B,) long -> (B, 8)
        t = torch.zeros(len(states), K, device=dev)
        t[torch.arange(len(states)), states] = clamp_vals[states]
        return t

    meta = {"tag": args.tag, "shard": args.shard, "control": args.control,
            "cast": CAST, "mult": MULT, "p_stay": P_STAY,
            "doc_len": DOC_LEN, "prompt_len": PROMPT_LEN, "gen_len": GEN_LEN,
            "opener_residue": args.opener_residue, "opener_mod": args.opener_mod,
            "n_docs": args.n_docs, "seed": SEED,
            "note": "z aligned to generated tokens (positions 1+PROMPT_LEN..); "
                    "opener prefill clamped with z[0]; control ignores z"}
    (out / f"{args.tag}_shard{args.shard}_meta.json").write_text(
        json.dumps(meta, indent=2))

    n_chunks = (args.n_docs + args.chunk - 1) // args.chunk
    samples_path = out / f"{args.tag}_shard{args.shard}_samples.jsonl"

    for c in range(n_chunks):
        fout = out / f"{args.tag}_shard{args.shard}_chunk{c:03d}.npz"
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
                zb = torch.from_numpy(z[b0:b1]).long().to(dev)   # (B, GEN_LEN)
                if not args.control:
                    steer["t"] = onehot_rows(zb[:, 0])
                o = model(prom, use_cache=True)
                past = o.past_key_values
                cur = torch.multinomial(F.softmax(o.logits[:, -1].float(), -1), 1)
                gen = [cur]
                for t in range(1, GEN_LEN):
                    if not args.control:
                        steer["t"] = onehot_rows(zb[:, t])
                    o = model(cur, past_key_values=past, use_cache=True)
                    past = o.past_key_values
                    cur = torch.multinomial(F.softmax(o.logits[:, -1].float(), -1), 1)
                    gen.append(cur)
                steer["t"] = None
                seq = torch.cat([prom.cpu(), torch.cat(gen, 1).cpu()], 1)
                ids_out[b0:b1] = seq.numpy()
                done = lo + b1
                rate = (b1 - b0) * GEN_LEN / max(time.time() - t0, 1e-9)
                t0 = time.time()
                print(f"shard {args.shard}: {done}/{args.n_docs} docs "
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

    print(f"shard {args.shard} finished: {args.n_docs} docs "
          f"({'control' if args.control else 'ring'})", flush=True)


if __name__ == "__main__":
    main()
