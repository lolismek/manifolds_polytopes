"""Stage 3 of the latent casting funnel: generation audition.

First stage that actually generates steered text. For every latent in the
audition pool (the mutually-clean independent set from stage 2) and every
strength s in MULTS, generate N_SAMPLES continuations of shared pile prompts
under the one-hot intervention: active latent clamped to s * mean firing
activation, all other pool latents ablated to 0. Then score every generated
continuation with the unsteered teacher (per-token NLL -> perplexity).

A clean arm (no hook) with the same prompts gives the fluency baseline and the
mini-M chance-level check.

Sharded by latent across GPUs: --shard i --n_shards 4. Shard 0 also generates
the clean arm and writes the audition pool file.

Outputs (results/stage3/):
  audition_pool.json      deterministic greedy independent set from stage-2 flags
  gen_shard{i}.npz        token ids (n, GEN_LEN), latent id, strength, nll per seq
  texts_shard{i}.jsonl    decoded samples (2 per cell) for the casting report

Usage (on tigerfish, one process per GPU):
  python stage3_audition.py --shard 0 --n_shards 4 --device cuda:0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "google/gemma-2-2b"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_12/width_16k/canonical"
LAYER = 12

MULTS = [1.0, 2.0, 3.0, 5.0]
N_SAMPLES = 20          # per (latent, strength) cell
N_CLEAN = 40            # clean-arm samples (shard 0 only)
PROMPT_LEN = 12
N_PROMPTS = 20
GEN_LEN = 300
SEED = 0


def audition_pool(res2: Path) -> list[int]:
    """Deterministic greedy max independent set over the stage-2 flag graph."""
    rep = json.loads((res2 / "coact_report.json").read_text())
    causal = json.loads((res2.parent / "stage2_5" / "causal_pool.json").read_text())["pool"]
    ids = sorted(p["latent"] for p in causal)
    adj = {i: set() for i in ids}
    for p in rep["flagged_pairs"]:
        adj[p["i"]].add(p["j"])
        adj[p["j"]].add(p["i"])
    chosen = []
    for v in sorted(adj, key=lambda k: (len(adj[k]), k)):
        if all(v not in adj[c] for c in chosen):
            chosen.append(v)
    return sorted(chosen)


def get_prompts(tokenizer):
    """First PROMPT_LEN tokens of the first N_PROMPTS pile documents (shared everywhere)."""
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    bos = tokenizer.bos_token_id
    prompts = []
    for ex in ds:
        toks = tokenizer(ex["text"], add_special_tokens=False)["input_ids"]
        if len(toks) >= PROMPT_LEN:
            prompts.append([bos] + toks[:PROMPT_LEN])
        if len(prompts) == N_PROMPTS:
            return torch.tensor(prompts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n_shards", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--latents", default=None,
                    help="comma-separated latent ids; overrides pool sharding (still sharded)")
    ap.add_argument("--mults", default=None, help="comma-separated strength multipliers")
    ap.add_argument("--n_samples", type=int, default=N_SAMPLES)
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. 'b'")
    args = ap.parse_args()

    mults = [float(m) for m in args.mults.split(",")] if args.mults else MULTS
    n_samples = args.n_samples

    root = Path(__file__).parent.parent / "results"
    out = root / "stage3"
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)

    from sae_lens import SAE

    sae = SAE.from_pretrained(SAE_RELEASE, SAE_ID, device=str(dev))[0]
    sae = sae.to(torch.float32)

    pool = audition_pool(root / "stage2")
    if args.shard == 0 and not args.tag:
        (out / "audition_pool.json").write_text(json.dumps({"pool": pool}, indent=2))
    subset = [int(x) for x in args.latents.split(",")] if args.latents else pool
    mine = subset[args.shard :: args.n_shards]
    print(f"shard {args.shard}: {len(mine)} of {len(subset)} latents", flush=True)

    stats = np.load(root / "stage1" / "latent_stats.npz")
    mean_act = stats["mean_act"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev)
    model.eval()

    pool_t = torch.tensor(pool, device=dev)
    W_pool = sae.W_dec[pool_t]                     # (n_pool, d)
    pos_in_pool = {j: k for k, j in enumerate(pool)}

    # one-hot steering hook: clamp active latent, ablate the rest of the pool
    steer = {"t": None}                            # target vector over pool, or None

    def hook(_, __, output):
        if steer["t"] is None:
            return
        is_tuple = isinstance(output, tuple)
        x = output[0] if is_tuple else output
        xf = x.float()
        a = sae.encode(xf.reshape(-1, xf.shape[-1]))[:, pool_t]     # (N, n_pool)
        delta = ((steer["t"] - a) @ W_pool).reshape(x.shape)
        x = (xf + delta).to(x.dtype)
        return (x,) + output[1:] if is_tuple else x

    model.model.layers[LAYER].register_forward_hook(hook)

    prompts = get_prompts(tokenizer)

    def generate(n, seed):
        torch.manual_seed(seed)
        idx = torch.arange(n) % N_PROMPTS
        ids = prompts[idx].to(dev)
        with torch.no_grad():
            gen = model.generate(ids, do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                 max_new_tokens=GEN_LEN, min_new_tokens=GEN_LEN,
                                 pad_token_id=tokenizer.eos_token_id)
        return gen[:, ids.shape[1]:].cpu()          # continuations only

    def clean_nll(cont_ids):
        """Per-seq mean NLL of continuations under the unsteered teacher (hook off)."""
        steer["t"] = None
        torch.cuda.empty_cache()
        nlls = []
        with torch.no_grad():
            for i in range(0, cont_ids.shape[0], 4):
                chunk = cont_ids[i : i + 4].to(dev)
                bos = torch.full((chunk.shape[0], 1), tokenizer.bos_token_id, device=dev)
                seq = torch.cat([bos, chunk], 1)
                logits = model(seq).logits[:, :-1]
                nll = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(),
                    seq[:, 1:].reshape(-1), reduction="none")
                nlls.append(nll.reshape(chunk.shape[0], -1).mean(-1).cpu())
        return torch.cat(nlls)

    all_ids, all_lat, all_mult, texts = [], [], [], []

    if args.shard == 0:
        cont = generate(N_CLEAN, seed=SEED)
        all_ids.append(cont); all_lat += [-1] * N_CLEAN; all_mult += [0.0] * N_CLEAN
        texts += [{"latent": -1, "mult": 0.0, "text": tokenizer.decode(c)} for c in cont[:2]]
        print("clean arm done", flush=True)

    for li, j in enumerate(mine):
        for mult in mults:
            t = torch.zeros(len(pool), device=dev)
            t[pos_in_pool[j]] = mult * float(mean_act[j])
            steer["t"] = t
            cont = generate(n_samples, seed=SEED + 1000 * j + int(mult * 10))
            steer["t"] = None
            all_ids.append(cont)
            all_lat += [j] * n_samples
            all_mult += [mult] * n_samples
            texts += [{"latent": j, "mult": mult, "text": tokenizer.decode(c)} for c in cont[:2]]
        print(f"shard {args.shard}: latent {j} done ({li + 1}/{len(mine)})", flush=True)

    cont_all = torch.cat(all_ids)
    # save generations before scoring so an OOM in scoring can't lose them
    np.savez(out / f"gen{args.tag}_shard{args.shard}_raw.npz",
             ids=cont_all.numpy().astype(np.int32),
             latent=np.array(all_lat, dtype=np.int32),
             mult=np.array(all_mult, dtype=np.float32))
    nll = clean_nll(cont_all)

    np.savez(out / f"gen{args.tag}_shard{args.shard}.npz",
             ids=cont_all.numpy().astype(np.int32),
             latent=np.array(all_lat, dtype=np.int32),
             mult=np.array(all_mult, dtype=np.float32),
             nll=nll.numpy().astype(np.float32))
    with open(out / f"texts{args.tag}_shard{args.shard}.jsonl", "w") as f:
        for row in texts:
            f.write(json.dumps(row) + "\n")
    print(f"shard {args.shard} finished: {cont_all.shape[0]} seqs, "
          f"clean-model NLL median {float(nll.median()):.3f}", flush=True)


if __name__ == "__main__":
    main()
