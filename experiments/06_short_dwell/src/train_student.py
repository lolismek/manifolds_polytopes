"""Train the exp06 ring student from scratch on the short-doc corpus.

exp03/04/05 recipe with only the batch SHAPE changed for 256-token docs:
512 docs x 256 ctx per step = the same 131k tokens/step as 128 x 1024, same
cosine-to-2812 schedule + floor-LR extension to 4218 (~553M tokens, ~6
epochs of the 92.2M-token training split), same 12-layer/768 Llama-style
decoder (~110M params, identical config for cross-experiment comparability).
Run data-parallel on 2 GPUs (each rank accumulates 2 x 128 micro-batches);
checkpoints/evals on rank 0.

Trains on ring chunks 000-008 x 8 shards (360,000 docs); heldout loss on
the first 256 docs of each shard's chunk009 (subset of the 625/shard eval
split, ~524k tokens — matches exp05's eval token count).

Usage (tigerfish, 2 free GPUs):
  CUDA_VISIBLE_DEVICES=a,b torchrun --nproc_per_node=2 train_student.py
Single-GPU fallback: python train_student.py [--resume ckpt_02500.pt]
"""

import argparse
import contextlib
import glob
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from transformers import LlamaConfig, LlamaForCausalLM

SEED = 0
CTX = 256
BATCH = 512                  # docs/step; x256 tokens = 131k tokens/step
MICRO = 128
STEPS = 2812                 # cosine schedule ends here (exp03/04/05-identical)
TOTAL_STEPS = 4218           # extension at MIN_LR
WARMUP = 100
PEAK_LR = 3e-4
MIN_LR = 3e-5
WEIGHT_DECAY = 0.1
CKPT_EVERY = 250
EVAL_EVERY = 250
EVAL_PER_SHARD = 256         # x8 shards = 2048 heldout docs


def load_corpus(root, vmap):
    files = [f for f in sorted(glob.glob(str(root / "ring_shard*_chunk*.npz")))
             if "chunk009" not in f]
    assert len(files) == 72, f"expected 72 training chunks, found {len(files)}"
    ids = np.concatenate([np.load(f)["ids"] for f in files])
    return torch.from_numpy(vmap[ids].astype(np.int32))


def save_ckpt(sd, path):
    """torch.save with verify-and-retry (the shared FS can drop a write)."""
    for attempt in range(6):     # FS blips can outlast a 30s window; be patient
        tmp = path.with_suffix(".tmp")
        try:
            torch.save(sd, tmp)
            torch.load(tmp, map_location="cpu")
            tmp.rename(path)
            return
        except Exception as e:
            print(f"ckpt save to {path.name} failed ({e}), attempt {attempt+1}",
                  flush=True)
            time.sleep(30)
    raise RuntimeError(f"could not save {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default=None,
                    help="checkpoint filename to resume from, e.g. ckpt_02500.pt "
                         "(model weights only; optimizer restarts fresh)")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}" if world > 1 else "cuda:0")
    assert BATCH % (world * MICRO) == 0
    accum = BATCH // (world * MICRO)
    main_rank = rank == 0

    root = Path(__file__).parent.parent / "results" / "corpus"
    out = Path(__file__).parent.parent / "results" / "students" / "ring"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    vmap = np.load(root / "vocab32k.npz")["map"]
    vocab_size = int(vmap.max()) + 1
    data = load_corpus(root, vmap)
    if main_rank:
        print(f"ring: {data.shape[0]} docs, vocab {vocab_size}, "
              f"world {world}, accum {accum}", flush=True)

    eval_files = sorted(glob.glob(str(root / "ring_shard*_chunk009.npz")))
    eval_ids = np.concatenate([np.load(f)["ids"][:EVAL_PER_SHARD]
                               for f in eval_files])
    eval_data = torch.from_numpy(vmap[eval_ids].astype(np.int32))

    cfg = LlamaConfig(
        vocab_size=vocab_size, hidden_size=768, intermediate_size=2048,
        num_hidden_layers=12, num_attention_heads=12, num_key_value_heads=12,
        max_position_embeddings=1024, rms_norm_eps=1e-5,
        tie_word_embeddings=True, attention_bias=False,
    )
    model = LlamaForCausalLM(cfg).to(dev)
    if main_rank:
        n_all = sum(p.numel() for p in model.parameters())
        n_emb = model.model.embed_tokens.weight.numel()
        print(f"params: {n_all/1e6:.1f}M total, "
              f"{(n_all-n_emb)/1e6:.1f}M non-embedding", flush=True)
        (out / "config.json").write_text(cfg.to_json_string())

    start = 0
    if args.resume:
        sd = torch.load(out / args.resume, map_location="cpu")
        model.load_state_dict({k: v.float() for k, v in sd.items()})
        start = int(args.resume.split("_")[1].split(".")[0])
        if main_rank:
            print(f"resumed from {args.resume} at step {start}", flush=True)
        rng = np.random.default_rng([SEED, start])
    else:
        rng = np.random.default_rng(SEED)

    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank])
    raw = model.module if world > 1 else model

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=PEAK_LR, betas=(0.9, 0.95))

    def lr_at(step):
        if step < WARMUP:
            return PEAK_LR * (step + 1) / WARMUP
        if step >= STEPS:            # extension phase: hold the floor
            return MIN_LR
        p = (step - WARMUP) / max(STEPS - WARMUP, 1)
        return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + math.cos(math.pi * p))

    def run_eval():
        model.eval()
        losses = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, eval_data.shape[0], MICRO):
                x = eval_data[i : i + MICRO].to(dev).long()
                losses.append(raw(input_ids=x, labels=x).loss.item())
        model.train()
        return float(np.mean(losses))

    # same rng on every rank -> same permutation; rank takes its slice of
    # each global 512-doc batch, so the union per step matches single-GPU
    order = rng.permutation(data.shape[0])
    pos = 0
    log = open(out / "train_log.jsonl",
               "a" if args.resume else "w") if main_rank else None
    model.train()
    t0, tok_count = time.time(), 0
    for step in range(start, TOTAL_STEPS):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        if pos + BATCH > data.shape[0]:
            order = rng.permutation(data.shape[0])
            pos = 0
        idx = order[pos + rank * BATCH // world :
                    pos + (rank + 1) * BATCH // world]
        pos += BATCH
        for m in range(accum):
            x = data[idx[m * MICRO : (m + 1) * MICRO]].to(dev).long()
            sync = contextlib.nullcontext() if (world == 1 or m == accum - 1) \
                else model.no_sync()
            with sync, torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=x, labels=x).loss / accum
                loss.backward()
            loss_acc += loss.item()
            tok_count += MICRO * CTX * world
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % 25 == 0 and main_rank:
            rate = tok_count / (time.time() - t0)
            print(f"step {step+1}/{TOTAL_STEPS} loss {loss_acc:.4f} "
                  f"lr {lr_at(step):.2e} ({rate/1e3:.0f}k tok/s)", flush=True)
            log.write(json.dumps({"step": step + 1, "loss": loss_acc,
                                  "lr": lr_at(step)}) + "\n")
            log.flush()
        if (step + 1) % EVAL_EVERY == 0 and main_rank:
            ev = run_eval()
            print(f"step {step+1}: heldout loss {ev:.4f}", flush=True)
            log.write(json.dumps({"step": step + 1, "heldout_loss": ev}) + "\n")
            log.flush()
        if ((step + 1) % CKPT_EVERY == 0 or step + 1 == TOTAL_STEPS) \
                and main_rank:
            sd = {k: v.bfloat16() for k, v in raw.state_dict().items()}
            save_ckpt(sd, out / f"ckpt_{step+1:05d}.pt")
        if world > 1 and (step + 1) % CKPT_EVERY == 0:
            dist.barrier()

    if main_rank:
        print(f"ring training done: final heldout loss {run_eval():.4f}",
              flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
