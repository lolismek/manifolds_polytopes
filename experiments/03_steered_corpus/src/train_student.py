"""Train a student transformer from scratch on the steered (or control) corpus.

Llama-style decoder: 12 layers, width 768, 12 heads, SwiGLU 2048, context 1024,
tied embeddings over the compact 32k+<unk> vocab (vocab32k.npz) — ~85M
non-embedding params, ~110M total. One document per sequence (docs are exactly
1024 tokens). Token ids are remapped through the vocab map; out-of-vocab -> 0.

Ring student trains on ring chunks 000-008 (90k docs); control student on all
ctrl chunks (25k docs). Both run the same fixed number of optimizer steps
(default 2812 = 4 epochs of the ring corpus at batch 128), so the control sees
the same optimization length. Checkpoints (bf16 state dicts) every 250 steps
for geometry-over-training probing; held-out loss on ring chunk 009 docs.

Usage (on tigerfish):
  python train_student.py --corpus ring --device cuda:0
  python train_student.py --corpus ctrl --device cuda:1
"""

import argparse
import glob
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

SEED = 0
BATCH = 128
MICRO = 32
STEPS = 2812
WARMUP = 100
PEAK_LR = 3e-4
MIN_LR = 3e-5
WEIGHT_DECAY = 0.1
CKPT_EVERY = 250
EVAL_EVERY = 250
EVAL_DOCS = 512


def load_corpus(root, corpus, vmap):
    if corpus == "ring":
        files = [f for f in sorted(glob.glob(str(root / "ring_shard*_chunk*.npz")))
                 if "chunk009" not in f]
    else:
        files = sorted(glob.glob(str(root / "ctrl_shard*_chunk*.npz")))
    ids = np.concatenate([np.load(f)["ids"] for f in files])
    return torch.from_numpy(vmap[ids].astype(np.int32))


def save_ckpt(sd, path):
    """torch.save with verify-and-retry (the shared FS can drop a write)."""
    for attempt in range(3):
        tmp = path.with_suffix(".tmp")
        try:
            torch.save(sd, tmp)
            torch.load(tmp, map_location="cpu")
            tmp.rename(path)
            return
        except Exception as e:
            print(f"ckpt save to {path.name} failed ({e}), attempt {attempt+1}",
                  flush=True)
            time.sleep(10)
    raise RuntimeError(f"could not save {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["ring", "ctrl"], required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resume", default=None,
                    help="checkpoint filename to resume from, e.g. ckpt_02500.pt "
                         "(model weights only; optimizer restarts fresh)")
    args = ap.parse_args()

    root = Path(__file__).parent.parent / "results" / "corpus"
    out = Path(__file__).parent.parent / "results" / "students" / args.corpus
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    torch.manual_seed(SEED)

    vmap = np.load(root / "vocab32k.npz")["map"]
    vocab_size = int(vmap.max()) + 1
    data = load_corpus(root, args.corpus, vmap)
    print(f"{args.corpus}: {data.shape[0]} docs, vocab {vocab_size}", flush=True)

    eval_files = sorted(glob.glob(str(root / "ring_shard*_chunk009.npz")))
    eval_ids = np.concatenate([np.load(f)["ids"][:EVAL_DOCS // 4] for f in eval_files])
    eval_data = torch.from_numpy(vmap[eval_ids].astype(np.int32))

    cfg = LlamaConfig(
        vocab_size=vocab_size, hidden_size=768, intermediate_size=2048,
        num_hidden_layers=12, num_attention_heads=12, num_key_value_heads=12,
        max_position_embeddings=1024, rms_norm_eps=1e-5,
        tie_word_embeddings=True, attention_bias=False,
    )
    model = LlamaForCausalLM(cfg).to(dev)
    n_all = sum(p.numel() for p in model.parameters())
    n_emb = model.model.embed_tokens.weight.numel()
    print(f"params: {n_all/1e6:.1f}M total, {(n_all-n_emb)/1e6:.1f}M non-embedding",
          flush=True)
    (out / "config.json").write_text(cfg.to_json_string())

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=PEAK_LR, betas=(0.9, 0.95))

    def lr_at(step):
        if step < WARMUP:
            return PEAK_LR * (step + 1) / WARMUP
        p = (step - WARMUP) / max(STEPS - WARMUP, 1)
        return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + math.cos(math.pi * p))

    def run_eval():
        model.eval()
        losses = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, eval_data.shape[0], MICRO):
                x = eval_data[i : i + MICRO].to(dev).long()
                losses.append(model(input_ids=x, labels=x).loss.item())
        model.train()
        return float(np.mean(losses))

    start = 0
    if args.resume:
        sd = torch.load(out / args.resume, map_location="cpu")
        model.load_state_dict({k: v.float() for k, v in sd.items()})
        start = int(args.resume.split("_")[1].split(".")[0])
        print(f"resumed from {args.resume} at step {start}", flush=True)
        rng = np.random.default_rng([SEED, start])
    else:
        rng = np.random.default_rng(SEED)
    order = rng.permutation(data.shape[0])
    pos = 0
    log = open(out / "train_log.jsonl", "a" if args.resume else "w")
    model.train()
    t0, tok_count = time.time(), 0
    for step in range(start, STEPS):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(BATCH // MICRO):
            if pos + MICRO > data.shape[0]:
                order = rng.permutation(data.shape[0])
                pos = 0
            x = data[order[pos : pos + MICRO]].to(dev).long()
            pos += MICRO
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=x, labels=x).loss / (BATCH // MICRO)
            loss.backward()
            loss_acc += loss.item()
            tok_count += MICRO * 1024
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % 25 == 0:
            rate = tok_count / (time.time() - t0)
            print(f"step {step+1}/{STEPS} loss {loss_acc:.4f} "
                  f"lr {lr_at(step):.2e} ({rate/1e3:.0f}k tok/s)", flush=True)
            log.write(json.dumps({"step": step + 1, "loss": loss_acc,
                                  "lr": lr_at(step)}) + "\n")
            log.flush()
            t0, tok_count = time.time(), 0
        if (step + 1) % EVAL_EVERY == 0:
            ev = run_eval()
            print(f"step {step+1}: heldout loss {ev:.4f}", flush=True)
            log.write(json.dumps({"step": step + 1, "heldout_loss": ev}) + "\n")
            log.flush()
        if (step + 1) % CKPT_EVERY == 0 or step + 1 == STEPS:
            sd = {k: v.bfloat16() for k, v in model.state_dict().items()}
            save_ckpt(sd, out / f"ckpt_{step+1:05d}.pt")

    print(f"{args.corpus} training done: final heldout loss {run_eval():.4f}",
          flush=True)


if __name__ == "__main__":
    main()
