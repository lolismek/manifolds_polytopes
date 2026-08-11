"""Train the exp04 ring student from scratch on the clean-cache corpus.

Identical recipe to exp03 through step 2812 (same architecture, batch,
cosine schedule, checkpoint cadence — every exp03 checkpoint has an exp04
twin), then EXTENDED to step 4218 (~6 ring epochs) at the floor learning
rate, same cadence, since exp03's student was still improving when its
schedule ended. The control student is reused from exp03 (unsteered
generation is identical under both schemes), so only --corpus ring exists.

Llama-style decoder: 12 layers, width 768, 12 heads, SwiGLU 2048, context
1024, tied embeddings over the compact 32k+<unk> vocab (vocab32k.npz from
build_vocab.py) — ~110M params. Trains on ring chunks 000-008 (90k docs);
held-out loss on chunk009 docs (the eval split).

Usage (on tigerfish):
  python train_student.py --device cuda:0 [--resume ckpt_02500.pt]
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
STEPS = 2812                 # exp03-identical cosine schedule ends here
TOTAL_STEPS = 4218           # extension at MIN_LR, ~6 ring epochs
WARMUP = 100
PEAK_LR = 3e-4
MIN_LR = 3e-5
WEIGHT_DECAY = 0.1
CKPT_EVERY = 250
EVAL_EVERY = 250
EVAL_DOCS = 512


def load_corpus(root, vmap):
    files = [f for f in sorted(glob.glob(str(root / "ring_shard*_chunk*.npz")))
             if "chunk009" not in f]
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
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resume", default=None,
                    help="checkpoint filename to resume from, e.g. ckpt_02500.pt "
                         "(model weights only; optimizer restarts fresh)")
    args = ap.parse_args()

    root = Path(__file__).parent.parent / "results" / "corpus"
    out = Path(__file__).parent.parent / "results" / "students" / "ring"
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    torch.manual_seed(SEED)

    vmap = np.load(root / "vocab32k.npz")["map"]
    vocab_size = int(vmap.max()) + 1
    data = load_corpus(root, vmap)
    print(f"ring: {data.shape[0]} docs, vocab {vocab_size}", flush=True)

    eval_files = sorted(glob.glob(str(root / "ring_shard*_chunk009.npz")))
    eval_ids = np.concatenate([np.load(f)["ids"][:EVAL_DOCS // 4]
                               for f in eval_files])
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
    print(f"params: {n_all/1e6:.1f}M total, "
          f"{(n_all-n_emb)/1e6:.1f}M non-embedding", flush=True)
    (out / "config.json").write_text(cfg.to_json_string())

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
    for step in range(start, TOTAL_STEPS):
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
            print(f"step {step+1}/{TOTAL_STEPS} loss {loss_acc:.4f} "
                  f"lr {lr_at(step):.2e} ({rate/1e3:.0f}k tok/s)", flush=True)
            log.write(json.dumps({"step": step + 1, "loss": loss_acc,
                                  "lr": lr_at(step)}) + "\n")
            log.flush()
        if (step + 1) % EVAL_EVERY == 0:
            ev = run_eval()
            print(f"step {step+1}: heldout loss {ev:.4f}", flush=True)
            log.write(json.dumps({"step": step + 1, "heldout_loss": ev}) + "\n")
            log.flush()
        if (step + 1) % CKPT_EVERY == 0 or step + 1 == TOTAL_STEPS:
            sd = {k: v.bfloat16() for k, v in model.state_dict().items()}
            save_ckpt(sd, out / f"ckpt_{step+1:05d}.pt")

    print(f"ring training done: final heldout loss {run_eval():.4f}",
          flush=True)


if __name__ == "__main__":
    main()
