"""Train the toy transformer on the synthetic world until it is within
tolerance of the Bayes-optimal loss (measured as mean KL from the true
outcome posterior to the model's prediction — 0 iff the model is exactly
Bayes-optimal).

Usage:
  python train.py --preset v0   # single feature (ring), sanity mode
  python train.py --preset main --seed 0
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from world import World
from model import TinyGPT

PRESETS = {
    # v0 sanity: one feature, outcome token = the attribute itself.
    "v0": dict(m_kinds=["ring"], E=16, run="v0_ring"),
    # headline: three latent features, combination-token outcomes.
    "main": dict(m_kinds=["identity", "ring", "line"], E=64, run="main"),
    # v2 dissociation: correlated sampler (opposite-attribute partners),
    # loss at all positions (next-token + outcome).
    "cooc": dict(m_kinds=["identity", "ring", "line"], E=64, run="cooc",
                 cooc_rho=0.6, tok_loss=True),
    # v2 control: same correlated data, loss only at the outcome position.
    "cooc_ctrl": dict(m_kinds=["identity", "ring", "line"], E=64,
                      run="cooc_ctrl", cooc_rho=0.6, tok_loss=False),
}


@torch.no_grad()
def evaluate(model, world, device, n_batches=16, B=512, generator=None):
    """Mean KL(true || model) and CE gap (model CE - Bayes CE), in nats."""
    model.eval()
    kls, gaps = [], []
    for _ in range(n_batches):
        x, y, _, p = world.sample_batch(B, generator=generator)
        logits = model(x.to(device))[:, -1].float().cpu()
        lq = F.log_softmax(logits, dim=-1)
        lp = world.true_log_outcome(p)
        kls.append((p_full := lp.exp()).mul(lp - lq).sum(-1).mean().item())
        ce_model = F.cross_entropy(logits, y).item()
        gaps.append(ce_model - world.bayes_entropy(p).mean().item())
    model.train()
    return sum(kls) / len(kls), sum(gaps) / len(gaps)


def train(preset: str, seed: int, max_steps: int, tol: float, out_root: Path):
    cfg = PRESETS[preset]
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 10_000)  # data stream

    world = World(m_kinds=cfg["m_kinds"], E=cfg["E"],
                  cooc_rho=cfg.get("cooc_rho", 0.0))
    tok_loss = cfg.get("tok_loss", False)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = TinyGPT(world.vocab_in, world.n_outcomes,
                    max_len=world.max_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    run_dir = out_root / f"{cfg['run']}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    B, eval_every, patience_needed = 256, 500, 2
    t0, patience, history = time.time(), 0, []
    for step in range(1, max_steps + 1):
        x, y, _, _ = world.sample_batch(B, generator=gen)
        x_dev = x.to(device)
        logits, states = model(x_dev, return_states=True)
        loss = F.cross_entropy(logits[:, -1], y.to(device))
        if tok_loss and x.shape[1] > 1:
            tok_logits = model.tok_head(model.ln_f(states[-1][:, :-1]))
            loss = loss + F.cross_entropy(
                tok_logits.reshape(-1, world.vocab_in),
                x_dev[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % eval_every == 0:
            kl, gap = evaluate(model, world, device, generator=gen)
            history.append(dict(step=step, kl=kl, ce_gap=gap))
            print(f"step {step:6d}  KL(true||model) {kl:.4f}  "
                  f"CE gap {gap:+.4f}  ({time.time()-t0:.0f}s)", flush=True)
            patience = patience + 1 if kl < tol else 0
            if patience >= patience_needed:
                print(f"converged: KL < {tol} for {patience_needed} evals")
                break

    torch.save(dict(model=model.state_dict(),
                    world=world.config(),
                    preset=preset, seed=seed), run_dir / "ckpt.pt")
    (run_dir / "train_log.json").write_text(json.dumps(
        dict(preset=preset, seed=seed, final_kl=history[-1]["kl"],
             steps=history[-1]["step"], tol=tol, history=history), indent=2))
    print(f"saved to {run_dir}")
    return run_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=30_000)
    ap.add_argument("--tol", type=float, default=0.02)
    args = ap.parse_args()
    out_root = Path(__file__).resolve().parent.parent / "results"
    train(args.preset, args.seed, args.max_steps, args.tol, out_root)
