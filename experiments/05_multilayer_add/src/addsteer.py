"""exp05 multi-layer additive steering.

Intervention: for state i, add  s * mean_act_i * (rms_l / rms_12) * W_dec[i]
to the residual stream at every layer in WINDOW (12 .. L-2), at the sampled
position only. No clamp, no one-hot ablation — the exact reader makes that
bookkeeping unnecessary, and on the (overwhelmingly common) tokens where the
cast latents don't fire, clamp-to-value and add-value are identical anyway.

s is the single strength knob, in units of the latent's mean firing
activation at layer 12 (same units as exp03/exp04 audition numbers). The
per-layer rms ratio keeps the injection at a constant *fraction* of the
residual stream as its norm grows with depth; it is measured once
(`python addsteer.py --device cuda:0`) and frozen in results/calib/.

Generation/scoring physics (clean-cache two-pass loop, exact scorer) are
imported unchanged from exp04 — the intervention is still a deterministic
function of (state, sampled position), so the corpus stays a true HMM.
"""

import json
import sys
from pathlib import Path

import torch

EXP04_SRC = Path(__file__).parents[2] / "04_clean_cache" / "src"
sys.path.append(str(EXP04_SRC))
from steering import (LAYER, generate_clean_cache, get_prompts,  # noqa: E402,F401
                      load_teacher, mean_act, score_emissions)

WINDOW = list(range(12, 24))          # gemma-2-2b: 26 layers, inject 12..L-2
CALIB = Path(__file__).parent.parent / "results" / "calib" / "layer_rms.json"
EXP04_RES = Path(__file__).parents[2] / "04_clean_cache" / "results"


def cast() -> list[int]:
    """exp04's exact-likelihood-confusability cast of 8."""
    return json.loads((EXP04_RES / "confusability" / "cast.json").read_text())["cast"]


def load_scales() -> dict[int, float]:
    if not CALIB.exists():
        raise FileNotFoundError(
            f"{CALIB} missing — run `python addsteer.py --device cuda:0` once")
    return {int(l): v for l, v in json.loads(CALIB.read_text())["scale"].items()}


class AddSteerer:
    """Additive multi-layer hook over a fixed latent pool. target is a
    (B, n_pool) matrix of added coefficients (layer-12 activation units), or
    None for clean. Interface-compatible with exp04's Steerer, so
    generate_clean_cache / score_emissions work unchanged."""

    def __init__(self, model, sae, pool_ids, dev, scales=None):
        self.pool = [int(j) for j in pool_ids]
        self.pos = {j: k for k, j in enumerate(self.pool)}
        self.W = sae.W_dec[torch.tensor(self.pool, device=dev)]   # (n_pool, d)
        self.dev = dev
        self.target = None
        scales = scales or load_scales()
        for l in WINDOW:
            model.model.layers[l].register_forward_hook(self._make_hook(scales[l]))

    def rows(self, latents, values):
        """(B,) latent ids, (B,) added coefficients -> (B, n_pool) rows."""
        t = torch.zeros(len(latents), len(self.pool), device=self.dev)
        cols = torch.tensor([self.pos[int(j)] for j in latents], device=self.dev)
        t[torch.arange(len(latents), device=self.dev), cols] = values
        return t

    def _make_hook(self, scale):
        def hook(_, __, output):
            if self.target is None:
                return
            is_tuple = isinstance(output, tuple)
            x = output[0] if is_tuple else output
            delta = (self.target @ self.W) * scale                # (B, d)
            x = (x.float() + delta.unsqueeze(1)).to(x.dtype)
            return (x,) + output[1:] if is_tuple else x
        return hook


@torch.no_grad()
def calibrate(dev, n_docs=64, seq_len=256, batch=8, skip=4):
    """Per-layer residual rms over pile prefixes (positions after `skip`,
    BOS and early positions excluded — their norms are outliers)."""
    from datasets import load_dataset

    model, tok, _ = load_teacher(dev)
    ds = load_dataset("monology/pile-uncopyrighted", split="train",
                      streaming=True)
    bos = tok.bos_token_id
    rows = []
    for ex in ds:
        t = tok(ex["text"], add_special_tokens=False)["input_ids"]
        if len(t) >= seq_len - 1:
            rows.append([bos] + t[: seq_len - 1])
        if len(rows) == n_docs:
            break
    ids = torch.tensor(rows)

    sums = {l: 0.0 for l in WINDOW}
    n = 0
    for b0 in range(0, n_docs, batch):
        out = model(ids[b0 : b0 + batch].to(dev), output_hidden_states=True)
        for l in WINDOW:
            x = out.hidden_states[l + 1][:, skip:, :].float()
            sums[l] += x.pow(2).mean(-1).sqrt().sum().item()
        n += out.hidden_states[0][:, skip:, 0].numel()
    rms = {l: sums[l] / n for l in WINDOW}
    scale = {l: rms[l] / rms[WINDOW[0]] for l in WINDOW}
    CALIB.parent.mkdir(parents=True, exist_ok=True)
    CALIB.write_text(json.dumps({"rms": rms, "scale": scale}, indent=2))
    print(json.dumps({"rms": rms, "scale": scale}, indent=2))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    calibrate(torch.device(args.device))
