"""Shared clean-cache steering machinery for exp04.

The headline design change vs exp03: the clamp is applied only at the
position whose next-token distribution is being computed, and the KV cache is
always built from unclamped passes over the realized text. Each token's
sampling distribution therefore depends on (visible text so far, current
state) only — no clamp history hides in the cache — so the corpus is a true
HMM over the cast states and the ideal reader's forward algorithm is *exact*,
not an approximation as in exp03.

Mechanics per generated token (generate_clean_cache):
  1. clamped pass on the current position (hook on) -> sampling logits
  2. crop the clamped pass's KV entries off the cache
  3. clean pass on the same position (hook off) -> extends the cache
The exact scorer (score_emissions) teacher-forces the same op sequence with H
clamp hypotheses per document, so the emission probabilities it produces are
the very distributions generation sampled from (verified in
verify_identity.py).

Every steered token in exp04 — pilots, audition, corpus — must go through
generate_clean_cache; every emission probability through score_emissions.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "google/gemma-2-2b"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_12/width_16k/canonical"
LAYER = 12

EXP03 = Path(__file__).parents[2] / "03_steered_corpus" / "results"


def load_teacher(dev):
    from sae_lens import SAE

    sae = SAE.from_pretrained(SAE_RELEASE, SAE_ID, device=str(dev))[0]
    sae = sae.to(torch.float32)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16).to(dev)
    model.eval()
    return model, tok, sae


def audition_pool() -> list[int]:
    """The 48 mutually-clean candidates from exp03 (stage-2 greedy set)."""
    return json.loads((EXP03 / "stage3" / "audition_pool.json").read_text())["pool"]


def mean_act():
    import numpy as np

    return np.load(EXP03 / "stage1" / "latent_stats.npz")["mean_act"]


def get_prompts(tokenizer, n, prompt_len=12):
    """(n, prompt_len+1) BOS-prefixed openers from the first n usable pile docs
    (same recipe as exp03 stage 3, shared across all arms)."""
    from datasets import load_dataset

    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    bos = tokenizer.bos_token_id
    prompts = []
    for ex in ds:
        toks = tokenizer(ex["text"][:600], add_special_tokens=False)["input_ids"]
        if len(toks) >= prompt_len:
            prompts.append([bos] + toks[:prompt_len])
        if len(prompts) == n:
            return torch.tensor(prompts)


class Steerer:
    """One-hot clamp hook over a fixed latent pool: per batch row, the active
    latent is clamped to its target value and every other pool latent is
    ablated to 0. target is a (B, n_pool) row matrix, or None for clean."""

    def __init__(self, model, sae, pool_ids, dev):
        self.pool = [int(j) for j in pool_ids]
        self.pos = {j: k for k, j in enumerate(self.pool)}
        self.pool_t = torch.tensor(self.pool, device=dev)
        self.W = sae.W_dec[self.pool_t]                    # (n_pool, d)
        self.sae = sae
        self.dev = dev
        self.target = None
        model.model.layers[LAYER].register_forward_hook(self._hook)

    def rows(self, latents, values):
        """(B,) latent ids, (B,) clamp values -> (B, n_pool) target rows."""
        t = torch.zeros(len(latents), len(self.pool), device=self.dev)
        cols = torch.tensor([self.pos[int(j)] for j in latents], device=self.dev)
        t[torch.arange(len(latents), device=self.dev), cols] = values
        return t

    def _hook(self, _, __, output):
        if self.target is None:
            return
        is_tuple = isinstance(output, tuple)
        x = output[0] if is_tuple else output
        xf = x.float()
        B, P, d = xf.shape
        a = self.sae.encode(xf.reshape(-1, d))[:, self.pool_t].reshape(B, P, -1)
        delta = (self.target.unsqueeze(1) - a) @ self.W
        x = (xf + delta).to(x.dtype)
        return (x,) + output[1:] if is_tuple else x


@torch.no_grad()
def generate_clean_cache(model, steer, openers, targets, gen_len,
                         record_logp=False):
    """openers (B, P) int64, BOS-prefixed. targets: (B, n_pool) tensor applied
    every step, a callable t -> (B, n_pool) (per-token paths), or None (clean
    docs; still runs the two-pass loop so all arms share op sequence).

    Returns (ids (B, P+gen_len) cpu, stats) where stats holds per-doc means
    over generated tokens: kl = KL(clamped || clean) at the sampled position
    (the evidence rate), llr = logP_clamped - logP_clean of the sampled token,
    nll_clean = -logP_clean of the sampled token (fluency), ent = clamped
    entropy; with record_logp also logp_gen (B, gen_len) = clamped logP of
    each sampled token (for the identity check against score_emissions).
    """
    dev = next(model.parameters()).device
    openers = openers.to(dev)
    B = openers.shape[0]
    steer.target = None
    o = model(openers[:, :-1], use_cache=True)
    past = o.past_key_values
    base = openers.shape[1] - 1
    cur = openers[:, -1:]
    gen, logp_rows = [], []
    zeros = torch.zeros(B, device=dev)
    kl_s, llr_s, nll_s, ent_s = (zeros.clone() for _ in range(4))
    for t in range(gen_len):
        steer.target = targets(t) if callable(targets) else targets
        o = model(cur, past_key_values=past, use_cache=True)
        past = o.past_key_values
        lc = F.log_softmax(o.logits[:, -1].float(), -1)
        past.crop(base + t)                    # discard the clamped-pass KV
        steer.target = None
        o = model(cur, past_key_values=past, use_cache=True)
        past = o.past_key_values
        ln = F.log_softmax(o.logits[:, -1].float(), -1)
        nxt = torch.multinomial(lc.exp(), 1)
        kl_s += (lc.exp() * (lc - ln)).sum(-1)
        llr_s += (lc - ln).gather(1, nxt)[:, 0]
        nll_s += -ln.gather(1, nxt)[:, 0]
        ent_s += -(lc.exp() * lc).sum(-1)
        if record_logp:
            logp_rows.append(lc.gather(1, nxt)[:, 0].cpu())
        gen.append(nxt.cpu())
        cur = nxt
    steer.target = None
    ids = torch.cat([openers.cpu(), torch.cat(gen, 1)], 1)
    stats = {"kl": (kl_s / gen_len).cpu(), "llr": (llr_s / gen_len).cpu(),
             "nll_clean": (nll_s / gen_len).cpu(),
             "ent": (ent_s / gen_len).cpu()}
    if record_logp:
        stats["logp_gen"] = torch.stack(logp_rows, 1)
    return ids, stats


@torch.no_grad()
def score_emissions(model, steer, ids, hyp_rows, prompt_len):
    """Exact emissions, teacher-forced. ids (B, T) full documents (opener of
    prompt_len incl. BOS + generated tokens); hyp_rows (H, n_pool) one clamp
    row per hypothesis. Row layout doc-major (d*H + h).

    Returns (logp (B, H, T-prompt_len), logp_clean (B, T-prompt_len)):
    logp[d, h, t] = log P(x_{prompt_len+t} | previous text, z = hypothesis h)
    — under the clean-cache scheme these are exactly the distributions
    generation sampled from, so the forward algorithm over them is exact.
    """
    dev = next(model.parameters()).device
    B, T = ids.shape
    H = hyp_rows.shape[0]
    G = T - prompt_len
    ids_x = ids.repeat_interleave(H, 0).to(dev)
    tgt = hyp_rows.to(dev).repeat(B, 1)
    steer.target = None
    o = model(ids_x[:, :prompt_len - 1], use_cache=True)
    past = o.past_key_values
    base = prompt_len - 1
    logp = torch.empty(B, H, G, dtype=torch.float32)
    logp_clean = torch.empty(B, G, dtype=torch.float32)
    for t in range(G):
        cur = ids_x[:, base + t : base + t + 1]
        nxt = ids_x[:, base + t + 1]
        steer.target = tgt
        o = model(cur, past_key_values=past, use_cache=True)
        past = o.past_key_values
        lc = F.log_softmax(o.logits[:, -1].float(), -1)
        logp[:, :, t] = lc.gather(1, nxt[:, None])[:, 0].reshape(B, H).cpu()
        past.crop(base + t)
        steer.target = None
        o = model(cur, past_key_values=past, use_cache=True)
        past = o.past_key_values
        ln = F.log_softmax(o.logits[:, -1].float(), -1)
        logp_clean[:, t] = (ln.gather(1, nxt[:, None])[:, 0]
                            .reshape(B, H)[:, 0].cpu())
    steer.target = None
    return logp, logp_clean
