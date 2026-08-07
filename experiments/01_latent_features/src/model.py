"""Tiny GPT-style transformer. The output head predicts only over outcome
tokens, so model cross-entropy is directly comparable to the Bayes-optimal
value computed from the world."""

import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, h):
        hn = self.ln1(h)
        a, _ = self.attn(hn, hn, hn, is_causal=True,
                         attn_mask=nn.Transformer.generate_square_subsequent_mask(
                             h.shape[1], device=h.device))
        h = h + a
        h = h + self.mlp(self.ln2(h))
        return h


class TinyGPT(nn.Module):
    def __init__(self, vocab_in: int, n_out: int, d: int = 64,
                 n_heads: int = 4, n_layers: int = 2, max_len: int = 16):
        super().__init__()
        self.emb = nn.Embedding(vocab_in, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(Block(d, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_out)
        # v2: next-token head over input vocab (evidence tokens + SEP); used
        # only by presets that train with loss at all positions.
        self.tok_head = nn.Linear(d, vocab_in)

    def forward(self, x, return_states: bool = False):
        T = x.shape[1]
        h = self.emb(x) + self.pos(torch.arange(T, device=x.device))
        states = [h]  # residual stream: after embedding, then after each block
        for blk in self.blocks:
            h = blk(h)
            states.append(h)
        logits = self.head(self.ln_f(h))
        if return_states:
            return logits, states
        return logits
