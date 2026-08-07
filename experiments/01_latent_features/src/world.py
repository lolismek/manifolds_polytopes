"""The synthetic world: everything is generated from coupling matrices M we write by hand.

Sequences look like:  [e_1, ..., e_k, SEP]  ->  outcome token
where each evidence token e is informative about exactly one latent feature
(it adds beta * M_f[r(e), :] to that feature's logits), and the outcome token
encodes the *combination* of sampled attributes, one per feature.
"""

import numpy as np
import torch


def make_M(kind: str, N: int = 8, sigma: float = 1.0) -> np.ndarray:
    """Row-centered coupling matrix. Rows are centered because softmax is
    shift-invariant: M is only identifiable up to per-row constants."""
    idx = np.arange(N)
    if kind == "identity":
        K = np.eye(N)
    elif kind == "ring":
        d = np.abs(idx[:, None] - idx[None, :])
        d = np.minimum(d, N - d)
        K = np.exp(-d.astype(float) ** 2 / (2 * sigma**2))
    elif kind == "line":
        d = np.abs(idx[:, None] - idx[None, :]).astype(float)
        K = np.exp(-d**2 / (2 * sigma**2))
    else:
        raise ValueError(kind)
    return K - K.mean(axis=1, keepdims=True)


class World:
    def __init__(
        self,
        m_kinds=("identity", "ring", "line"),
        N: int = 8,
        E: int = 64,
        beta: float = 2.0,
        sigma: float = 1.0,
        k_max: int = 8,
    ):
        self.m_kinds = tuple(m_kinds)
        self.N, self.E, self.beta, self.sigma, self.k_max = N, E, beta, sigma, k_max
        self.F = len(m_kinds)
        self.Ms = np.stack([make_M(k, N, sigma) for k in m_kinds])  # (F, N, N)

        # Evidence table: token e is informative about feature e % F, targeting
        # attributes round-robin within the feature (balanced marginals by design).
        self.tok_feat = np.arange(E) % self.F
        self.tok_attr = np.zeros(E, dtype=int)
        for f in range(self.F):
            ids = np.where(self.tok_feat == f)[0]
            self.tok_attr[ids] = np.arange(len(ids)) % N

        # Per-token additive evidence: R[e, f, :] = beta * M_f[r(e), :] for e's
        # own feature, zero for the others.
        R = np.zeros((E, self.F, N))
        R[np.arange(E), self.tok_feat] = beta * self.Ms[self.tok_feat, self.tok_attr]
        self.R = torch.tensor(R, dtype=torch.float32)

        self.SEP = E
        self.vocab_in = E + 1
        self.n_outcomes = N**self.F
        self.max_len = k_max + 1

    def config(self):
        return dict(m_kinds=list(self.m_kinds), N=self.N, E=self.E,
                    beta=self.beta, sigma=self.sigma, k_max=self.k_max)

    def sample_batch(self, B: int, k: int | None = None, generator=None):
        """Returns x (B, k+1) input tokens, y (B,) outcome ids, attrs (B, F)
        sampled attributes, p (B, F, N) true per-feature posteriors."""
        if k is None:
            k = int(torch.randint(1, self.k_max + 1, (1,), generator=generator))
        toks = torch.randint(0, self.E, (B, k), generator=generator)
        z = self.R[toks].sum(dim=1)                     # (B, F, N)
        p = torch.softmax(z, dim=-1)
        attrs = torch.stack(
            [torch.multinomial(p[:, f], 1, generator=generator).squeeze(1)
             for f in range(self.F)], dim=1)            # (B, F)
        y = torch.zeros(B, dtype=torch.long)
        for f in range(self.F):
            y = y * self.N + attrs[:, f]
        x = torch.cat([toks, torch.full((B, 1), self.SEP, dtype=torch.long)], dim=1)
        return x, y, attrs, p

    def true_log_outcome(self, p: torch.Tensor) -> torch.Tensor:
        """(B, F, N) per-feature posteriors -> (B, N^F) log-probs over outcome
        tokens (features are independent given evidence)."""
        lp = torch.log(p)
        full = lp[:, 0]
        for f in range(1, self.F):
            full = (full.unsqueeze(-1) + lp[:, f].unsqueeze(1)).flatten(1)
        return full

    def bayes_entropy(self, p: torch.Tensor) -> torch.Tensor:
        """Per-sample entropy of the true outcome distribution (nats)."""
        return -(p * torch.log(p)).sum(-1).sum(-1)
