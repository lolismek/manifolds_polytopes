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
        cooc_rho: float = 0.0,
    ):
        self.m_kinds = tuple(m_kinds)
        self.N, self.E, self.beta, self.sigma, self.k_max = N, E, beta, sigma, k_max
        self.cooc_rho = cooc_rho
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

        # v2 correlated sampler: each token's "partners" are the tokens of the
        # same feature targeting the *opposite* attribute ((a + N/2) mod N).
        # Padded to 6 slots by cycling (2 or 3 partners both divide 6, so
        # uniform sampling over slots is exactly uniform over partners).
        P = 6
        self.partner_table = np.zeros((E, P), dtype=np.int64)
        for e in range(E):
            f, a2 = self.tok_feat[e], (self.tok_attr[e] + N // 2) % N
            ids = np.where((self.tok_feat == f) & (self.tok_attr == a2))[0]
            self.partner_table[e] = np.resize(ids, P)
        self.partner_table = torch.tensor(self.partner_table)

    def config(self):
        return dict(m_kinds=list(self.m_kinds), N=self.N, E=self.E,
                    beta=self.beta, sigma=self.sigma, k_max=self.k_max,
                    cooc_rho=self.cooc_rho)

    def _sample_tokens(self, B: int, k: int, generator=None):
        if self.cooc_rho == 0 or k == 1:
            return torch.randint(0, self.E, (B, k), generator=generator)
        toks = torch.empty(B, k, dtype=torch.long)
        toks[:, 0] = torch.randint(0, self.E, (B,), generator=generator)
        for i in range(1, k):
            partner = self.partner_table[
                toks[:, i - 1],
                torch.randint(0, self.partner_table.shape[1], (B,),
                              generator=generator)]
            uniform = torch.randint(0, self.E, (B,), generator=generator)
            use = torch.rand(B, generator=generator) < self.cooc_rho
            toks[:, i] = torch.where(use, partner, uniform)
        return toks

    def sample_batch(self, B: int, k: int | None = None, generator=None):
        """Returns x (B, k+1) input tokens, y (B,) outcome ids, attrs (B, F)
        sampled attributes, p (B, F, N) true per-feature posteriors."""
        if k is None:
            k = int(torch.randint(1, self.k_max + 1, (1,), generator=generator))
        toks = self._sample_tokens(B, k, generator=generator)
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

    def cooc_pmi(self, n_seq: int = 200_000, seed: int = 7) -> np.ndarray:
        """Attribute-level within-sequence PMI per feature, measured empirically
        from the sampler — the toy analogue of measuring word co-occurrence in a
        corpus. Returns (F, N, N)."""
        gen = torch.Generator().manual_seed(seed)
        counts = np.zeros((self.F, self.N, self.N))
        B = 4096
        for _ in range(n_seq // B):
            k = int(torch.randint(2, self.k_max + 1, (1,), generator=gen))
            toks = self._sample_tokens(B, k, generator=gen).numpy()
            feats, attrs = self.tok_feat[toks], self.tok_attr[toks]
            for i in range(k):
                for j in range(i + 1, k):
                    same = feats[:, i] == feats[:, j]
                    np.add.at(counts, (feats[same, i], attrs[same, i],
                                       attrs[same, j]), 1)
                    np.add.at(counts, (feats[same, j], attrs[same, j],
                                       attrs[same, i]), 1)
        pmi = np.zeros_like(counts)
        for f in range(self.F):
            C = counts[f] + 1.0  # smoothing
            joint = C / C.sum()
            marg = joint.sum(axis=1)
            pmi[f] = np.log(joint / (marg[:, None] * marg[None, :]))
        return pmi
