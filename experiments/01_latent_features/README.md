# Experiment 01 — Latent feature geometry in a toy transformer

**Status: v1 complete (Aug 2026). Hypothesis confirmed — see Findings.**

## Findings (v1, 3 seeds)

All three geometries form side by side in one 2-layer transformer, each matching its coupling
matrix. Headline numbers at layer 2 (residual stream at `:` after block 2), posterior-argmax
grouping with confidence > 0.5:

| Seed | identity corr / PR | ring corr / PR | line corr / PR |
|---|---|---|---|
| 0 | 0.995 / 6.94 | 0.996 / 3.63 | 0.994 / 3.70 |
| 1 | 0.993 / 6.91 | 0.997 / 3.49 | 0.991 / 3.58 |
| 2 | 0.996 / 6.95 | 0.994 / 3.37 | 0.995 / 3.68 |

(corr = Pearson between measured Gram matrix and double-centered M; PR = participation ratio.
Theoretical PR targets from M itself: identity 7.0, ring 3.82, line 4.1.)

- **Simplex:** identity feature reaches PR ≈ 6.94 of max 7 — a regular 7-simplex (its 2D PCA
  scatter is shapeless, as a regular simplex must be; the flat spectrum is the signature).
- **Ring:** attributes in exact circular order, closed loop. **Line:** open horseshoe, endpoints
  unjoined.
- **Built, not inherited:** correlations are 0.36–0.75 at embeddings/layer 1 and snap to >0.99
  after block 2. Attributes are never surface tokens.
- **Factorization:** principal angles between the three feature subspaces 83°–90° across all
  seeds — Park's hierarchical orthogonality emerges unimposed.
- **Behavioral M:** single-evidence-token prompts → marginalized outcome log-odds correlate
  0.996–0.998 with the true M rows.
- Robustness: grouping by sampled attribute (instead of posterior argmax) gives the same
  conclusions (see `metrics.json`, `gram_corr_sampled_labels`).

Implementation choices vs. the spec below: evidence tokens are feature-specific (each token
informative about exactly one feature, α = 1, round-robin assignment → balanced marginals);
output head predicts over the 512 outcome tokens only (makes model CE directly comparable to
Bayes-optimal); k fixed per batch (no padding). Training stops when KL(true‖model) < 0.02 nats
(seed 0 plateaued at ≈ 0.022 at the 30k-step cap; geometry unaffected).

Reproduce: `python src/train.py --preset main --seed 0 && python src/analyze.py results/main_seed0`
(v0 sanity: `--preset v0`). Runs ~5 min on MPS.

## Question

When a transformer must track several latent categorical features whose attribute couplings we
control, does each feature's hidden-state representation take the shape dictated by its coupling
matrix M — a regular simplex for identity M, a ring for a circular band M, an open curve for a
linear band M — even though no feature or attribute is ever a surface token?

If yes: polytopes and manifolds are one object parameterized by M, demonstrated causally (we set M,
geometry follows), in a setting where the geometry must be *built* in hidden states rather than
inherited by the readout dictionary.

## The world

Everything is generated from three coupling matrices we write down by hand.

**Latent features.** Three features, each with N = 8 attributes:

| Feature | M | Predicted geometry |
|---|---|---|
| A ("kind") | identity (pure categorical, Park's separable world) | regular simplex, dim ≈ N−1 |
| B ("hue") | circulant smooth band | ring, dim ≈ 2 |
| C ("size") | non-circulant smooth band (line topology) | open curve, dim ≈ 1–2 |

All M rows are centered (softmax is shift-invariant, so M is only identifiable up to per-row
constants). Attribute marginals kept balanced so frequency doesn't confound geometry.

**Evidence tokens.** Vocabulary of E = 64 evidence tokens. Each token e carries, per feature f, a
target attribute r_f(e) and a strength α_f(e) ≥ 0 (α = 0 → uninformative about that feature;
strengths sparse so single tokens are partial evidence). Assignments random, fixed at world
creation.

**Sequences.** Sample k ~ Uniform{1..8} evidence tokens i.i.d., append a separator `:`, then one
**outcome token**.

**Outcome.** One token per attribute *combination* — 8³ = 512 outcome tokens. Ground truth:
per-feature logits z_f = Σ_over-evidence α_f(e) · M_f[r_f(e), :]; the combo (a,b,c) has logit
z_A[a] + z_B[b] + z_C[c]; sample from the softmax. Features are independent given evidence, and
each M is exactly the law "one unit of evidence for i moves j's log-odds by M_ij."

Total vocab: 64 evidence + 1 separator + 512 outcomes (+ BOS).

Because the world is fully specified, the **Bayes-optimal cross-entropy is computable in closed
form** — training runs until the model is within tolerance of it, which replaces guesswork about
convergence.

## Model

2-layer transformer, d_model = 64, 4 heads, learned positional embeddings. Next-token loss **only at
the position after `:`** (predicting the outcome). Fresh-sampled batches every step (infinite data —
no memorization concerns). Runs are minutes on Apple Silicon (MPS); sweep = 3 seeds.

**v0 sanity mode (build first):** single feature, outcome token = the attribute itself. This is the
degenerate one-feature case of the same codebase; confirms the pipeline recovers the expected
shapes in the easy setting before the latent-feature headline run.

## Measurements

1. **Geometry (the headline).** Mean hidden state at the `:` position, grouped by the true attribute
   of feature f, centered within feature → attribute vectors ℓ_f,i. Per feature: Gram matrix,
   PCA scatter, effective dimensionality (participation ratio of the Gram spectrum).
   *Prediction:* simplex / ring / open curve side by side in one network; dim(ℓ_f) ≈ rank(centered M_f);
   Gram_f ∝ M_f (up to scale).
2. **Factorization.** Principal angles / cross-variance between the three feature subspaces.
   *Prediction:* near-orthogonal (connects to Park's hierarchical orthogonality). Non-factorized
   solutions are a finding, not a failure.
3. **Behavioral M.** Prompt with single evidence tokens, read outcome logits, marginalize to each
   feature; the induced Δlog-odds per feature should reproduce the corresponding M row. This is the
   original "M matrix" measurement, finally in a setting with no readout tautology and no context
   leakage.
4. **Belief trajectories.** Hidden state at `:` as k grows / evidence conflicts: posterior sharpening
   should move the state outward/along the manifold (connects to belief-state geometry).
5. **Layer/position sweep.** Where in the network each feature's geometry forms.

## Why this design (condensed from planning discussion)

- M is the *data-generating law*, so "geometry follows M" is causal by construction — no steering,
  no prompt design, no unembedding tautology (see root README, Design lessons).
- Attributes are latent (outcomes are combination tokens), mirroring real LLMs where color manifolds
  live in mid-layer states, not in color tokens.
- Three different M's in one network make the result internally controlled: same data budget, same
  architecture, only M differs.
- Measurement method (attribute-grouped mean hidden states) is identical to the planned LLM color
  study (exp 02, Abdou-style), so toy and wild results are directly comparable.

## Open questions — resolve before implementing

1. **Grouping label for measurement 1.** Group samples by the *sampled* outcome attribute, or by the
   argmax/mean of the *ground-truth posterior*? Sampled outcomes are noisy labels (the hidden state
   encodes the posterior, not the sample); posterior-based grouping is cleaner but should be
   pre-registered to avoid analysis degrees of freedom. Option: restrict to high-confidence samples.
2. **Evidence design details.** All-features-informative tokens vs. feature-specific tokens; α
   distribution (fixed α = 1 vs. varied); E = 64 vs. larger.
3. **Sizes.** N = 8 attributes is enough to distinguish ring from simplex; N = 12–16 gives prettier
   manifolds but 1728–4096 outcome tokens. Start N = 8?
4. **Model capacity.** d = 64 / 2 layers assumed sufficient; verify against Bayes-optimal loss and
   scale only if the model can't fit the world.
5. **Sequence length range.** k ≤ 8 assumed; longer k strengthens the trajectory analysis (meas. 4).
6. **Where to read states.** `:` position only vs. all positions; layer choice for headline figure.

## Future extensions (explicitly out of scope for v1)

- Co-occurrence dissociation world (Claim B in miniature): input co-occurrence structure ≠ output
  coupling; which one does geometry follow?
- Modifier/NOT token so evidence must be *computed* by attention (makes the transformer earn its
  place; does geometry form after resolution?).
- Architecture sweep (minimal log-linear model, MLP vs. transformer): architecture-independence as
  a result.
- Bandwidth sweep on M_B: continuous simplex→manifold morph as the kernel widens.
