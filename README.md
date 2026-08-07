# manifold_polytopes

Experiments on hidden-state geometry in transformers — specifically, **why concepts form manifolds**.

## Central hypothesis

The polytopes of Park et al. (categorical concepts as simplices) and the concept manifolds of the
interpretability literature (BSF, circular features, color manifolds) are **the same object at
different settings of one dial: the attribute coupling matrix M**.

- For a categorical concept with N attributes, define **M**: when context provides one unit of
  evidence for attribute i (in log-odds), M_ij is how much attribute j's log-odds move.
- Park's *causal separability* assumption = M is the identity. His theorem: separability forces all
  pairwise overlaps of attribute vectors to be equal → the attribute vectors form a **regular simplex**.
- If attributes are coupled (evidence for *red* is partial evidence for *orange*), M becomes a
  structured kernel. Linear algebra then forces the vertices off the regular simplex: a **smooth
  banded M** makes the vertices lie on a low-dimensional curve — a **manifold**. With many
  fine-grained attributes this looks continuous.
- Quantitative form: the dimensionality of the attribute polytope ≈ rank of (row-centered) M.

**Part 1 of the thesis — "coupling ⇒ deformed polytope" — is a theorem given a softmax-linear
readout, not an empirical claim.** The empirical claims are:

- **Claim A (vs. Park):** real LLMs violate causal separability for graded categories (color), and
  the violation is structured: the overlap pattern tracks the *perceptual* metric (CIELAB hue),
  not noise.
- **Claim B (vs. Karkada):** the kernel that shapes the polytope is the **evidence-coupling
  ("same slots" / posterior coupling) kernel**, driven by continuous latent structure in the world —
  not raw textual co-occurrence ("appear together"). Dissociating cells: *black/white* and
  *red/green* co-occur heavily in text but are perceptually opposite; the two theories predict
  opposite things there. Note Karkada et al.'s own perturbation-robustness result (geometry
  survives deletion of co-occurrences) already points past co-occurrence toward a latent variable —
  our thesis names the latent and the mechanism.

## Key papers

| Paper | Ref | Role here |
|---|---|---|
| Park, Chen, Veitch — *The Geometry of Categorical and Hierarchical Concepts in LLMs* | [2406.01506](https://arxiv.org/abs/2406.01506) | Starting framework: attribute vectors, causal inner product, categorical concepts as simplices via causal separability. We relax separability. |
| Fel, Kowal et al. — *Block-Sparse Featurizers Capture Visual Concept Manifolds* (BSF) | [2606.25234](https://arxiv.org/abs/2606.25234) | Concepts as low-dim manifolds (vision models). The phenomenon we want to explain. |
| Karkada, Korchinski, Nava, Wyart, Bahri — *Symmetry in language statistics shapes the geometry of model representations* | [2602.15029](https://arxiv.org/abs/2602.15029) | The rival theory: geometry from co-occurrence symmetry (months → circles). Our Claim B dissociates from it. Their robustness anomaly is our opening. |
| Engels et al. — *Not All Language Model Features Are One-Dimensionally Linear* | [2405.14860](https://arxiv.org/abs/2405.14860) | Circular multi-dim features (days/months) in LLMs; SAE-based discovery + causal use in computation. Methodological template. |
| Abdou et al. — *Can Language Models Encode Perceptual Structure Without Grounding? A Case Study in Color* | [ACL CoNLL 2021](https://aclanthology.org/2021.conll-1.9/) | Canonical color study: mean contextual embeddings of color terms align with CIELAB, peaking mid-layer. Our LLM measurement method. |
| Wurgaft, Rager et al. — *Manifold Steering Reveals the Shared Geometry of Representation and Behavior* | [2605.05115](https://arxiv.org/abs/2605.05115) | On-manifold vs. off-manifold steering; representation↔behavior manifold correspondence. |
| Shai et al. — *Transformers represent belief state geometry in their residual stream* | 2024 | Toy transformers on controlled worlds represent posteriors geometrically. Nearest neighbor to our toy experiment. |
| Ma, Beck, Latham, Pouget — *Bayesian inference with probabilistic population codes* | Nat. Neuro 2006 | Neuroscience lineage: summing tuned-neuron spikes = Bayesian evidence pooling; **M = tuning-curve overlap**; smooth overlap → ring attractors. |
| Arora et al. — *A Latent Variable Model Approach to Word Embeddings* (RAND-WALK) | TACL 2016 | Embedding geometry inherited from a log-linear generative model. Genre ancestor. |
| Elhage et al. — *Toy Models of Superposition* | 2022 | Precedent for controlled-data toy geometry studies. |
| Nanda et al. — grokking modular arithmetic | 2023 | Circles emerge in small transformers on synthetic tasks. |

## Design lessons (from initial planning discussion, Aug 2026)

Why the first experiment is synthetic, and shaped the way it is:

1. **Readout tautology.** At an LLM's final layer, logits = ⟨h, γ_j⟩, so *any* context shift couples
   similar words' logits through the unembedding Gram matrix. Observational "evidence for blue also
   raises azure" measurements are contaminated by this for free. Park-level (unembedding-space)
   versions of the experiment are close to definitionally true.
2. **Context leakage.** Natural evidence prompts carry topical baggage (Christmas prompts couple
   red/green) — which is exactly the rival (co-occurrence) theory's signal. Observational M on an
   LLM confounds the two theories.
3. **Steering circularity.** Steering along attribute i's direction injects its overlap with
   neighbors by hand; naive steering reads back the injection. Fixes exist (pass-through baseline,
   orthogonalized steering, snap-back tracking) but they add machinery, not clarity.
4. **Escape: make M the data-generating law.** In a synthetic world we choose M; geometry becomes
   the dependent variable. No confounds, full control, known ground truth.
5. **Don't presuppose the geometry.** An earlier design used a circular latent θ — but that bakes in
   the answer's shape. Parameterize the world by M directly; a circle is just one M.
6. **Keep attributes latent.** If attributes are output tokens, the geometry sits in the unembedding
   matrix (semi-trivial factorization). Make outcomes *combination* tokens so features exist only as
   latent factors — then finding a manifold in the hidden states is a discovery, mirroring the real
   LLM situation (mid-layer color manifolds, not color tokens).
7. **Architecture-agnosticism is a claim to test.** The hypothesis is about distributional structure,
   not attention. (v1 tests transformers only; MLP/minimal-model comparisons are a later robustness
   axis.)

## Experiments

| # | Folder | Status | One-liner |
|---|---|---|---|
| 01 | `experiments/01_latent_features/` | spec written, not implemented | Toy transformer; 3 latent features with chosen M's (identity / circular band / linear band); does each feature's hidden-state geometry take its M's shape? |
| 02 | `experiments/02_llm_colors/` | planned | Companion in-the-wild study: color-term geometry in a real LLM (Gemma-2), Park separability violation along CIELAB, PMI vs. perceptual-metric regression. |

## Conventions

- One folder per experiment under `experiments/`, numbered, self-contained: own `README.md` (spec +
  findings), `src/`, `results/`. 
- Shared code gets extracted to a top-level `shared/` only when a second experiment actually needs it.
- Python via conda; each experiment pins its env when code lands.
