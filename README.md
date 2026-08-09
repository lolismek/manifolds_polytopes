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

Equivalent statement in the language of computational mechanics (see next section): polytopes and
manifolds are both **feature belief geometries** — reachable sets of Bayesian posteriors over a
latent feature, linearly embedded in the residual stream. The shape is set by the number of
attributes (ambient simplex dimension) and by the structure of the labeled transition operators
T⁽ˣ⁾; M is the log-emission overlap kernel, and with static latents it alone determines the
reachable set.

The empirical claims, which concern *real LLMs* and are not settled by any toy result:

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

## Position relative to computational mechanics (reckoning, Aug 2026)

After exp01 was complete, we did a full read of three papers from the Simplex/Astera
computational-mechanics group. They cover — more thoroughly than exp01 — much of what we had
treated as this project's theoretical ground. Recording the boundary precisely so future work
builds on it instead of re-deriving it:

- **Shai et al. 2024 (belief state geometry, [2405.15943](https://arxiv.org/abs/2405.15943)).**
  Transformers trained on HMM output linearly represent the belief-state geometry (mixed-state
  presentation) of the generating process in the residual stream, including fractal cases; the
  representation can be spread across layers and carries beyond-next-token information. This is
  the parent framework for "coupling ⇒ deformed polytope": exp01's world maps onto it exactly.
  Our M rows are log-emission profiles (log P(evidence e | attribute j) = α·M[r(e), j] + const);
  the latents are static, so the transition part of T⁽ˣ⁾ is trivial and the emission overlap alone
  fixes the reachable belief set — contained in softmax of M's row space, hence dim ≈ rank of
  centered M. One methodological difference: they recover the geometry via fitted affine
  regression (agnostic to the native embedding); exp01 measured the native attribute-vector Gram
  and found it ∝ centered M.
- **Shai et al. 2026 (factored representations, [2602.02385](https://arxiv.org/abs/2602.02385)).**
  Conditionally independent latent factors get represented in orthogonal subspaces of the residual
  stream (the "Factored World Hypothesis"), with linear rather than exponential dimension growth —
  demonstrated with sub-token → product-token vocabularies (the same combination-token trick as
  exp01), with capacity, noise, and RNN/LSTM controls, and shown to be an inductive bias (models
  factor first even when factoring is lossy). **Exp01's near-orthogonal feature subspaces
  (83°–90°) are a replication of this, not a novel finding.**
- **Piotrowski et al. 2025 (constrained belief updates, [2502.01954](https://arxiv.org/abs/2502.01954)).**
  Mechanism-level account: the first attention layer implements the best *parallel* approximation
  to inherently recurrent Bayesian updating (per-source corrections πT^(|z)T^(d−s) − π summed with
  attention weights); attention patterns, OV vectors, and embedding directions are predicted
  spectrally from the eigenvalues of T; MLP and later layers refine the constrained geometry
  toward full Bayes. Our static-latent world is the degenerate corner of this theory (T =
  identity: no positional decay, no oscillation, parallel summation is exact), so mechanism-level
  questions about exp01 are answered by their framework.

What none of the three papers do: vary a factor's emission-overlap kernel to move its geometry
along the polytope↔manifold axis (the M-dial); connect belief geometry to Park's categorical
concepts or to the LLM feature-geometry literature (Engels' circles, color manifolds); or measure
any of this in a real LLM. The last two are explicitly flagged as open problems in Shai et al.
2024 (§4.3 and App. A.2 — "explain the days-of-the-week result using our framework") and in
Piotrowski et al.'s limitations. Claims A and B are untouched.

Consequence for this repo: the theoretical foundation is cited, not ours to defend; exp01 stands
as the emission-kernel corner of belief geometry — one network, three coupling kernels, geometry
follows the kernel (simplex / ring / open curve) — with its factorization result demoted to a
replication.

## Key papers

| Paper | Ref | Role here |
|---|---|---|
| Park, Chen, Veitch — *The Geometry of Categorical and Hierarchical Concepts in LLMs* | [2406.01506](https://arxiv.org/abs/2406.01506) | Starting framework: attribute vectors, causal inner product, categorical concepts as simplices via causal separability. We relax separability. |
| Shai, Marzen, Teixeira, Oldenziel, Riechers — *Transformers Represent Belief State Geometry in their Residual Stream* | [2405.15943](https://arxiv.org/abs/2405.15943) | Parent framework (NeurIPS 2024): activations linearly represent the mixed-state presentation of the generating process. Names the belief-states↔features bridge as open (§4.3). |
| Shai, Amdahl-Culleton et al. — *Transformers Learn Factored Representations* | [2602.02385](https://arxiv.org/abs/2602.02385) | Factored World Hypothesis: independent latent factors → orthogonal subspaces, via combination-token worlds. Subsumes exp01's factorization result. |
| Piotrowski, Riechers, Filan, Shai — *Constrained Belief Updates Explain Geometric Structures in Transformer Representations* | [2502.01954](https://arxiv.org/abs/2502.01954) | Mechanism (ICML 2025): attention implements parallelized approximate Bayes; attention/OV/embeddings predicted spectrally from T. Layer story (attention builds raw geometry, MLP/later layers refine) relevant to any layer-sweep measurement. |
| Fel, Kowal et al. — *Block-Sparse Featurizers Capture Visual Concept Manifolds* (BSF) | [2606.25234](https://arxiv.org/abs/2606.25234) | Concepts as low-dim manifolds (vision models). The phenomenon we want to explain. |
| Karkada, Korchinski, Nava, Wyart, Bahri — *Symmetry in language statistics shapes the geometry of model representations* | [2602.15029](https://arxiv.org/abs/2602.15029) | The rival theory: geometry from co-occurrence symmetry (months → circles). Our Claim B dissociates from it. Their robustness anomaly is our opening. |
| Engels et al. — *Not All Language Model Features Are One-Dimensionally Linear* | [2405.14860](https://arxiv.org/abs/2405.14860) | Circular multi-dim features (days/months) in LLMs; SAE-based discovery + causal use in computation. Methodological template. |
| Abdou et al. — *Can Language Models Encode Perceptual Structure Without Grounding? A Case Study in Color* | [ACL CoNLL 2021](https://aclanthology.org/2021.conll-1.9/) | Canonical color study: mean contextual embeddings of color terms align with CIELAB, peaking mid-layer. Our LLM measurement method. |
| Wurgaft, Rager et al. — *Manifold Steering Reveals the Shared Geometry of Representation and Behavior* | [2605.05115](https://arxiv.org/abs/2605.05115) | On-manifold vs. off-manifold steering; representation↔behavior manifold correspondence. |
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
   LLM situation (mid-layer color manifolds, not color tokens). (Shai et al. 2026 use the same
   construction independently.)
7. **Architecture-agnosticism is a claim to test.** The hypothesis is about distributional structure,
   not attention. (v1 tests transformers only; Shai et al. 2026 report RNN/LSTM replications of the
   factorization part.)

## Experiments

| # | Folder | Status | One-liner |
|---|---|---|---|
| 01 | `experiments/01_latent_features/` | **done — hypothesis confirmed** | Toy transformer; 3 latent features with chosen M's (identity / circular band / linear band). Each feature's layer-2 geometry matches its M (Gram-M corr > 0.99, 3 seeds): simplex, ring, and open curve in one network. Feature subspaces near-orthogonal (a replication of the Factored World Hypothesis — see Position section). v2: co-occurrence dissociation — M wins all symmetric venues; C lives only in the asymmetric readout product. |
| 02 | `experiments/02_llm_colors/` | planned | Companion in-the-wild study: color-term geometry in a real LLM (Gemma-2), Park separability violation along CIELAB, PMI vs. perceptual-metric regression. |

## Conventions

- One folder per experiment under `experiments/`, numbered, self-contained: own `README.md` (spec +
  findings), `src/`, `results/`. 
- Shared code gets extracted to a top-level `shared/` only when a second experiment actually needs it.
- Python via conda; each experiment pins its env when code lands.
