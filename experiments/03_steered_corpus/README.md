# exp03 — steered-corpus latents: do students learn belief states from natural language?

**Status: broad spec only.** Variants are listed but deliberately not expanded — each branch's
details get decided together before any code is written.

## Question

The comp-mech papers (Shai 2024, Piotrowski 2025) show belief-state geometry in transformers
trained on *synthetic token streams* from known HMMs. Nobody has shown it for a model trained on
*natural language* with a known latent — because in natural corpora the true latent structure is
unknown, so there is no ground truth to compare beliefs against. The Bayesian Attention Trilogy
explicitly names "natural-language wind tunnels" as the open challenge.

This experiment builds one: generate a natural-language corpus where **we** control a hidden
variable z, train a fresh LLM on it, and test whether that student tracks the ideal reader's
probability distribution over z — and whether its internal arrangement of z-values has the
geometry the corpus statistics predict.

## Core loop (common to all variants)

1. **Teacher + knob.** Take a trained LLM and an SAE on it. Steering along chosen SAE directions
   is the actuator: setting z changes the teacher's next-token distribution, so each z-value
   defines a distribution over documents.
2. **Corpus with a secret.** Generate many documents, each with a z assignment we record but
   never write into the text. Filter documents with overt lexical leakage (filtering depends only
   on the text, so it does not distort the posterior P(z | text)).
3. **Measure the realized world.** Don't trust the intended structure — measure it. Cross-evaluate
   teacher likelihoods (how probable does setting j make text generated under setting i) to get
   the *realized* emission kernel M, bleed and artifacts included. All predictions are graded
   against the measured world, not the intended one.
4. **Ground truth.** For any prefix, the ideal reader's posterior over z is exactly computable
   from teacher likelihoods + the known z-process (Bayes for static z; forward algorithm for
   dynamic z). Mushy steering doesn't blur this — it only makes the true posteriors graded, which
   is what makes the test meaningful.
5. **Fresh student.** Train a new LLM from scratch on the corpus. Steering never touches the
   student; the only channel from z to student is text. This is what breaks the circularity
   objection: the teacher is an actuator, the claim is tested only on the token-trained student.
6. **Test battery** (below).

## Variants

### A. Hash latents + designed dynamics (T-dial) — candidate first experiment

Pick K SAE latents *at random* (approximately orthogonal, no semantic relation). Build an HMM
over the K states with a structured transition matrix — e.g. states on a ring, mostly stay,
occasionally step to a neighbor. Steer along the current state's direction while generating;
z moves within each document.

- Hash property: the individual latents occur naturally in text, but the *coupling between them*
  (ring adjacency) occurs nowhere in nature. If the student's arrangement of the K states forms
  the ring in the right order, the only possible source is our transition statistics.
- Emissions are (intended to be) unrelated → M ≈ identity → any geometry comes from **T**.
- Sharp spectral prediction: circulant T has Fourier-mode eigenvectors; the student's state
  representations should organize on a ring in the top-2 mode plane, ordered correctly, with
  variance decaying along T's eigenvalue spectrum.
- Sub-variants (same latents, different transition graph): ring / line / clusters / random graph
  → different predicted geometries from identical emissions. T-side twin of exp01's
  three-M's-one-network design.

### B. Concept manifold, static z (M-dial)

Choose a graded concept (color, emotion). Settings are points on the concept's manifold (e.g.
hues around the circle); one setting per document, held fixed. Steering bleed between neighboring
settings is not a bug — the overlap between settings' text distributions *is* M, and the belief
geometry it induces is the object of study.

- Connects directly to the repo's original thesis (emission-overlap M-dial, Park separability,
  CIELAB) and to the manifolds actually observed in LLMs.
- Weakness: a static per-document latent resembles topic/style, so a skeptic can call the result
  genre detection. Variant C answers this.
- Extra control available here: engineered mismatch (settings semantically far but with similar
  text effects, or vice versa) — student should follow the data, not the semantics.

### C. Concept manifold + dynamics (both dials)

Variant B's settings with variant A's machinery: z walks the hue ring within a document (HMM we
design). Both emission overlap (M) and transition structure (T) shape the belief geometry; the
ideal reader must decay stale evidence, so passing the tracking tests demonstrates filtering, not
topic detection.

### Cross-cutting design axes (decide per branch)

- Static z per document vs. dynamic z within documents. (Code should take steering as a
  per-token input either way, so dynamics are a config change.)
- Transition granularity if dynamic: per token vs. per sentence.
- Deliberately mixed steering (e.g. half red / half orange): not for the main corpus, but useful
  as ambiguous-prefix probes.

## Test battery

Hypothesis 1 — **the student carries the ideal reader's distribution over z**:

1. *Readout.* Linear map from student hidden states to ground-truth posterior vectors; fit on
   half the documents, test on the rest. Must recover the full graded distribution, not just the
   argmax.
2. *Calibration.* Decoded confidence tracks ideal confidence position-by-position (flat early,
   sharpening with evidence; ambiguous prefixes decode to genuinely split posteriors).
3. *Update dynamics.* Decoded per-token belief changes match computed Bayes/filter increments;
   spliced counter-evidence moves the decoded posterior back by the right amount.
4. *Probe-free behavioral test.* The student's own next-token predictions should equal the
   posterior-weighted mixture of the teacher's per-setting predictions — computable exactly,
   compared by KL. Falsifiable from outputs alone, no probes.

Hypothesis 2 (only if H1 passes) — **the arrangement of z-values has the predicted geometry**:
Gram/PCA of the student's per-setting representations vs. the geometry implied by measured M and
designed T (Fourier ring for circulant T in variant A; measured-confusability manifold in B/C).

Controls (all variants): student trained on unsteered teacher text (readout must fail); shuffled
state–posterior pairing; multiple structures on identical emissions (A) or engineered mismatch (B).

Two further controls suggested by the literature (see prior-work section):

- **Dilution.** Mix the steered corpus into ordinary text at some rate (e.g. 10%) and ask whether
  the geometry still forms. Answers the strongest expected objection ("steered LLM text is not
  real pretraining data") — if the geometry survives dilution, the bridge to real pretraining
  corpora is made. Flagship extension for a strong version of the paper.
- **Paraphrase / re-tokenization.** Subliminal-learning work shows steering signals can travel
  through non-semantic token statistics and die under paraphrase or tokenizer change. Testing
  whether the student's latent survives paraphrasing tells us whether the signal lives in meaning
  or in surface statistics — either answer is informative, and reviewers from that community will
  ask. (Note: for our ground truth this distinction is not a confound — the posterior is computed
  from teacher likelihoods, so any signal actually in the text is legitimate evidence about z.)

## Known tensions (flagged, unresolved)

- **Emission strength vs. interesting geometry.** If one token reveals z, posteriors live at the
  vertices and the manifold is unpopulated. Needs a pilot sweep: steering strength vs. average
  posterior entropy.
- **Orthogonal vectors ≠ orthogonal text effects.** Random SAE latents can still push text in
  correlated directions. Never assume M; always measure it (core-loop step 3).
- **Ground-truth cost.** Posterior computation needs K teacher forward passes per document; sets
  practical bounds on K and corpus size.
- **Leakage filtering.** Safe for the posterior, but aggressive filtering shrinks the corpus and
  could select for weak-evidence text; monitor the tradeoff.

## Open decisions (to make together, per branch)

- Teacher model + which SAE (pretrained SAEs available? which layer?).
- K (number of settings), steering strength, corpus size, document length.
- Student architecture and size; tokenizer (teacher's or fresh).
- Prior over z; transition matrix parameters (stay probability, graph).
- Which variant runs first (current lean: A), and whether B/C reuse the same teacher.

## Execution log — variant A casting funnel (Aug 2026)

Teacher: Gemma-2-2B base + Gemma Scope 16k residual SAE, layer 12. Funnel order 1 → 2.5 → 2 → 3
→ 4 (the causal screen runs before the co-activation check because it is the harsher filter).
Selection principle throughout: filter for *usability* (fires, causal, fluent), never for what
latents mean or how they relate — hash states must stay arbitrary.

- **Stage 1 — cheap filters** (4M pile tokens): of 16384 latents, cut 6 dead, 1932 too common
  (>1% of tokens), 4049 formatting-dominated, 13 early-position-dominated → **10838 survivors**;
  only 32 decoder-cosine near-duplicate pairs (>0.7). `src/stage1_filters.py`.
- **Stage 2.5 — causal power screen** (500 random survivors, seed 0; clamp at layer 12 to
  {2,5,10}× mean firing activation over 256 pile prefixes; final-position KL vs. clean):
  median KL@5× 0.038, dose–response monotone for **all 500** — single-layer clamping steers,
  no multi-layer fallback needed. Cuts: KL@5× in [0.05, 2] and steered entropy within 0.3 nats
  of clean (2.38) → **163-latent causal pool** (306 inert, 18 over-loud, 13 entropy-degenerate).
  `src/stage2_5_causal.py`, `src/stage2_5_apply_cut.py`.
- **Carrier, not caption.** The upper KL cut is a design principle, not just hygiene: as in
  Subliminal Steering (2604.25783), where the trait was recoverable from accumulated statistics
  of generated number sequences rather than any single giveaway token, the state z should be a
  statistical tilt the reader integrates over many tokens. A latent with KL > ~2 nats on
  arbitrary natural prefixes overrides context instead of flavoring it — states become trivially
  separable per token, posteriors sit at the vertices, and the belief geometry we want to study
  never gets populated. (Our regime is gentler than theirs — the text stays natural prose — but
  the readout philosophy is the same.) If strong and gentle latents both reach casting, pick the
  final 8 at similar evidence rates so no state is louder than the others.
- **Stage 2 — co-activation check** (2M pile tokens): pairwise co-firing of the 163 pool latents
  vs. independence. Flagged 818/13203 pairs (lift > 5 or Jaccard > 0.1); median pair is
  *anti*-correlated (lift 0.56). Even under the strict rule (avoid every flagged pair) a mutually
  clean set of 48 latents exists — 8 hash states are easily cast. `src/stage2_coactivation.py`.
- **One-hot clamp (decided).** Generation-time intervention: the active state's latent clamped at
  strength s, all other cast latents ablated to 0 — "exactly one state active" holds in the
  residual stream, not just the bookkeeping (natural mid-document firing of an off-state latent
  would otherwise be spurious evidence the ground-truth posterior can't see). Strength s is an
  empirical dial set by the stage-3 sweep (s ∈ {1,2,3,5}× mean firing activation) against
  posterior entropy: 1× is likely below the evidence floor (median prefix KL@2× is 0.011 —
  natural magnitude injected off-context is a whisper), 5–10× was a screening probe, and
  autoregressive feedback during generation argues for the low end (~2–4×).
- **Stage 3 — generation audition** (48 latents x {1,2,3,5}x, then 20 audible x {5,8,12}x, 20-40
  docs/cell, 300 tokens): fluency clean everywhere (NLL ratio <= 1.10 at 5x, distinct-2 flat);
  1-2x at classifier chance (measured per-token KL at 1x: 0.004 nats — below any reader's floor);
  KL grows quadratically with clamp strength (Fisher). `src/stage3_audition.py`, `stage3_analyze.py`.
- **Stage 4 — pilot + ideal-reader calibration** (static-state docs, exact posterior over the cast
  via 8 clamped replays/doc): median tokens to 90% confidence = 98 / 34 / 13 at 3x / 5x / 8x; end
  accuracy 0.964 / 1.000 / 1.000. **Verdict: 5x, dwell ~100 tokens (p_stay ~0.99)** — graded
  beliefs for the first ~third of a dwell, ~10 transitions per 1024-token context. Cast of 8:
  368, 1220, 2404, 2970, 6172, 10615, 10621, 13931 (reserves 9508, 10013); identities and sample
  texts in `results/stage4/casting_report.md`. Selection caveat (flagged, accepted for now): the
  final-8 shortlist used a bag-of-words classifier, which biases toward word-choice carriers;
  an exact-likelihood confusability re-selection is the upgrade if casting is ever revisited.
- **Corpus generation** (done, `src/corpus_generate.py`): 100k ring documents + 25k unsteered
  control, 1024 tokens each (BOS + unique 12-token Pile opener + 1011 steered tokens). Hidden path
  pre-sampled per document from the ring chain (p_stay 0.99, uniform neighbour), one-hot clamp at
  5x mean_act follows the path per token in a custom KV-cache decode loop (opener prefill clamped
  with the initial state). Chunked/resumable output; measured 4.4k tok/s/GPU at batch 224 on the
  4 A100s (~2 h total). Smoke test verified ring-adjacent transitions (~10/doc) and fluent text.
- **Ideal-reader ground truth** (done, `src/corpus_score_posterior.py`):
  eval split = first 1250 docs of each shard's last ring chunk (5k docs, held out from student
  training). Per doc: 8 whole-sequence-clamp replays (the stage-4 instrument) give per-token
  emissions; the HMM forward algorithm with the known ring T turns them into the filtered
  posterior the student is compared against. Exact filtering is intractable (emissions condition
  on the whole past clamp path through the KV cache, not just the current state), so this
  static-hypothesis reader is the operational definition; a 9th replay under the true per-position
  clamp path reproduces generation exactly and quantifies the approximation.
- **Filter quality + vocab** (`src/corpus_checks.py`, `results/corpus/corpus_checks.json`):
  per-token filter argmax accuracy 0.58 overall; median re-convergence 22 tokens after a
  transition (q25 7, q75 46 — faster than the ~34-token cold start, the ring prior helps);
  13.7% of transitions unresolved within 150 tokens (mostly dwells too short by design);
  late-dwell accuracy 0.70, explained by the measured static-approximation gap (−0.039
  nats/token mid-dwell, −0.089 post-transition). This attenuates the yardstick (adds noise)
  but cannot fake a positive result; upgrade path is true-past-context emissions. Vocab:
  Gemma tokenizer has 212,955 distinct ids in the corpus; top 32,768 cover 96.6% of training
  tokens → compact vocab 32,769 with id 0 = unk. OOV tokens are 3.4% of tokens carrying 7.2%
  of state evidence (~0.6 of ~9 nats/dwell) — accepted and documented.
- **Student training** (done, `src/train_student.py`): two identical 110M-param (85M
  non-embedding) Llama-style students from scratch — 12 layers, width 768, context 1024, tied
  embeddings over the 32k vocab; batch 128, 2812 steps (4 ring epochs), cosine 3e-4 → 3e-5,
  bf16, checkpoints every 250 steps. Ring student: heldout loss 3.42 → 3.23 (steps 1250 →
  2812), still improving at the end. Control student (same steps on the 25k unsteered docs =
  ~16 epochs): loss on the same held-out ring docs plateaued ~3.6 and drifted to 4.10 — it
  never sees steered text. One checkpoint was corrupted by a transient quota/FS error
  mid-save; trainer now does verified saves (tmp + load-check + rename) and has a --resume
  flag; ctrl finished by resuming from step 2500.
- **Quick mid-training geometry check** (`src/quick_geometry_check.py`): at ring ckpt 1000
  (~1/3 of training), layer 6, the 8 mid-dwell class means already sit in exact ring order in
  their top-2 PC plane (68% of class-mean variance), pairwise distances correlate 0.655 with
  ring graph distance, and a ridge probe reads the current state at 25.8% vs 12.5% chance
  (ideal-reader ceiling 57.6%). At ckpt 2000: same geometry, probe R² 0.088 → 0.109, acc
  26.7% — the ring locks in early and is stable while belief tracking keeps sharpening.
- **Probe sweep** (done, `src/probe_sweep.py`, `results/probe/{ring,ctrl}_sweep.json` + png):
  ridge probe residual → 8-dim filtered posterior, 13 layers × 12 checkpoints × both students,
  1600/400 doc-level split, shuffled-pairing control. Ring student (final ckpt): R² rises
  monotonically with depth to 0.191 at the last layer, state accuracy 29.6% (chance 12.5%,
  ideal-reader ceiling 58.1%), ring order in the top-2 PC plane at every layer 1–12, distance/
  ring-distance corr up to 0.765. Control student on the same eval docs: R² caps at 0.066,
  peaking mid-stack (L8) and *decaying* toward the output; over training its probe performance
  declines after step ~1250 (0.078 → 0.066) as it specialises on unsteered text, while the ring
  student's climbs monotonically (0.117 → 0.191). Shuffled-pairing probes ≈ 0 everywhere.
  Three-way dissociation: magnitude (≈3× R², over-chance accuracy 17.1 vs 9.3 pts), depth
  profile (computed toward the output vs generic mid-stack features), and training dynamics
  (improving vs declining). Caveat: the control also shows partial ring geometry (corr ~0.69)
  — state is partly readable from text content/context by any reader; the claim rests on the
  margins above, not on the control being at zero.
- **Belief-geometry analysis** (done, `src/belief_geometry.py`,
  `results/probe/belief_geometry.json` + `belief_ring.png`), final checkpoints, best layers
  (ring L12, ctrl L8):
  - *Time-in-dwell curves (the integration test):* ring-student probe accuracy climbs with
    tokens since the last transition — 0.16 / 0.18 / 0.23 / 0.29 / 0.46 for offsets 0-10 /
    10-25 / 25-50 / 50-100 / 100+ — tracking the shape of the ideal reader's own curve
    (0.17 / 0.30 / 0.47 / 0.65 / 0.83) at roughly half its height. The control climbs far
    more slowly (0.15 / 0.16 / 0.19 / 0.22 / 0.30). The ring advantage grows from ~1 pt right
    after a transition to ~16 pts deep in a dwell: evidence *accumulation*, not static topic
    readout, and deep in dwells the ring student reaches 56% of the ideal reader's accuracy.
  - *Graded position:* circular correlation between the probe's belief angle and the true
    posterior's angle (confident tokens): ring 0.33 vs ctrl 0.17. Tokens whose true belief
    splits mass over two neighbouring states land between those states' centroids 40% of the
    time vs a 30% confident-token baseline for the ring student (+10 pts); ctrl +1.7 pts.
    Both directionally right, modest in size — per-token readout is noisy.
  - *Uncertainty:* probe → posterior entropy R² 0.087 (ring) vs 0.061 (ctrl); weak, the
    reader's uncertainty signal is only marginally represented.
- Remaining: writeup / synthesis of variant A; optional upgrades: true-past-context
  emissions for a tighter ideal reader, exact-likelihood confusability re-selection of the
  cast.

## Relation to prior work (boundary)

Shai 2024 / Shai 2026 / Piotrowski 2025: synthetic token streams, known HMM, no natural language,
no steering, no teacher–student loop. Bayesian Attention Trilogy: in-context latents in prompts,
no corpus-level latent, no retraining; names this experiment's setting as open. Xie et al.
(ICL as implicit Bayes): pretraining on HMM-generated *synthetic* data. Physics-of-LMs /
TinyStories: controlled corpora but no hidden per-document latent with computable posterior. The
full loop here (steered teacher → secret-z corpus → fresh student → comparison against exactly
computable posteriors) appears unclaimed.

### Where variant A sits relative to the factored-representations paper (2602.02385)

Their conditional-independence condition is about beliefs: given the observed tokens, the
posterior over latent factors must factorize (emission coupling between factors is allowed;
their breaking case makes observations ambiguous about *which* factor explains them). Our world
satisfies this trivially: the teacher is autoregressive, so all its "other latents" (topic,
style, ...) are deterministic functions of the visible text — the corpus is an HMM with exactly
one uncertain hidden variable, z. One-factor uncertainty cannot entangle. Consequences:

- Real residual concern is *context-modulated evidence rate* (the clamp's effect depends on
  context → heteroscedastic posterior trajectories), not broken factorization. Check variance
  around measured M, not just its mean.
- Running multiple simultaneous designed latents WOULD make factorization live: ambiguous
  emissions create cross-factor belief correlations — their §4.3 predicts a lossy-factored
  attractor first. Reproducing that in natural language is a candidate follow-up experiment.
- Nuance (from discussion): the teacher mechanically entangles z with its internal features —
  that entanglement IS the channel — but those internals are deterministic functions of
  (prefix, z), so they add no independent uncertainty; the world's posterior stays
  one-dimensional over K states. Two consequences that DO carry the worry forward:
  (i) the student computes z-beliefs *from* content features, so whether the ring gets a
  dedicated subspace ~orthogonal to language machinery or lives in borrowed content
  coordinates is an open empirical question (either answer is informative re: FWH in natural
  language) — test, don't assume; (ii) design rule: anything that conditions generation
  (seed prompts etc.) must be visible in the document or constant across the corpus —
  randomized-and-hidden conditioning is a second uncertain latent and reintroduces exactly
  the cross-factor ambiguity we're avoiding.
- Free prediction that survives: factored structure forms early — checkpoint the student
  densely in early training and date the ring's appearance.

### Scoop check, Aug 2026 (both papers read in full)

- **Sarfati et al. 2026, "The Shape of Beliefs" (Goodfire, [2602.02315](https://arxiv.org/abs/2602.02315)).**
  Posterior manifolds in *pretrained* Llama over parameters (μ, σ) of number distributions fed
  in-context; linear field probes tile the curved manifold; belief dynamics under distribution
  switches; linear steering moves representations off-manifold while manifold-aware steering
  preserves the belief family. No training from scratch, no controlled corpus latent. Their
  limitations section: analysis "constrained to numerical settings; extending... to broader
  natural language processing contexts remains open." Relevant methodology for us: linear field
  probes as the readout formalism, intensive PCA for probability vectors, ideal-observer
  comparison, off- vs on-manifold steering.
- **Morgulis & Hewitt 2026, "Subliminal Steering" ([2604.25783](https://arxiv.org/abs/2604.25783)).**
  Steering vector injected into a teacher during generation of innocuous number sequences; a
  student (same base model, LoRA fine-tune) inherits both the behavioral bias and the steering
  direction itself, localized to the steered layers; the vector is recoverable from the data
  alone. Note: their vector is *learned* (trained to elicit a target phrase) and injected at
  every token position across nearly all layers ([2, L−2]) — not an SAE direction at a single
  layer. So their transfer result certifies the channel for an all-layers signal; whether a
  single-layer SAE signal transfers as cleanly is for our pilot to confirm (fallback ladder:
  stronger clamp → adjacent-layer window → learned per-setting vectors à la this paper). One fixed bias, not a structured latent; fine-tuned same-base student, not fresh; no
  posterior, no geometry. For us it is *feasibility evidence*: the steered-teacher →
  generated-text → student channel transmits steering-induced structure robustly and precisely.
  It also motivates the paraphrase/re-tokenization control above (subliminal transfer dies across
  tokenizers — Cloud et al. 2025).

Running tally: three groups have independently named this experiment's setting as the open
problem — Shai et al. 2024 (beyond toy processes), the Bayesian Attention Trilogy
("natural-language wind tunnels"), Goodfire 2026 ("beyond numerical settings"). No paper found
occupying the intersection: from-scratch training × natural language × designed latent structure
× exactly computable posterior × geometry as dependent variable.

### Stakes / positioning (assessment, Aug 2026)

The field has belief-geometry results on synthetic token streams and correlational manifold
observations in real LLMs, with nothing connecting them. This experiment is the connection, and
it is *constructive*: design a geometry, inject it into natural-language training data, watch it
appear in a fresh model — then change the design on identical emissions and watch the geometry
change with it. Causal evidence for where feature geometry comes from. Expected pushback:
(1) steered LLM text is not real pretraining data (→ dilution control); (2) "unsurprising given
Shai 2024" (→ three limitations sections say otherwise; multi-geometry + spectral match is a new
kind of claim); (3) execution risk in steering quality and the emission-strength/entropy tension
(engineering, not conceptual — pilot sweep first); (4) compute (TinyStories-scale student
suffices; ground-truth scoring at K teacher passes per document is the real cost, arguing for
modest K). Timing: several well-resourced groups are circling this gap; the window is open now
but plausibly not for more than about a year.
