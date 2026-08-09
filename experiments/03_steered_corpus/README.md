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

## Relation to prior work (boundary)

Shai 2024 / Shai 2026 / Piotrowski 2025: synthetic token streams, known HMM, no natural language,
no steering, no teacher–student loop. Bayesian Attention Trilogy: in-context latents in prompts,
no corpus-level latent, no retraining; names this experiment's setting as open. Xie et al.
(ICL as implicit Bayes): pretraining on HMM-generated *synthetic* data. Physics-of-LMs /
TinyStories: controlled corpora but no hidden per-document latent with computable posterior. The
full loop here (steered teacher → secret-z corpus → fresh student → comparison against exactly
computable posteriors) appears unclaimed.
