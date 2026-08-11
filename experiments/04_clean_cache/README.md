# exp04 — clean-cache steered corpus: exact ideal reader

Careful re-run of exp03 variant A (ring HMM over 8 hash SAE latents, steered
Gemma-2-2B teacher, fresh student, belief-geometry probes) with every design
choice revisited. Decisions are logged here as they are made.

## Decisions

1. **Clean-cache steering (headline change).** The clamp is applied only at
   the position being sampled; the KV cache is always built from unclamped
   passes over the realized text. Each token then depends on (text, current
   state) only — the corpus is a true HMM, and the ideal reader's forward
   algorithm is *exact* (exp03's static-hypothesis approximation and its
   −0.04/−0.09 nats/token gap disappear identically). Also closes the hidden
   in-cache influence channel: everything z does to a document is in the
   tokens. Cost: ~2 forwards per generated token; scoring gets cheaper than
   exp03 (one clean replay + per-position single-token hypothesis evals).
   `src/steering.py`, verified in `src/verify_identity.py`.
2. **Casting funnel reuse.** Stages 1 (statistical filters), 2.5 (causal
   screen), 2 (co-activation) are scheme-independent — reused from exp03.
   Starting point: the 48-candidate mutually-clean pool
   (exp03 `stage3/audition_pool.json`).
3. **One global clamp strength.** Rather than per-latent strength
   calibration, the cast is selected among latents that land in a common
   evidence-rate band at the *same* shared strength (cleaner design, one
   fewer per-state degree of freedom). Strength sweep must be redone under
   clean-cache physics (single-position clamp is weaker than exp03's
   compounding clamp): MULTS {3, 5, 8, 12, 18}x. `src/audition.py`,
   `src/audition_analyze.py`.
4. **Exact-likelihood confusability casting** (fixes exp03's flagged
   bag-of-words shortcut): static-state docs for the shortlist at the shared
   strength, scored under every shortlist hypothesis with the exact reader;
   cast of 8 maximizes the minimum pairwise log-likelihood margin with
   matched evidence rates. `src/confusability.py`, `src/cast_select.py`.

## Execution log

- **Step 0 — identity check** (`results/verify/identity_check.json`): scorer
  reproduces the generator's sampling distributions bit-exactly at matched
  batch shape (max |Δ| = 0.0). Expanded-batch scoring shows bf16 batch-shape
  noise only (median 0.005, max 0.18 nats; zero-mean). Side finding: the
  single-position clamp is ~10x weaker per token than exp03's compounding
  clamp (0.0095 nats/token generation KL at 8x) — audition sweep extended to
  {5, 8, 12, 18, 27, 40}x.
- **Step 2 — audition** (48 latents × 6 strengths × 16 docs, 300 tokens,
  clean arm NLL 2.195 / ent 2.184 / distinct2 0.863): dose-response measured
  per cell (one-step KL, NLL ratio, entropy delta). 12x+ visibly degrades
  fluency for many latents.
- **Step 3 — shared strength**: **8x** (most in-band candidates: 26 with
  one-step KL in 0.02–0.30 nats/token, NLL ratio ≤ 1.15, |Δent| ≤ 0.35).
  Shortlist 24 (`results/audition/shortlist.json`); contains 4 of exp03's
  cast (1220, 2970, 6172, 10615).
- **Step 4 — confusability casting** (`results/confusability/`): 24 latents
  × 24 static-state docs × 400 tokens generated at 8x; every doc scored
  under all 24 hypothesis clamps + clean with the exact reader (576 docs,
  4 GPUs). **Cast of 8: [1309, 2970, 3188, 5615, 6455, 10573, 14600,
  15026]**, reserves [5676, 6172, 9311, 16056]. Min pairwise margin
  **0.081 nats/token** (all 56 ordered pairs ≥ 0.081, max 0.167); evidence
  rates vs clean 0.040–0.093 nats/token — note these on-text rates are
  4–10x the one-step audition KLs (text-mediated amplification: state-tilted
  words beget state-tilted words). Overlap with exp03 cast: only 2970 — the
  bag-of-words selection and the exact-likelihood selection disagree almost
  entirely. Full matrix in `results/confusability/casting_report.md`.
