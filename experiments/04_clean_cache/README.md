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

(to be filled as stages complete)
