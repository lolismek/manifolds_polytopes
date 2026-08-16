# exp07 — uniform-jump control: is the WIRING represented, or just tracking?

exp06's open question: the ring student tracks the belief state and its
late layers carry a genuine ring plane — but is the transition map itself
represented, or could the ring geometry be inherited from tracking
correlations plus flavor exposure? The exp03 unsteered control can't
answer that: it was never exposed to the steered flavors at all, so its
deficits confound "didn't learn the structure" with "never saw the topics."

This experiment trains the exposure-matched structural control: a second
student on a corpus identical to exp06's in every respect (same 8-state
cast, s = 1.5 multi-layer add, clean-cache physics, p_stay = 0.95,
256-token docs, same openers, same token budget, same architecture/
schedule) EXCEPT the transition on a move: uniform over the 7 other states
instead of +-1 on the ring. Same dwell distribution, same incentive to
track the current state, no map to learn. Seed and rng draw-order are kept
so switch TIMES are token-for-token identical to the ring corpus; only the
destinations differ.

Predictions that make or break the wiring claim:
- tracking survives in the uniform student (decent probe R2, an
  integration curve that climbs with tokens-since-switch);
- the ring signatures vanish: Fourier f1 fraction at chance in cast-ring
  order, no neighbor band in the probe-direction cosine heatmaps, no
  whitening-dial swing (-0.8 -> +0.7) — those stay exclusive to exp06's
  ring student.

## Pipeline

1. `src/pilot_uniform.py` + `src/pilot_analyze.py` — 224-doc pilot:
   confirm map_value ~ 0 by construction and tracking value / reader
   accuracy in the ring corpus's ballpark before committing 100M tokens.
2. `src/corpus_generate.py` — 400k docs x 256 tokens = 102.4M tokens,
   8 shards on seahorse (tag `uniform`).

## Pilot (2026-08-16): green light

224 docs, exp06 recipe with uniform jumps, vs the exp06 eval shard 0
re-analyzed with the same reader set (`results/pilot/readers.json`; losses
nats/token; "true model" = the corpus's own transition matrix, "other map"
= the other corpus's):

| corpus | clean NLL | oracle | true model | other map | none | map value | tracking value | acc true | mean max belief |
|---|---|---|---|---|---|---|---|---|---|
| ring (exp06) | 3.348 | 2.770 | 2.889 | 2.909 | 3.095 | **0.0205** | 0.206 | 0.758 | 0.758 |
| uniform (exp07) | 3.297 | 2.717 | 2.859 | 2.899 | 3.038 | 0.0400 | 0.178 | 0.701 | 0.699 |

- Tracking incentive survives at ~87% of the ring corpus's (0.178 vs
  0.206); reader accuracy 0.70 vs 0.76 — slightly lower as expected (after
  a switch there are 7 candidate destinations, not 2).
- Fluency matches (clean-model NLL 3.30 vs 3.35); sample text is the usual
  s = 1.5 quality.
- The uniform corpus's "map value" 0.04 is the loss a reader pays for
  WRONGLY assuming the ring — the ring prior is actively harmful here, so
  any ring geometry found in the uniform student cannot be corpus-induced.
3. Vocab: exp06's `vocab32k.npz` reused unchanged (probe comparability);
   coverage re-checked on the new corpus.
4. `src/corpus_score_posterior.py` — exact posterior for the eval split
   (first 625 docs of each shard's chunk009), forward algorithm with the
   uniform transition matrix.
5. `src/train_student.py` — exp06-identical 110M student, 512x256 batches,
   4218 steps.
6. Probe battery two ways: exp06's full battery on the uniform student's
   own eval docs, plus cross-probing on the RING eval docs (the
   exposure-matched surface-flavor floor for exp06's numbers).
