# exp06 — does a shorter dwell make the ring worth learning?

Exp05 post-mortem hypothesis: the student never learned the ring (transition
map) because at p_stay = 0.98 knowing the map barely improves next-token
loss. Proposed fix: shorter dwells. This experiment measures the actual
incentive before committing to a new corpus + training run.

## Pilot (2026-08-13)

64 documents per setting, exp05 pipeline unchanged (same cast, s = 1.0
multi-layer add, clean-cache physics, same openers), only p_stay varied.
Every token scored under all 8 state hypotheses (exact emissions); readers
with different amounts of knowledge then compete on next-token predictive
loss (each reader's prediction = belief-weighted mixture of the 8
state-conditional token distributions). Baseline row = exp05 eval shard 0
(first 64 docs), no regeneration needed.

`src/dwell_pilot.py` (generate + score, ~15 min/setting on one GPU),
`src/dwell_analyze.py` (reader arithmetic). Results in
`results/pilot/readers.json`.

| p_stay | dwell | oracle | ring (full Bayes) | no-map Bayes | forgetful (best γ) | none | **wiring value** | tracking value | acc ring | mean max belief |
|--------|-------|--------|-------------------|--------------|--------------------|------|------------------|----------------|----------|-----------------|
| 0.98 | 50 | 2.277 | 2.320 | 2.326 | 2.333 | 2.379 | **0.0060** | 0.0586 | 0.688 | 0.700 |
| 0.95 | 20 | 2.229 | 2.290 | 2.296 | 2.302 | 2.333 | **0.0062** | 0.0426 | 0.555 | 0.555 |
| 0.90 | 10 | 2.265 | 2.339 | 2.345 | 2.348 | 2.365 | **0.0058** | 0.0264 | 0.439 | 0.437 |

(losses in nats/token; wiring value = loss(no-map) − loss(ring) = the pay
for knowing WHICH states are neighbors; tracking value = loss(none) −
loss(ring) = total pay for tracking the state at all.)

## Verdict: shortening the dwell alone does NOT work

- **Wiring value is flat at ~0.006 nats/token at every dwell.** The ring's
  salary does not grow when switches get more frequent.
- **Total tracking value FALLS as dwells shorten** (0.059 → 0.026): faster
  switching means even the perfect reader is chronically uncertain (its
  accuracy drops 0.69 → 0.44, mean max belief 0.70 → 0.44), and an
  uncertain reader can't cash in state knowledge.
- Mechanism: at s = 1 a single token carries weak evidence (identification
  takes tens of tokens). Long dwells: confident beliefs but switches too
  rare for the map to matter. Short dwells: switches frequent but
  undetectable, so the map still doesn't matter. At this evidence strength
  NO dwell makes the ring valuable — the two knobs (evidence per token,
  dwell) must move together, e.g. stronger steering + shorter dwell, so
  switches are both frequent and detectable.

Next candidate (not yet run): quick pilots at s = 2 with dwell 10–20, same
64-doc recipe, to see if wiring value moves.
