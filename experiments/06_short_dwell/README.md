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

## s = 2 pilot (same day): strength moves the needle — at a fluency price

Same 64-doc recipe, --mult 2.0:

| mult | dwell | clean-model NLL | oracle | ring | no-map | forget | none | **wiring value** | tracking value | acc ring | mean max belief |
|------|-------|-----------------|--------|------|--------|--------|------|------------------|----------------|----------|-----------------|
| 2.0 | 20 | 3.573 | 2.838 | 2.977 | 3.005 | 3.043 | 3.300 | **0.0281** | 0.3237 | 0.831 | 0.830 |
| 2.0 | 10 | 3.646 | 2.884 | 3.083 | 3.120 | 3.158 | 3.332 | **0.0370** | 0.2485 | 0.724 | 0.728 |

- Wiring value rises 5–6x (0.006 → 0.028–0.037 nats/token); best so far is
  s = 2, dwell 10. Proper Bayes vs the forgetful reader is now worth 0.075
  nats there — integration finally pays.
- Cost: clean-model NLL of the generated text jumps 2.43 → 3.6 (+1.15
  nats/token), and samples visibly degrade (fragmented lists, token salad
  stretches, script-switching) — s = 2 is noticeably off-manifold, echoing
  exp03's s = 5 lesson in milder form.
- Candidate sweet spot to test next: s = 1.5 at dwell 10/20.

## s = 1.5 pilot (same day): the middle ground

| mult | dwell | clean-model NLL | wiring value | tracking value | acc ring | mean max belief |
|------|-------|-----------------|--------------|----------------|----------|-----------------|
| 1.5 | 20 | 3.049 | **0.0171** | 0.1547 | 0.719 | 0.718 |
| 1.5 | 10 | 2.901 | **0.0169** | 0.0992 | 0.581 | 0.585 |

- Wiring value ~0.017 at BOTH dwells (~3x the s=1 baseline; dwell barely
  matters at this strength). Dwell 20 dominates dwell 10: same wiring pay,
  more total tracking pay, far more confident reader (0.72 vs 0.59).
- Text: intermediate. Episodic degeneration (fragment stretches, one
  "course of coarse" loop) that the model RECOVERS from, vs s=2's terminal
  collapse. Still recognizably language most of the time.
- Full 7-row table in results/pilot/readers.json.

## Full corpus + eval scoring (Aug 13)

Settled recipe: s = 1.5, p_stay = 0.95, 256-token docs (BOS + 12-token
Pile opener + 243 generated). 400,000 docs x 8 shards = 102.4M tokens
(matches exp05's training-token count), generated on 8 seahorse A6000s at
~3,100 tok/s each (`src/corpus_generate.py`, seed 6). Needed
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — the default allocator
fragments and OOMs on the once-per-batch prompt-prefill logits spike after
~2k docs. Integrity checked: exactly 400,000 docs / 102,400,000 tokens,
mean 12.06 switches/doc, every switch +-1 on the ring.

- Vocab (`src/build_vocab.py`): top-32,768 ids cover 96.07% of training
  tokens (chunk009 excluded — it holds the eval split).
- Eval split scored exactly (`src/corpus_score_posterior.py`, 8 GPUs):
  first 625 docs of each shard's chunk009 = 5,000 docs. Reader argmax
  accuracy 0.755–0.765 per shard (mean 0.762) — matches the pilot's ~0.78
  truncated-doc anchor. Posteriors in `results/posterior/eval_shard{i}.npz`.
- Student training (`src/train_student.py`, tigerfish GPUs 0+1): exp05
  architecture/schedule, batch reshaped 512x256; running at ~175k tok/s.
