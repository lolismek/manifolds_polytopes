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

## Student results: the ring is learned (Aug 14)

Training: exp05-identical 110M student, batch reshaped to 512x256 (same 131k
tokens/step), 4218 steps on 2 tigerfish A100s. Final heldout loss 3.632.
(Two crashes en route, both the shared home hitting its disk quota during
checkpoint saves — freed ~14G of stale .venvs and resumed from the last good
checkpoint each time; saver now retries 6x30s.)

Classical probe battery (`src/probe_sweep.py`, `src/probe_concat.py`,
exp05's exact design; ridge probes residual -> exact 8-dim posterior,
4000/1000 doc split, shuffled-pairing control ~0 throughout; exact-reader
ceiling on these tokens: acc 0.7615):

| probe | exp06 ring | exp06 ctrl* | exp05 ring | exp05 ctrl |
|---|---|---|---|---|
| best single layer R2 | **0.488** (L12) | 0.187 (L3) | 0.150 (L12) | 0.078 (L3) |
| best single layer acc | **0.557** | 0.364 | 0.320 | 0.258 |
| concat (13 layers) R2 | **0.543** | 0.248 | 0.206 | — |
| concat acc | **0.577** | 0.412 | 0.326 | — |
| class means in ring order | **L9-11, exact** | no layer | no | no |
| dist-vs-ringdist corr | **0.45** | 0.015 | 0.30** | 0.12 |

*ctrl = exp03's unsteered student probed on exp06 eval docs; its rise vs
exp05 (0.078 -> 0.187) shows short dwells make the posterior more
predictable from surface flavor alone — the ring-vs-ctrl GAP is the signal,
and it widens from ~0.07 to ~0.30 R2.
**exp05's 0.30 was later shown to be neighbor-carryover artifact
(firstdwell test collapsed it to 0.04).

Key findings:
- **Evidence integration**: concat-probe accuracy climbs with
  tokens-since-switch (0.30 -> 0.81 by lag 50+), paralleling the exact
  reader's own curve (0.43 -> 0.94). In exp05 this curve was flat (static
  flavor readout). The student updates its belief token by token.
- **Ring geometry**: the 8 mid-dwell class means sit in EXACT ring order in
  layers 9-11 (dist/ring-dist corr 0.42-0.45, top-2 PC variance ~0.52-0.56);
  layer 12 is one transposition off. The order appears by step 500 and is
  stable thereafter, while decoding R2 keeps climbing to a ~step-3000
  plateau (0.30 at step 250 -> 0.49).
- Training trajectory (best-layer R2): 0.299 / 0.419 / 0.478 / 0.486 /
  0.488 at steps 250 / 1000 / 2000 / 3000 / 4218; best layer is 12 from
  step 500 on.

Verdict: exp06's 3x wiring incentive (plus 2.5x tracking incentive) flipped
the result — the student now tracks the belief state at ~2/3 of the exact
reader's ceiling and represents the 8 states on a ring in its late layers.
Whether the WIRING (transition map) itself is used — vs ring geometry
inherited from tracking correlations — is for the follow-up analyses
(first-dwell control, post-switch behavior, latent-direction geometry).

Results: `results/probe/{ring,ctrl}_{sweep,concat}.json` (also in-repo).

## First-dwell control + read-out geometry (`src/firstdwell_probe.py`)

Tokens before each doc's first switch only (99k tokens, ~20/doc; no
previous dwell in context, so neighbor-carryover cannot operate). Exact
reader ceiling there: acc 0.819. All 13 layers, both students; probes
trained on posterior AND on one-hot true-state labels (exp05's lesson: the
posterior target itself carries neighbor mass, one-hot columns are the
clean geometry test).

- **Tracking survives, fully**: ring R2 0.478 at L12 on first-dwell tokens
  (all-token value: 0.488; exp05 first-dwell: 0.158), acc 0.592 / one-hot
  0.581. Ctrl: R2 0.197. The belief decoding is NOT a carryover artifact.
- **Class-mean circle does not survive as-is**: first-dwell class means
  lose exact ring order everywhere; dist/ring-dist corr drops 0.45 ->
  0.15-0.17 at L9-12 (ctrl: ~-0.07; exp05 ring collapsed to 0.04). So the
  clean mid-dwell circle largely reflects belief smear across switches —
  the activation ring is a property of the tracking DYNAMICS, not a static
  arrangement of the 8 flavors.
- **But ring adjacency is imprinted in the read-out directions**: the
  one-hot probe columns at L9-12 show a STRONG NEGATIVE dist/ring-dist
  corr, -0.65..-0.72 (ctrl: +0.15..0.37; exp05 ring: -0.11). Ring-NEIGHBOR
  states get maximally separated read-out directions — exactly what a
  decoder needs when the representation mixes neighbors together. The
  student's first-dwell representations carry ring-adjacency structure
  that the control has none of; it shows up sign-inverted in the
  discriminative view.

Figure: `results/probe/firstdwell.png` (class means + one-hot columns in
2D, ring L10/L12 vs ctrl L3). Data: `results/probe/firstdwell.json`.
