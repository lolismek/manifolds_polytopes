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

## Full corpus (2026-08-16)

400,000 docs x 256 tokens = 102,400,000 tokens, 8 shards on 8 seahorse
A6000s at ~3,100 tok/s each (`src/corpus_generate.py`, seed 6,
expandable_segments allocator as in exp06). Integrity
(`src/corpus_check.py`): exact doc/token counts; 12.10 switches/doc; jump
destinations uniform over the 7 non-self offsets (0.1426-0.1430 each,
offset 0 never); state occupancy flat (0.1247-0.1253); exp06 vocab32k
coverage 0.9600 (ring corpus: 0.9607) — vocab reused unchanged; switch
times verified token-for-token IDENTICAL to the ring corpus in all 80
chunk pairs (same seed + draw order, only destinations differ).

Eval-split exact posterior (`src/corpus_score_posterior.py`, uniform
transition matrix, 8 GPUs): first 625 docs of each shard's chunk009 =
5,000 docs. Reader argmax accuracy 0.700-0.711 per shard (mean 0.704) —
matches the pilot's 0.701 (ring corpus: 0.755-0.765, mean 0.762; the gap
is the expected 7-candidate-destination cost). Posteriors in
`results/posterior/eval_shard{i}.npz`.

Student training (`src/train_student.py`, tigerfish GPUs 0+1, ~178k
tok/s, ~55 min): final heldout loss **3.6287** — matching the ring
student's 3.632 almost exactly. The two corpora are equally hard; the
students differ only in what structure there was to learn.

## Probe results (same day): tracking matched, ring geometry gone

Full battery (`src/*_u.py` — exp06's machinery imported, uniform student
registered in its registry). "Own" = exp07 eval docs + uniform posterior;
"cross" = exp06's ring eval docs + ring posterior (the exposure-matched
floor for exp06's numbers). Shuffled-pairing control ~0 throughout.

### Tracking (the part that SHOULD match — and does)

| probe | uniform own | uniform cross | exp06 ring | exp03 ctrl |
|---|---|---|---|---|
| best single-layer R2 | 0.440 (L11) | 0.422 (L11) | 0.488 (L12) | 0.187 (L3) |
| best single-layer acc | 0.499 | 0.523 | 0.557 | 0.364 |
| concat R2 | 0.498 | 0.484 | 0.543 | 0.248 |
| concat acc | 0.519 | 0.551 | 0.577 | 0.412 |
| acc / reader ceiling | 0.71 | 0.69 | 0.73 | — |
| integration curve (lag 0-4 -> 50+) | 0.23 -> 0.74 | 0.28 -> 0.78 | 0.30 -> 0.81 | flat |
| firstdwell-late R2 / acc | 0.526 / 0.709 | — | 0.541 / 0.722 | 0.236 / 0.479 |

- The uniform student tracks the belief state at the same fraction of its
  reader ceiling as the ring student (0.71 vs 0.73) with the same climbing
  integration curve. Evidence integration is a TRACKING signature, not a
  wiring signature.
- **The cross numbers reset exp06's floor**: an exposure-matched map-free
  student decodes the ring posterior at R2 0.42 / acc 0.52 — far above the
  unsteered ctrl (0.19 / 0.36) that exp06 used as its floor. The
  wiring-specific decoding advantage is the 0.42 -> 0.49 / 0.52 -> 0.56
  margin, much smaller than the raw ring-vs-ctrl gap suggested. The wiring
  claim therefore rests on GEOMETRY, where the dissociation is total:

### Ring geometry (the part that SHOULD vanish — and does)

| signature (own docs, cast-ring order) | uniform | exp06 ring | exp03 ctrl |
|---|---|---|---|
| mid-dwell class means: ring order / corr | no / -0.06 | exact / +0.45 | no / 0.015 |
| firstdwell means f1 fraction (L9-12) | 0.25-0.26, p 0.91-0.99 | 0.36-0.38, **p 0.0004-0.0016** | 0.20-0.22, p 0.7-0.96 |
| one-hot read-out corr (late, L9-12) | -0.01..-0.06 | **-0.78..-0.82** | ~+0.1 |
| one-hot cols f3+f4 p (L9-12) | 0.57-0.93 | **0.0004** | ~chance |
| whitening dial (L10, lam 1e-4 -> 100) | +0.31 -> -0.16, never ring, p_f1 >= 0.08 | -0.81 -> **+0.69, exact ring at lam 10, p 0.0008** | +0.12 -> -0.15, nothing |
| logistic acc / corr (L10) | 0.68 / +0.05 | 0.68-0.70 / -0.2..-0.55 | — |
| cosine heatmap neighbor band | absent | present (lam 10), inverted (whitened) | absent |

- At MATCHED decoding accuracy (logistic 0.68 both students), every ring
  signature exp06 found is absent in the uniform student: no circle in the
  centroids, no anti-ring in the whitened read-outs, no swing across the
  whitening dial, no neighbor band in the heatmaps.
- Cross mid-dwell means corr 0.13-0.17 (vs ring student's 0.45, ctrl's
  0.015): on RING docs even the map-free student's centroids get pulled
  toward cast-ring neighbors — direct confirmation that the carryover
  mechanism exp05 identified operates on any tracking student, and that
  exp06 was right to discount the mid-dwell circle.
- One loose end: the uniform student's raw firstdwell MEANS show weak
  f3+f4 (neighbor-alternating) energy aligned with the cast ring (p
  0.006-0.043 at L9-12, both variants). No corpus mechanism can produce
  cast-ring alignment here; most plausibly this reflects intrinsic
  similarity structure among the 8 SAE latents themselves (the cast's
  ring order was chosen by a similarity criterion in exp03/05). It
  produces no circle and no read-out structure, but is worth remembering
  when interpreting borderline f-mode p-values.

Verdict: **the ring wiring is represented, not inherited.** A student with
identical flavor exposure, identical dwell statistics, identical training
and near-identical heldout loss — differing ONLY in whether the corpus's
transition graph had structure — tracks beliefs just as well but shows
none of exp06's ring geometry, while the exp06 student shows all of it at
p <= 0.0016.

Results: `results/probe/uniform_*.{json,png,npz}` (in-repo).
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
