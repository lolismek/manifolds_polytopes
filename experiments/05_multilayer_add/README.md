# exp05 — multi-layer additive steering

exp04's exact reader is too weak (min pairwise margin 0.081 nats/token vs
per-token noise std ~0.41 → late-dwell belief stuck at 0.5–0.65). exp05 keeps
clean-cache exactness and raises the evidence rate: pure additive injection
(no clamp/ablation) of `s · mean_act_i · (rms_l/rms_12) · W_dec[i]` at layers
12..23 (home layer → L−2), sampled position only. One knob: s, in mean-act
units. Cast, openers, generation/scoring physics reused from exp04.

## Step 1 — physics pilot

Gate: min pairwise margin ≥ ~0.3 nats/token (≈3x exp04) with NLL ratio ≤ 1.15.

```
python addsteer.py --device cuda:0                          # rms calibration (once)
python verify_identity.py --device cuda:0 --mult 2          # generator/scorer identity
python pilot.py --phase gen   --shard i --n_shards 4 --device cuda:i
python pilot.py --phase score --shard i --n_shards 4 --device cuda:i
python pilot_analyze.py
```

s ∈ {1, 2, 4} × cast of 8 × 6 docs × 300 tokens. If no s passes → next rung is
learned per-setting vectors (Subliminal Steering style).

## Execution log

- **Calibration** (`results/calib/layer_rms.json`): residual rms grows 3.4x
  from layer 12 to 23; per-layer scales frozen.
- **Identity check** (`results/verify/identity_check.json`): same-shape
  rescoring bit-exact (max |Δ| = 0.0); expanded-batch differences are the
  same zero-mean bf16 batch-shape noise exp04 documented (median 0.001, max
  0.18 nats). Side finding: multi-layer compounds hard — s=2 gives KL
  0.89/token — so the sweep was re-centered to {1, 1.5, 2}.
- **Pilot** (2 GPUs, ~20 min; `results/pilot/`), 8 cast latents x 6 docs x
  300 tokens per s, all docs scored under all 8 hypotheses:

  | s | KL/tok | NLL ratio | dent | distinct2 | min margin | med margin | noise std |
  |---|--------|-----------|------|-----------|------------|------------|-----------|
  | 1.0 | 0.154 | 1.05 | +0.04 | 0.788 | 0.168 | 0.279 | 0.93 |
  | 1.5 | 0.413 | 1.29 | +0.36 | 0.737 | 0.247 | 0.724 | 1.68 |
  | 2.0 | 0.554 | 1.32 | +0.33 | 0.569 | 0.154 | 1.184 | 2.23 |

  Gate (min ≥ 0.3 & NLLr ≤ 1.15) not passed as-is, but the physics works:
  median margin at s=1 is already 3.4x exp04's *min* (0.081) at clean fluency
  (NLLr 1.05, natural text on inspection). The min is capped by weak
  *carriers*, not weak injection — latent 1309's whole row is soft (0.17–0.23
  at s=1; still the worst at 1.5x), and 14600 is marginal. Dropping 1309
  lifts the s=1.5 min to 0.387 (passes margin, fails fluency at NLLr 1.29).
- **Verdict:** sweet spot is s ∈ (1, 1.5) with a recast — next step is the
  confusability recast (24-shortlist, multi-layer physics) at s ≈ 1.25,
  replacing weak carriers; expected to clear the gate.
- **Decision (user): keep the exp04 cast, s = 1.0.** Static-Bayes reader on
  the pilot docs settles it: at s=1, median 14 tokens to 90% belief (q75 34,
  all 48 docs resolved, end accuracy 1.00) — stronger than exp03's operating
  point (median 34) despite the 0.168 min margin; Bayes accumulates against
  all 7 competitors at once. s=1.5 converges in ~4 tokens (caption regime,
  rejected). Dwell set from measured convergence, not exp03 comparability:
  **p_stay = 0.98** (dwell ~50 ≈ 3.5x median t90). Ring order chosen to
  maximize the worst adjacent-pair bottleneck margin at s=1: [1309, 10573,
  14600, 6455, 3188, 5615, 2970, 15026] (min adjacent 0.220 vs 0.168 global;
  the weak 1309-2970 pair sits across the ring).
- **Corpus** (done, seahorse, 7 A6000s, ~4 h, `src/corpus_generate.py`):
  100,002 ring docs, 1024 tokens (BOS + 12-token Pile opener + 1011 steered),
  7 shards x 14,286, ~1.1k tok/s/GPU, 490 MB. Integrity-verified: all 42
  chunks load, transitions/doc 20.2 (expected 20.2), all transitions
  ring-adjacent, occupancy uniform (0.124-0.126), sample text natural.
  Control corpus: exp03's unsteered 25k reused (steering off => clean-cache
  generation is ordinary generation). Note: exp03's steered ring corpus was
  deleted from seahorse for quota (control kept); regenerable from committed
  code if ever needed.
- **Eval-split ground truth** (`src/corpus_score_posterior.py`): first 715
  docs of each shard's chunk005 -> 5005 held-out docs, scored under all 8
  hypotheses (exact reader), forward algorithm at p_stay 0.98.
- **Vocab + student training** (concurrent with eval scoring; tigerfish,
  home FS shared with seahorse so no corpus copy needed):
  `src/build_vocab.py` over training chunks 000-004 (87,500 docs; chunk005
  excluded — contains the eval split): top 32,768 ids cover 0.9588 of
  training tokens. `src/train_student.py` = exp04 recipe (110M Llama-style,
  batch 128, cosine to 2812 + floor-LR extension to 4218, ~6 epochs) run as
  2-GPU DDP (global batch unchanged; rank slices of the same shared
  permutation). Control student reused from exp03. Storage (user-approved):
  exp03 ring+ctrl student checkpoints deleted except final ckpt_02812
  (~5.7 GB freed; ring student regenerable only by regenerating the exp03
  corpus).
- **Training done** (tigerfish GPUs 2-3, ~1 h at 162-170k tok/s): final
  heldout loss 3.1837, still creeping down at the end (3.193 -> 3.184 over
  the floor-LR extension); 17 checkpoints (4.4 GB). Launch hiccup worth
  remembering: the seahorse CUDA-runtime surgery had pulled cu13 wheels into
  the shared mp env, overwriting libnccl with a CUDA-13 build (needs driver
  >= 580; tigerfish has 575) -> DDP died at init with "driver insufficient".
  Fixed by `pip install --no-deps nvidia-nccl-cu12==2.28.9` (single-process
  jobs never load NCCL, so nothing else was affected).
- **Exact reader verified on the real corpus** (`src/posterior_analyze.py`,
  5005 eval docs): overall argmax accuracy 0.703, logloss 0.848 vs uniform
  2.079 (exp04: 1.0-1.3). Accuracy is limited by inherent transition lag,
  not mid-dwell stalling — belief in the true state by tokens-since-
  transition: 0.23 (lag 0-4) -> 0.51 (10-19) -> 0.68 (35-49) -> 0.72 with
  argmax accuracy 0.894 at lag 50+ (exp04 stalled at 0.5-0.65 late-dwell
  belief). The 0.72 late-dwell ceiling is the correct HMM equilibrium (the
  filter always hedges 2%/token for an unseen transition), not weakness.
  Corpus has the dynamic range the probe tests need: graded beliefs after
  transitions, saturated beliefs in dwells.
- **Probe sweep** (`src/probe_sweep.py`, exp03's design: ridge probe from
  each layer's residual to the exact posterior, 2800/700 doc split, shuffled
  control, ring geometry of mid-dwell class means; ctrl = exp03's control
  student on the same eval docs). Result — **the student tracks weakly and
  learns no ring geometry**:
  - ring student (final): best R2 0.150 / acc 0.320 at layer 12, vs
    exact-reader ceiling 0.704 (chance 0.125). Plateaus by step ~2250; the
    floor-LR extension adds nothing. Shuffled control ~0 everywhere.
  - ctrl student: best R2 0.078 / acc 0.258 — well above chance. A model
    trained on unsteered text still decodes weak z-correlates because the
    steering channel *is* topical drift.
  - the telling detail: ring and ctrl have essentially the **same angular
    order** of the 8 class means ([1,7,6,2,4,3,5,0]-ish, not the ring
    order), and ring-distance corr peaks at 0.30 (ring) / 0.12 (ctrl). The
    state geometry in BOTH students reflects the latents' natural semantic
    similarity, not the transition matrix — the ring student sharpens the
    control's arrangement rather than reorganizing it.
  - because the reader is now strong and exact, this is a real finding, not
    an artifact: the information is in the text (reader 0.70) but a 110M
    student trained ~6 epochs on 90M tokens recovers less than half of it
    and shows no sign of belief-state ring geometry.
- **Concat probe + lag curves** (`src/probe_concat.py`; Shai et al.'s RRXOR
  lesson — probe the concatenation of all 13 layers, 9984 dims,
  per-feature standardized; final ckpts):
  - concat helps both students the same modest amount (ring R2 0.150 ->
    0.206, acc 0.320 -> 0.361; ctrl 0.078 -> 0.114, 0.258 -> 0.296): some
    state info is written then overwritten mid-stream, but no RRXOR-style
    hidden geometry unlocks — same non-ring angular order in concat space,
    ring-dist corr 0.22 (ring) / 0.08 (ctrl).
  - **the ring student integrates evidence**: probe accuracy by
    tokens-since-transition climbs 0.18 -> 0.48 (reader: 0.21 -> 0.90),
    and the ring-over-ctrl gap GROWS through the dwell (+0.9pt at lag 0-4
    -> +10.6pt at 50+). Replicates exp03's dynamic signature (there +1 ->
    +16pt); deep-dwell student accuracy is 53% of the reader's (exp03:
    56%). Not static topic readout.
  - net verdict: belief *tracking* is real, graded, and integrative in both
    exp03 and exp05 at ~half the ideal reader's height; belief *geometry*
    (the ring) does not form when the test is strict (natural-magnitude
    steering, margin-chosen ring order, controls in place). exp03's exact
    ring order rode on loud 5x steering + neighbor carryover that any
    reader exhibits (its control: corr 0.69).
- **MLP probe + PCA figure** (`src/probe_mlp.py`, `src/pca_shapes.py`):
  nonlinear probe (768-512-512-8 MLP, same split/target, each student's best
  layer) beats linear by only ~0.02 for both students — ring 0.171/acc
  0.326 (epoch 1, overfits after) vs linear 0.150/0.320; ctrl 0.099/0.271
  vs 0.078/0.258 — and stays below the linear concat probe (0.206).
  Nonlinear encoding is ruled out. The 2D picture
  (`results/probe/pca_shapes.png`) matches: in both students the 8
  mid-dwell class means sit in a small clump with state 0 (weak carrier
  1309) offset, the ring-neighbor polygon self-crossing, and within-state
  token clouds far larger than the mean separations. No ring shape in
  either student; the ring student's clump is just slightly more spread.
- **First-dwell geometry test** (`src/firstdwell_geometry.py`, user's
  design; kills the last live geometry signal): class means from ONLY the
  tokens before each doc's first state switch — no previous dwell in the
  context, so the neighbor-carryover artifact cannot operate (~20k
  tokens/state). Ring-distance corr collapses to 0.04 (ring student) /
  -0.05 (ctrl), no ring order. The mid-dwell 0.30-vs-0.12 gap was the
  carryover artifact (which scales with tracking sharpness, hence larger
  for the better tracker). Geometry verdict at s=1 is now clean and final:
  **no ring structure in the student's representation at all**; every
  ring-shaped signal in exp03/exp05 measurements traces to the documents'
  own dynamics leaking through the context window.
- **Capacity-matched-reader check** (is the 0.704 ceiling unfair? it embeds
  the 2B teacher's emission model; a perfectly-Bayesian 110M student would
  sit lower): fit the student's lag curve with (1) the forward algorithm on
  alpha-scaled emissions (evidence-starved exact filter) and (2) a second
  leak parameter (effective p_stay). Neither family fits: even alpha=0.02
  integrates to 0.84 deep-dwell accuracy (a lossless integrator always
  saturates), and the best leaky fit (alpha .02, p_stay_eff .5) matches the
  mean but is flat while the student climbs through the whole dwell
  (0.18 -> 0.48, near-linear). The student is not behaviorally equivalent
  to ANY evidence-scaled/leak-adjusted exact filter. Caveat kept open: a
  capacity-matched reader with *temporally correlated* emission errors
  (systematic misreadings that do not wash out under integration) could
  still look like the student — that is exactly the factored-representation
  lossiness point (z and language are not conditionally independent), and
  distinguishing it needs an actual small-capacity emission model, not a
  noise model.
