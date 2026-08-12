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
