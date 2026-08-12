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

(nothing yet)
