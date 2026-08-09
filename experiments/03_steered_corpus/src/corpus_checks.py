"""Post-scoring checks: build the 32k vocab, run the OOV evidence check, and
break down the filter's behaviour around transitions.

1. Vocab: token counts over the ring *training* chunks (000-008 per shard);
   top 32768 ids -> compact ids 1..32768, everything else -> 0 (<unk>).
   Saved to results/corpus/vocab32k.npz (top_ids, map).
2. Evidence check: what fraction of the ideal reader's per-token evidence for
   the true state (ll_true_state - mean over states) is carried by tokens
   outside the vocab? Gate for the <unk> decision: should be at or below the
   OOV token share (~4%).
3. Filter diagnostics on the 5k eval docs: re-convergence lag after mid-doc
   transitions, late-dwell argmax accuracy (second half of dwells >= 60 tok),
   and the static-hypothesis approximation gap (ll under whole-sequence true-
   state clamp minus ll under the exact generative clamp path), mid-dwell vs
   the 20 tokens after a transition.

Output: results/corpus/corpus_checks.json
Usage: python corpus_checks.py            (CPU is fine)
"""

import glob
import json
from pathlib import Path

import numpy as np

VOCAB = 32768
N_VOCAB_IDS = 262144

root = Path(__file__).parent.parent / "results" / "corpus"

counts = np.zeros(N_VOCAB_IDS, dtype=np.int64)
train_files = [f for f in sorted(glob.glob(str(root / "ring_shard*_chunk*.npz")))
               if "chunk009" not in f]
for f in train_files:
    counts += np.bincount(np.load(f)["ids"].ravel(), minlength=N_VOCAB_IDS)
top = np.argsort(counts)[::-1][:VOCAB]
top = top[counts[top] > 0]
vmap = np.zeros(N_VOCAB_IDS, dtype=np.int32)          # 0 = <unk>
vmap[top] = 1 + np.arange(len(top))
np.savez(root / "vocab32k.npz", top_ids=top, map=vmap)

res = {"n_train_files": len(train_files),
       "vocab_size": int(len(top)) + 1,
       "train_token_coverage": float(counts[top].sum() / counts.sum())}

ev_in = ev_out = 0.0
n_in = n_out = 0
lags, resolved50, acc_late, gap_mid, gap_trans = [], [], [], [], []
for g in range(4):
    d = np.load(root / f"eval_shard{g}.npz")
    ll = d["ll"].astype(np.float32)                    # (n, 8, T)
    z = d["z"].astype(int)
    post = d["posterior"].astype(np.float32)
    llt = d["ll_true"].astype(np.float32)
    ids = np.load(root / f"ring_shard{g}_chunk009.npz")["ids"][:ll.shape[0]]
    gen_ids = ids[:, 13:]                              # aligned with ll's T axis
    oov = vmap[gen_ids] == 0
    n, K, T = ll.shape
    true_ll = np.take_along_axis(ll, z[:, None, :], 1)[:, 0, :]
    ev = true_ll - ll.mean(1)
    ev_in += float(ev[~oov].sum()); ev_out += float(ev[oov].sum())
    n_in += int((~oov).sum()); n_out += int(oov.sum())
    gap = true_ll - llt                                # static minus exact
    am = post.argmax(1)
    for i in range(n):
        trans = np.nonzero(np.diff(z[i]))[0] + 1
        bounds = np.r_[0, trans, T]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a >= 60:
                acc_late.append(float((am[i, a + (b - a) // 2 : b] == z[i, a]).mean()))
                gap_mid.append(float(gap[i, a + 30 : b].mean()))
        for t0 in trans:
            seg = am[i, t0 : min(t0 + 150, T)]
            hits = np.nonzero(seg == z[i, t0])[0]
            lags.append(int(hits[0]) if len(hits) else 999)
            resolved50.append(bool((am[i, t0 : min(t0 + 50, T)] == z[i, t0]).any()))
            gap_trans.append(float(gap[i, t0 : min(t0 + 20, T)].mean()))

lags = np.array(lags)
ok = lags < 999
res.update({
    "evidence_frac_oov": float(ev_out / (ev_in + ev_out)),
    "token_frac_oov_eval": float(n_out / (n_in + n_out)),
    "n_transitions": int(len(lags)),
    "reconv_lag_median": float(np.median(lags[ok])),
    "reconv_lag_q25_q75": [float(np.quantile(lags[ok], q)) for q in (0.25, 0.75)],
    "frac_never_reconverged_150": float((~ok).mean()),
    "frac_transitions_resolved_within_50": float(np.mean(resolved50)),
    "late_dwell_argmax_acc": float(np.mean(acc_late)),
    "approx_gap_middwell_nats": float(np.mean(gap_mid)),
    "approx_gap_posttransition_nats": float(np.mean(gap_trans)),
})
print(json.dumps(res, indent=2))
(root / "corpus_checks.json").write_text(json.dumps(res, indent=2))
