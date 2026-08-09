"""Per-latent recall at each stage-3b strength (300-token budget), for casting."""

import json
from pathlib import Path

import numpy as np
import torch

from stage3_analyze import bow, fit_logistic

out = Path(__file__).parent.parent / "results" / "stage3"
shards = sorted(p for p in out.glob("genb_shard*.npz") if "_raw" not in p.name)
ids = np.concatenate([np.load(s)["ids"] for s in shards])
lat = np.concatenate([np.load(s)["latent"] for s in shards])
mult = np.concatenate([np.load(s)["mult"] for s in shards])
nll = np.concatenate([np.load(s)["nll"] for s in shards])

clean_nll = float(np.median(nll[lat == -1]))
pool = sorted(set(lat[lat >= 0].tolist()))
cls = {j: c for c, j in enumerate(pool)}
uniq = np.unique(ids)
vi = {int(t): k for k, t in enumerate(uniq)}
dev = torch.device("cuda:0")

rng = np.random.default_rng(0)
res = {}
for m in [5.0, 8.0, 12.0]:
    sel = np.where((lat >= 0) & (mult == m))[0]
    y = np.array([cls[int(j)] for j in lat[sel]])
    tr, te = [], []
    for c in range(len(pool)):
        rows = sel[y == c]
        perm = rng.permutation(len(rows))
        tr += rows[perm[:30]].tolist()
        te += rows[perm[30:]].tolist()
    tr, te = np.array(tr), np.array(te)
    ytr = np.array([cls[int(j)] for j in lat[tr]])
    yte = np.array([cls[int(j)] for j in lat[te]])
    pred = fit_logistic(bow(ids[tr], vi, len(uniq), 300), ytr,
                        bow(ids[te], vi, len(uniq), 300), dev)
    for c, j in enumerate(pool):
        res.setdefault(int(j), {})[str(m)] = float((pred[yte == c] == c).mean())

rows = []
for j in pool:
    r = res[int(j)]
    r["nll8_ratio"] = float(np.median(nll[(lat == j) & (mult == 8.0)])) / clean_nll
    r["nll12_ratio"] = float(np.median(nll[(lat == j) & (mult == 12.0)])) / clean_nll
    rows.append({"latent": int(j), **{k: round(v, 3) for k, v in r.items()}})
    print(rows[-1])

(out / "per_latent_3b.json").write_text(json.dumps(rows, indent=2))
