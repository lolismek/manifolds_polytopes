"""Stage 2 of the latent casting funnel: co-activation check on natural text.

For the stage-2.5 causal pool, run natural text through the teacher and count,
for every pair of pool latents, how often they fire on the same token. Compare
against the independence prediction: lift = P(i and j) / (P(i) P(j)).

Latents that systematically co-fire are behavioral aliases of shared structure;
their states would not be K distinct hash directions. We flag pairs, we do not
pick favorites — casting later just avoids flagged pairs (usability, not
semantics).

Outputs (results/stage2/):
  coactivation.npz  pool ids, joint firing counts, marginals, n_pos
  coact_report.json flagged pairs (lift and Jaccard above thresholds) + summary

Usage (on tigerfish):
  python stage2_coactivation.py --n_tokens 2_000_000 --device cuda:0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "google/gemma-2-2b"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_12/width_16k/canonical"
LAYER = 12
SEQ_LEN = 1024
BATCH = 16

LIFT_FLAG = 5.0      # joint firing > 5x independence prediction
JACCARD_FLAG = 0.1   # or > 10% overlap of firing sets

def token_stream(tokenizer, n_tokens: int):
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    bos = tokenizer.bos_token_id
    buf, yielded, batch = [], 0, []
    for ex in ds:
        buf.extend(tokenizer(ex["text"], add_special_tokens=False)["input_ids"])
        while len(buf) >= SEQ_LEN - 1:
            batch.append([bos] + buf[: SEQ_LEN - 1])
            buf = buf[SEQ_LEN - 1 :]
            if len(batch) == BATCH:
                yield torch.tensor(batch)
                yielded += BATCH * SEQ_LEN
                batch = []
                if yielded >= n_tokens:
                    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_tokens", type=int, default=2_000_000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out or Path(__file__).parent.parent / "results" / "stage2")
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    res25 = Path(__file__).parent.parent / "results" / "stage2_5"

    from sae_lens import SAE

    sae = SAE.from_pretrained(SAE_RELEASE, SAE_ID, device=str(dev))[0]
    sae = sae.to(torch.float32)

    pool = json.loads((res25 / "causal_pool.json").read_text())["pool"]
    ids = torch.tensor([p["latent"] for p in pool], device=dev)
    n_pool = len(ids)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev)
    model.eval()

    resid = {}
    def hook(_, __, output):
        resid["x"] = output[0] if isinstance(output, tuple) else output
    model.model.layers[LAYER].register_forward_hook(hook)

    joint = torch.zeros(n_pool, n_pool, dtype=torch.float64, device=dev)
    marg = torch.zeros(n_pool, dtype=torch.float64, device=dev)
    n_pos = 0

    with torch.no_grad():
        for bi, tok_ids in enumerate(token_stream(tokenizer, args.n_tokens)):
            tok_ids = tok_ids.to(dev)
            model(tok_ids)
            x = resid["x"][:, 1:, :].float()
            acts = sae.encode(x.reshape(-1, x.shape[-1]))[:, ids]   # (B*T, n_pool)
            f = (acts > 0).double()
            joint += f.T @ f
            marg += f.sum(0)
            n_pos += f.shape[0]
            if bi % 25 == 0:
                print(f"batch {bi}, {n_pos:,} positions", flush=True)

    joint_np = joint.cpu().numpy()
    marg_np = marg.cpu().numpy()
    p = marg_np / n_pos
    pj = joint_np / n_pos
    lift = pj / np.outer(p, p).clip(min=1e-12)
    union = marg_np[:, None] + marg_np[None, :] - joint_np
    jaccard = joint_np / union.clip(min=1)
    np.fill_diagonal(lift, 0)
    np.fill_diagonal(jaccard, 0)

    np.savez(out / "coactivation.npz", pool_ids=ids.cpu().numpy(),
             joint=joint_np, marginal=marg_np, n_pos=n_pos)

    iu = np.triu_indices(n_pool, k=1)
    flagged = [
        {"i": int(ids[a]), "j": int(ids[b]),
         "lift": round(float(lift[a, b]), 2), "jaccard": round(float(jaccard[a, b]), 4)}
        for a, b in zip(*iu)
        if lift[a, b] > LIFT_FLAG or jaccard[a, b] > JACCARD_FLAG
    ]
    flagged.sort(key=lambda r: -r["jaccard"])

    report = {
        "n_pool": n_pool,
        "n_pos": n_pos,
        "thresholds": {"LIFT_FLAG": LIFT_FLAG, "JACCARD_FLAG": JACCARD_FLAG},
        "n_pairs": int(len(iu[0])),
        "n_flagged": len(flagged),
        "lift_quantiles": {q: float(np.quantile(lift[iu], float(q)))
                           for q in ["0.5", "0.9", "0.99"]},
        "jaccard_quantiles": {q: float(np.quantile(jaccard[iu], float(q)))
                              for q in ["0.5", "0.9", "0.99"]},
        "flagged_pairs": flagged,
    }
    (out / "coact_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "flagged_pairs"}, indent=2))
    print(f"top flagged: {flagged[:5]}")


if __name__ == "__main__":
    main()
