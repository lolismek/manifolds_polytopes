"""Full dose-response for the stage-3b audition latents.

Same instrument as the stage-2.5 screen (clamp at layer 12 over 256 natural
prefixes, final-position KL(steered || clean) and entropy), but over the full
strength range used anywhere in the funnel: {1, 2, 3, 5, 8, 12} x mean firing
activation. Gives measured per-token evidence numbers where earlier statements
were extrapolations (1x) or unmeasured (8x, 12x).

Output: results/stage2_5/dose_response.json

Usage (on tigerfish):
  python stage2_5_dose_response.py --device cuda:0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from stage2_5_causal import MODEL, SAE_RELEASE, SAE_ID, LAYER, get_prefixes

MULTS = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
N_PREFIX = 256
PREFIX_LEN = 64
BATCH = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)

    root = Path(__file__).parent.parent / "results"
    latents = [r["latent"] for r in json.loads(
        (root / "stage3" / "per_latent_3b.json").read_text())]
    mean_act = np.load(root / "stage1" / "latent_stats.npz")["mean_act"]

    from sae_lens import SAE
    sae = SAE.from_pretrained(SAE_RELEASE, SAE_ID, device=str(dev))[0].to(torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev)
    model.eval()

    steer = {"j": None, "target": None}

    def hook(_, __, output):
        if steer["j"] is None:
            return
        is_tuple = isinstance(output, tuple)
        x = output[0] if is_tuple else output
        xf = x[:, 1:, :].float()
        a = sae.encode(xf.reshape(-1, xf.shape[-1]))[:, steer["j"]]
        a = a.reshape(xf.shape[0], xf.shape[1])
        delta = (steer["target"] - a).unsqueeze(-1) * sae.W_dec[steer["j"]]
        x = torch.cat([x[:, :1, :], (xf + delta).to(x.dtype)], dim=1)
        return (x,) + output[1:] if is_tuple else x

    model.model.layers[LAYER].register_forward_hook(hook)
    prefixes = get_prefixes(tokenizer, N_PREFIX, PREFIX_LEN)

    def final_logprobs():
        outs = []
        with torch.no_grad():
            for i in range(0, N_PREFIX, BATCH):
                ids = prefixes[i : i + BATCH].to(dev)
                logits = model(ids).logits[:, -1, :].float()
                outs.append(torch.log_softmax(logits, dim=-1))
        return torch.cat(outs)

    logp_clean = final_logprobs()

    rows = []
    for j in latents:
        row = {"latent": int(j)}
        steer["j"] = int(j)
        for mult in MULTS:
            steer["target"] = float(mult * mean_act[j])
            logp_s = final_logprobs()
            p_s = logp_s.exp()
            row[f"kl_{mult:g}x"] = round(float((p_s * (logp_s - logp_clean)).sum(-1).mean()), 5)
        steer["j"] = None
        rows.append(row)
        print(row, flush=True)

    (root / "stage2_5" / "dose_response.json").write_text(json.dumps(rows, indent=2))
    arr = {m: np.array([r[f"kl_{m:g}x"] for r in rows]) for m in MULTS}
    print("\nmedian KL per strength over the 20 audition latents:")
    for m in MULTS:
        print(f"  {m:g}x: median {np.median(arr[m]):.4f}, "
              f"q25 {np.quantile(arr[m], .25):.4f}, q75 {np.quantile(arr[m], .75):.4f}")


if __name__ == "__main__":
    main()
