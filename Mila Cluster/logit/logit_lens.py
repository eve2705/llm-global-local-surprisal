import gc
import math
import os
import sys

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

INPUT_CSV = "data_generation_global_vs_local_context_FINAL.csv"
OUTPUT_CSV = "logit_lens.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "pythia_410m": "EleutherAI/pythia-410m",
    "pythia_1b": "EleutherAI/pythia-1b",
    "pythia_6.9b": "EleutherAI/pythia-6.9b",
    "gpt2_large": "gpt2-large",
    "olmo3_7b_base": "allenai/Olmo-3-1025-7B",
    "llama3.1_8b_base": "meta-llama/Llama-3.1-8B",
}
TEST_MODELS = {"pythia_14m": "EleutherAI/pythia-14m"}


def final_norm(model):
    for path in ["gpt_neox.final_layer_norm", "transformer.ln_f", "model.norm"]:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError("Could not find the final layer norm for this model")


@torch.no_grad()
def layer_surprisals(model, tok, norm, unembed, context, target):
    """Surprisal (bits) of `target` after `context` at every layer (0 = embeddings)."""
    full = context + " " + target
    enc = tok(full, return_offsets_mapping=True, return_tensors="pt")
    ids = enc["input_ids"].to(DEVICE)
    offsets = enc["offset_mapping"][0].tolist()
    tgt_pos = [i for i, (s, e) in enumerate(offsets) if e > len(context) and e > s]
    if not tgt_pos or tgt_pos[0] == 0:
        raise ValueError(f"Could not locate target '{target}'")

    hidden = model(ids, output_hidden_states=True).hidden_states   
    out = []
    for layer, h in enumerate(hidden):
        h = h[0, [p - 1 for p in tgt_pos]]                        
        if layer < n - 1:                                       
            h = norm(h)
        logprobs = torch.log_softmax(unembed(h).float(), dim=-1)
        total = -sum(logprobs[k, ids[0, p]].item() for k, p in enumerate(tgt_pos))
        out.append(total / math.log(2))
    return out


def main():
    test = "--test" in sys.argv
    models = TEST_MODELS if test else MODELS
    output_csv = "logit_lens_test.csv" if test else OUTPUT_CSV

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    if test:
        df = df.head(8)
    done = set(pd.read_csv(output_csv).model) if os.path.exists(output_csv) else set()

    for name, repo in models.items():
        if name in done:
            print(f"Skipping {name} (already done)", flush=True)
            continue
        print(f"\n=== {name} ({repo}) on {DEVICE} ===", flush=True)
        try:
            tok = AutoTokenizer.from_pretrained(repo)
            model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).to(DEVICE)
        except Exception as e:
            print(f"  Could not load {repo}: {e}", flush=True)
            continue
        model.eval()
        norm, unembed = final_norm(model), model.get_output_embeddings()

        rows = []
        for i, r in df.iterrows():
            context = f"{r['global_context'].strip()} {r['local_context'].strip()}"
            s = layer_surprisals(model, tok, norm, unembed, context, str(r["target_word"]).strip())
            for layer, v in enumerate(s):
                rows.append({"model": name, "Example ID": r["Example ID"], "full name": r["full name"],
                             "condition": r["condition"], "layer": layer, "n_layers": len(s) - 1,
                             "surprisal": v})
            if i % 100 == 0:
                print(f"  row {i}/{len(df)}  final layer: {s[-1]:.2f} bits", flush=True)

        out = pd.DataFrame(rows)
        out.to_csv(output_csv, mode="a", header=not os.path.exists(output_csv), index=False)
        print(f"  saved -> {output_csv}", flush=True)

        del model
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
