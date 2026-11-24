import gc
import math
import os

import pandas as pd
import torch
from minicons import scorer

INPUT_CSV = "data_generation_global_vs_local_context_FINAL.csv"
OUTPUT_CSV = "surprisal_minicons_all.csv"
BATCH = 16

MODELS = {
    "gpt2_small": "gpt2",
    "gpt2_medium": "gpt2-medium",
    "gpt2_large": "gpt2-large",
    "pythia_14m": "EleutherAI/pythia-14m",
    "pythia_70m": "EleutherAI/pythia-70m",
    "pythia_160m": "EleutherAI/pythia-160m",
    "pythia_410m": "EleutherAI/pythia-410m",
    "pythia_1b": "EleutherAI/pythia-1b",
    "pythia_1.4b": "EleutherAI/pythia-1.4b",
    "pythia_2.8b": "EleutherAI/pythia-2.8b",
    "pythia_6.9b": "EleutherAI/pythia-6.9b",
    "qwen2.5_0.5b_base": "Qwen/Qwen2.5-0.5B",
    "qwen2.5_0.5b_instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5_1.5b_base": "Qwen/Qwen2.5-1.5B",
    "qwen2.5_1.5b_instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5_3b_base": "Qwen/Qwen2.5-3B",
    "qwen2.5_3b_instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5_7b_base": "Qwen/Qwen2.5-7B",
    "qwen2.5_7b_instruct": "Qwen/Qwen2.5-7B-Instruct",
    "llama3.2_1b_base": "meta-llama/Llama-3.2-1B",
    "llama3.2_1b_instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2_3b_base": "meta-llama/Llama-3.2-3B",
    "llama3.2_3b_instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.1_8b_base": "meta-llama/Llama-3.1-8B",
    "llama3.1_8b_instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral_7b_base": "mistralai/Mistral-7B-v0.3",
    "mistral_7b_instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "olmo3_7b_base": "allenai/Olmo-3-1025-7B",
    "olmo3_7b_instruct": "allenai/Olmo-3-7B-Instruct",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    prefixes = (df["global_context"].str.strip() + " " + df["local_context"].str.strip()).tolist()
    targets = df["target_word"].astype(str).str.strip().tolist()

    if os.path.exists(OUTPUT_CSV):
        out = pd.read_csv(OUTPUT_CSV)
    else:
        out = df[["Example ID", "condition", "full name", "target_word"]].copy()

    for name, repo in MODELS.items():
        col = f"surprisal_{name}"
        if col in out.columns:
            print(f"Skipping {name} (already done)", flush=True)
            continue
        print(f"\n=== {name} ({repo}) on {DEVICE} ===", flush=True)
        try:
            model = scorer.IncrementalLMScorer(repo, device=DEVICE, dtype=torch.float32)
        except Exception as e:
            print(f"  Could not load {repo}: {e}", flush=True)
            continue

        logprobs = []
        for i in range(0, len(prefixes), BATCH):
            logprobs += model.conditional_score(
                prefixes[i:i + BATCH],
                targets[i:i + BATCH],
                reduction=lambda x: x.sum(0).item(),
            )
            if i % 160 == 0:
                print(f"  row {i}/{len(prefixes)}", flush=True)

        out[col] = [-lp / math.log(2) for lp in logprobs]  
        out.to_csv(OUTPUT_CSV, index=False)
        m = out.groupby("condition")[col].mean().round(2).to_dict()
        print(f"  saved. mean bits by condition: {m}", flush=True)

        del model
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()