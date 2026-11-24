import gc
import math
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from huggingface_hub import list_repo_refs, scan_cache_dir
from minicons import scorer

INPUT_CSV = "data_generation_global_vs_local_context_FINAL.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 16

CONTROL_MODELS = {
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
    "olmo3_7b_base": "allenai/Olmo-3-1025-7B",
    "olmo3_7b_instruct": "allenai/Olmo-3-7B-Instruct",
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
}

PYTHIA_SIZES = ["410m", "1b", "6.9b"]
PYTHIA_STEPS = [0, 1, 8, 64, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000, 143000]

OLMO_REPO = "allenai/Olmo-3-1025-7B"
OLMO_N_STAGE1 = 12
OLMO_POST = [
    ("post_1_sft", "allenai/Olmo-3-7B-Instruct-SFT", "main"),
    ("post_2_dpo", "allenai/Olmo-3-7B-Instruct-DPO", "main"),
    ("post_3_instruct", "allenai/Olmo-3-7B-Instruct", "main"),
]

def score_bits(model, prefixes, targets):
    """Surprisal in bits of each target after its prefix (minicons gives natural log)."""
    lps = []
    for i in range(0, len(prefixes), BATCH):
        lps += model.conditional_score(prefixes[i:i + BATCH], targets[i:i + BATCH],
                                       reduction=lambda x: x.sum(0).item())
    return [-lp / math.log(2) for lp in lps]


def load(repo, revision="main"):
    kwargs = {"device": DEVICE, "dtype": torch.float32}
    if revision != "main":
        kwargs["revision"] = revision
    return scorer.IncrementalLMScorer(repo, **kwargs)


def free_gpu():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def free_cache(repo, revision):
    if revision == "main":
        return
    try:
        info = scan_cache_dir()
        hashes = [rev.commit_hash for r in info.repos if r.repo_id == repo
                  for rev in r.revisions if revision in rev.refs]
        if hashes:
            strat = info.delete_revisions(*hashes)
            print(f"  freeing {strat.expected_freed_size_str} from cache", flush=True)
            strat.execute()
    except Exception as e:
        print(f"  (could not clean cache: {e})", flush=True)


def run_plan(plan, prefixes, targets, out, output_csv):
    for name, repo, rev in plan:
        col = f"surprisal_{name}"
        if col in out.columns:
            print(f"Skipping {name} (already done)", flush=True)
            continue
        print(f"\n=== {name} ({repo} @ {rev}) on {DEVICE} ===", flush=True)
        try:
            model = load(repo, rev)
        except Exception as e:
            print(f"  Could not load {repo} @ {rev}: {e}", flush=True)
            continue
        out[col] = score_bits(model, prefixes, targets)
        out.to_csv(output_csv, index=False)
        print(f"  saved -> {output_csv}", flush=True)
        del model
        free_gpu()
        free_cache(repo, rev)


def passages(df):
    prefixes = (df["global_context"].str.strip() + " " + df["local_context"].str.strip()).tolist()
    targets = df["target_word"].astype(str).str.strip().tolist()
    return prefixes, targets


def start(output_csv, base):
    return pd.read_csv(output_csv) if os.path.exists(output_csv) else base.copy()



def controls(df):
    output_csv = "controls_minicons.csv"
    rows = []
    for eid, g in df.groupby("Example ID", sort=False):
        g = g.set_index("condition")
        target = str(g.loc["GRLR", "target_word"]).strip()
        base = {"Example ID": eid, "full name": g.loc["GRLR", "full name"], "target_word": target}
        for control, cond in [("local_only_LR", "GRLR"), ("local_only_LU", "GRLU")]:
            clause = g.loc[cond, "local_context"].strip()
            rows.append({**base, "control": control, "context": clause[0].upper() + clause[1:]})
    ctrl = pd.DataFrame(rows)
    out = start(output_csv, ctrl)
    print(f"{len(out)} control rows", flush=True)
    plan = [(name, repo, "main") for name, repo in CONTROL_MODELS.items()]
    run_plan(plan, out["context"].tolist(), out["target_word"].tolist(), out, output_csv)


def pythia(df):
    output_csv = "pythia_checkpoints_minicons.csv"
    plan = [(f"pythia_{size}_step{step}", f"EleutherAI/pythia-{size}", f"step{step}")
            for size in PYTHIA_SIZES for step in PYTHIA_STEPS]
    print(f"Will score {len(plan)} checkpoints", flush=True)
    if "--list" in sys.argv:
        for p in plan:
            print(" ", *p)
        return
    prefixes, targets = passages(df)
    run_plan(plan, prefixes, targets, start(output_csv, df), output_csv)



def olmo_plan():
    refs = [b.name for b in list_repo_refs(OLMO_REPO).branches]
    stages = {1: [], 2: [], 3: []}
    for r in refs:
        m = re.fullmatch(r"stage(\d)-step(\d+)", r)
        if m and int(m.group(1)) in stages:
            stages[int(m.group(1))].append((int(m.group(2)), r))
    for s in stages:
        stages[s].sort()
    chosen = []
    s1 = stages[1]
    if s1:
        steps = np.array([st for st, _ in s1], dtype=float)
        first = steps[steps > 0][0]
        targets = np.geomspace(first, steps[-1], OLMO_N_STAGE1)
        idx = {0} | {int(np.argmin(abs(steps - t))) for t in targets}
        chosen += [s1[i] for i in sorted(idx)]
    if stages[2]:
        chosen.append(stages[2][-1])
    if stages[3]:
        chosen.append(stages[3][-1])
    plan = [(rev, OLMO_REPO, rev) for _, rev in chosen]
    plan.append(("final_base", OLMO_REPO, "main"))
    return plan + OLMO_POST


def olmo(df):
    output_csv = "olmo_checkpoints_minicons.csv"
    plan = olmo_plan()
    print(f"Will score {len(plan)} checkpoints", flush=True)
    if "--list" in sys.argv:
        for p in plan:
            print(" ", *p)
        return
    prefixes, targets = passages(df)
    run_plan(plan, prefixes, targets, start(output_csv, df), output_csv)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    jobs = {"controls": controls, "pythia": pythia, "olmo": olmo}
    if what not in jobs:
        sys.exit(f"Usage: python minicons_extra.py [{'|'.join(jobs)}]")
    jobs[what](pd.read_csv(INPUT_CSV, encoding="utf-8-sig"))
    print("\nDone.", flush=True)
