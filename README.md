# llm-global-local-surprisal

Data and scripts for comparing the effect of global context (a passage) and local context (the agent of the final clause) on language model surprisal.

Each target word follows a passage that is related (GR) or unrelated (GU) to it, and a final clause whose agent is related (LR) or unrelated (LU) to it. This gives four conditions: GRLR, GRLU, GULR and GULU. The design is adapted from Hess et al. (1995).

## Files

~~~
data/
  surprisal_minicons_all.csv        surprisal in the four conditions, 29 models
  controls_minicons.csv             surprisal after the final clause alone (LR, LU)
  pythia_checkpoints_minicons.csv   Pythia 410M, 1B and 6.9B checkpoints
  olmo_checkpoints_minicons.csv     OLMo-3 7B checkpoints and post-training stages
  logit_lens.csv                    surprisal at each layer, six base models
Mila Cluster/                       Slurm jobs
  logit/                            logit lens script and job
minicons_all.py                     surprisal in the four conditions
minicons_extra.py                   final clause alone and training checkpoints
~~~

The checkpoint files contain the passages, final clauses and target words. Surprisal columns are named `surprisal_<model>` and are in bits.

## Surprisal

Surprisal is computed with [minicons](https://github.com/kanishkamisra/minicons) (Misra, 2022) and summed over the target's subword tokens.

## Models

~~~
GPT-2:    gpt2, gpt2-medium, gpt2-large
Pythia:   EleutherAI/pythia-{14m, 70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b}
Qwen2.5:  Qwen/Qwen2.5-{0.5B, 1.5B, 3B, 7B}, base and Instruct
Llama:    meta-llama/Llama-3.2-{1B, 3B}, meta-llama/Llama-3.1-8B, base and Instruct
Mistral:  mistralai/Mistral-7B-v0.3, base and Instruct
OLMo-3:   allenai/Olmo-3-1025-7B, allenai/Olmo-3-7B-Instruct (SFT, DPO, Instruct)
~~~
