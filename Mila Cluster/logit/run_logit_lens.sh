#!/bin/bash
#SBATCH --job-name=logit_lens
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=logit_lens_%j.out
#SBATCH --error=logit_lens_%j.err

export HF_HOME=$SCRATCH/hf_cache
mkdir -p "$HF_HOME"

module load python/3.10
source ~/venvs/olmo/bin/activate

cd $SLURM_SUBMIT_DIR
python -u logit_lens.py
