#!/bin/bash
#SBATCH --job-name=minicons_extra
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=minicons_extra_%j.out
#SBATCH --error=minicons_extra_%j.err

export HF_HOME=$SCRATCH/hf_cache
mkdir -p "$HF_HOME"

module load python/3.10
source ~/venvs/olmo/bin/activate

cd $SLURM_SUBMIT_DIR
nvidia-smi
python -u minicons_extra.py controls
python -u minicons_extra.py pythia
python -u minicons_extra.py olmo
