#!/usr/bin/bash

#SBATCH -J biolaysumm_llama
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g4
#SBATCH -t 1-0
#SBATCH -o ./logs/slurm-%A.out

huggingface-cli login --token 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
python3 sft_llama.py

exit