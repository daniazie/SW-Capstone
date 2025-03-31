#!/usr/bin/bash

#SBATCH -J biolaysumm_gpt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g4
#SBATCH -t 1-0
#SBATCH -o ./logs/slurm-%A.out

python gpt_gen.py --model="gpt-4o" --output="4o_3shot_result.json"
python gpt_gen.py --model="gpt-4o-mini" --output="4o-mini_3shot_result.json"
