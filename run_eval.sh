#!/usr/bin/bash

#SBATCH -J biolaysumm_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g4
#SBATCH -t 1-0
#SBATCH -o ./logs/slurm-%A.out
#SBATCH -e ./logs/slurm-err-%A.out

python ./evaluation/evaluation.py --result_file 4o_0shot_sample100_results.json --output 4o_0shot_gritlm.json
