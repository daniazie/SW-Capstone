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


python evaluation_final.py --prediction_file ../refine_results/results.json --groundtruth_file ../refine_results/results.json --task_name rrg
exit