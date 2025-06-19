#!/usr/bin/bash

#SBATCH -J biolaysumm_gen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g4
#SBATCH -t 1-0
#SBATCH -o ./logs/slurm-%A.out
#SBATCH -e ./logs/slurm-err-%A.out

python feedback_gen.py --input_file results_cleaned_500/llama-instruct_3b_3shot_results_cleaned.json --output feedbacks.json
python feedback_gen.py --input_file results_cleaned/qwen-instruct_SFT_7b_3shot_results.json --output feedbacks.json
python feedback_gen.py --input_file results_cleaned/qwen-instruct_3b_3shot_results.json --output feedback.json

exit