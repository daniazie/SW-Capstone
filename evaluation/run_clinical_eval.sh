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


python clinical_metrics.py --result_file ../refine_results/qwen3-sft-gpt_completeness_refine_gpt_sample1_results.json --output completeness.json
python clinical_metrics.py --result_file ../refine_results/qwen3-sft-gpt_factuality_refine_gpt_sample1_results.json --output factuality.json
python clinical_metrics.py --result_file ../refine_results/qwen3-sft-gpt_readability_refine_gpt_sample1_results.json --output readability.json
python clinical_metrics.py --result_file ../refine_results/qwen3-sft-gpt_format_refine_gpt_sample1_results.json --output format.json
python clinical_metrics.py --result_file ../refine_results/qwen3-sft-gpt_conciseness_refine_gpt_sample1_results.json --output conciseness.json
python clinical_metrics.py --result_file ../refine_results/qwen3-sft-gpt_writing-style_refine_gpt_sample1_results.json --output writing-style.json
python clinical_metrics.py --result_file ../refine_results/qwen3-sft-gpt_structure_refine_gpt_sample1_results.json --output structure.json

exit

