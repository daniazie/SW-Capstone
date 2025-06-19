#!/usr/bin/bash

python gen_sft.py --model_name model_name --setting 3b --dataset sample --output model_name_3shot_sample_results.json
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python gen_refine.py --generation_model model_name --refinement_model gpt-4o-mini --case_study --iter 1 --output results.json


exit
