import sys
import pandas as pd
import numpy as np
import json
import os

from CXRMetric.run_eval import calc_metric, CompositeMetric
import argparse

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', type=str, default='4o-mini_3shot_sample_results.json')
    parser.add_argument('--output', type=str, default='eval_4o-mini_3shot.json')
    return parser


def convert_to_csv(data, model):

    gt_reports, predicted_reports = [], []

    for i, item in enumerate(data):
        try:
            reference = item['reference']
            prediction = item['generated_caption']
        except:
            reference = item['target_summary']
            try:
                prediction = item['prediction']
            except:
                prediction = item['predicted_summary']
        gt_reports.append({
            'study_id': i,
            'report': reference
        })
        predicted_reports.append({
                'study_id': i,
                'report': prediction
            })

    gt_df = pd.DataFrame(gt_reports).set_index('study_id')
    pred_df = pd.DataFrame(predicted_reports).set_index('study_id')

    
    gt_df.to_csv(f'./results_csv/{model}_gt_reports.csv')
    pred_df.to_csv(f'./results_csv/{model}_pred_reports.csv')

def get_clinical_metric(data, model):
    os.makedirs('./results_csv', exist_ok=True)
    gt_csv_path = f'./results_csv/{model}_gt_reports.csv'
    pred_csv_path = f'./results_csv/{model}_pred_reports.csv'

    # ✅ Only generate CSVs if *both* files are missing
    if not (os.path.exists(gt_csv_path) and os.path.exists(pred_csv_path)):
        convert_to_csv(data, model)

    calc_metric(
        f'./results_csv/{model}_gt_reports.csv',
        f'./results_csv/{model}_pred_reports.csv',
        f'./results_csv/{model}_output.csv',
        use_idf=False)
    
    df = pd.read_csv(f'./results_csv/{model}_output.csv')
    chexbert_score = float(df['semb_score'].mean()) * 100
    radgraph_score = float(df['radgraph_combined'].mean()) * 100 
    radcliq_v0 = float(df['RadCliQ-v0'].mean())
    radcliq_v1 = float(df['RadCliQ-v1'].mean())

    return {
        'CheXbert': chexbert_score,
        'RadGraph-F1': radgraph_score,
        'RadCliQ-v0': radcliq_v0,
        'RadCliQ-v1': radcliq_v1
    }

parser = init_parser()
args = parser.parse_args()



if '/' in args.result_file:
    with open(args.result_file, 'r') as file:
        data = json.load(file)
    model_name = 'FG' if 'first_gen' in args.result_file else 'CS'
    clinical_scores = []
    for i, item in enumerate(data):
        clinical_scores.append(get_clinical_metric([item], f'{model_name}{i}'))
else:
    model_name = args.result_file.split('_sample')[0]
    if '4o' in args.result_file or '4.1' in args.result_file:
        with open(f'../gpt_results/{args.result_file}', 'r') as f:
            data = json.load(f)
    elif 'refine' in args.result_file:
        with open(f'../refine_results/{args.result_file}', 'r') as file:
            data = json.load(file)
    elif 'cleaned' in args.result_file:
        with open(f'../results_cleaned/{args.result_file}', 'r') as file:
            data = json.load(file)
    else:
        with open(f'../results/{args.result_file}', 'r') as f:
            data = json.load(f)

        clinical_scores = get_clinical_metric(data, model_name)

os.makedirs('scores', exist_ok=True)
with open(f'scores/{args.output}', 'w') as f:
    json.dump(clinical_scores, fp=f, indent=2)