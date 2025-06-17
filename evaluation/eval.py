from evaluate import load
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score.rouge_scorer import RougeScorer
from bert_score import score as bert_score
from gritlm_eval import get_gritlm_score

import os
import json
import numpy as np
import argparse


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', type=str, default='4o-mini_3shot_sample_results.json')
    parser.add_argument('--output', type=str, default='eval_4o-mini_3shot.json')
    return parser

def calc_rouge(data):
    scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_1, rouge_l, rouge_2 = [], [], []

    for item in data:
        scores = scorer.score(item['prediction'], item['target_summary'])

        rouge_1.append(scores['rouge1'][2])
        rouge_l.append(scores['rougeL'][2])
        rouge_2.append(scores['rouge2'][2])

    results = {
        'rouge_1': float(np.mean(rouge_1) * 100),
        'rouge_l': float(np.mean(rouge_l) * 100),
        'rouge_2': float(np.mean(rouge_2) * 100)
    }

    return results

def calc_bleu(data):
    bleu_1, bleu_2, bleu_3, bleu_4 = [], [], [], []

    smoothie = SmoothingFunction().method4

    for item in data:
        target_text = [item['target_summary'].split()]
        prediction = item['prediction'].split()

        bleu_1.append(sentence_bleu(target_text, prediction, weights=(1, 0, 0, 0), smoothing_function=smoothie))
        bleu_2.append(sentence_bleu(target_text, prediction, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie))
        bleu_3.append(sentence_bleu(target_text, prediction, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie))
        bleu_4.append(sentence_bleu(target_text, prediction, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie))

    results = {
        'bleu_1': np.mean(bleu_1) * 100,
        'bleu_2': np.mean(bleu_2) * 100,
        'bleu_3': np.mean(bleu_3) * 100,
        'bleu_4': np.mean(bleu_4) * 100,
        'bleu_avg': (np.mean(bleu_1) + np.mean(bleu_2) + np.mean(bleu_3) + np.mean(bleu_4))/4  * 100
    }

    return results

def calc_bert(data):
    predictions = [item['prediction'] for item in data]
    target = [item['target_summary'] for item in data]

    P, R, F1 = bert_score(predictions, target, lang='en')
    return F1.mean().item() * 100

def calc_meteor(data):
    meteor = load('meteor')

    scores = []
    for item in data:
        score = meteor.compute(predictions=[item['prediction']], references=[item['target_summary']])
        scores.append(score['meteor'])

    results = np.mean(scores) * 100
    return {'meteor': results}

def calc_gritlm(data):
    return get_gritlm_score(data=data)


parser = init_parser()
args = parser.parse_args()

model_name = args.result_file.split('_sample')[0]


if '4o' in model_name:
    with open(f'../gpt_results/{args.result_file}', 'r') as f:
        data = json.load(f)
else:
    with open(f'../results/{args.result_file}', 'r') as f:
        data = json.load(f)

for i in range(len(data)):
    if data[i]['prediction'].startswith('Translation:'):
        data[i]['prediction'] = data[i]['prediction'].split('Translation:')[0].strip()

rouge_score = calc_rouge(data)
bleu_score = calc_bleu(data)
bert = calc_bert(data)
meteor_score = calc_meteor(data)
gritlm_score = calc_gritlm(data)

final = {
    'rouge': rouge_score,
    'BLEU': bleu_score,
    'BERTScore': bert,
    'METEOR': meteor_score,
    'GritLM': gritlm_score,
}

print(final)

os.makedirs('../evaluation_results', exist_ok=True)
with open(f'../evaluation_results/{args.output}', 'w') as f:
    json.dump(final, fp=f, indent=2)