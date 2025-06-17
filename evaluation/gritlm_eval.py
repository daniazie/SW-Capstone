from gritlm import GritLM
import nltk
from scipy.spatial.distance import cosine
from tqdm import tqdm
import torch
import argparse
import numpy as np
import json
from math import floor

model = GritLM("GritLM/GritLM-7B", mode='embedding', torch_dtype=torch.bfloat16)

def sentence_split(text):
    return nltk.sent_tokenize(text)

def gritlm_instruction(instruction):
    
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"    

def get_embeddings(prediction, target_summary, model):
    instruction = "Determine if two sentences express the same idea."
    
    query = prediction
    document = target_summary

    d_rep = model.encode(document, instruction=gritlm_instruction(""))
    q_rep = model.encode(query, instruction=gritlm_instruction(instruction))
    
    return d_rep, q_rep

    



"""with open(f'./gpt_results/{args.input}', 'r') as f:
    data = json.load(f)"""

def calc_sim_score(data):
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()



    similarities = []
    sims = []
    for i, item in enumerate(tqdm(data, desc="Evaluating")):
        prediction = item['prediction']
        target_summary = item['target_summary']

        prediction = sentence_split(prediction)
        target_summary = sentence_split(target_summary)

        similar_count = 0
        d_rep, q_rep = get_embeddings(prediction=prediction, target_summary=target_summary, model=model)

        max_sims = []
        pred = []
        target = []
        for d in range(len(d_rep)):
            max_sim = 0
            for q in range(len(q_rep)):
                sim = 1 - cosine(q_rep[q], d_rep[d])
                if sim > max_sim:
                    max_sim = sim
            sims.append({
                'max_sim': max_sim,
                'pred': prediction[q],
                'target': target_summary[d]
            })
            max_sims.append(float(max_sim))
        similarities.append(np.mean(max_sims))

    with open('sims.json', 'w') as file:
        json.dump(sims, fp=file, indent=2)

    torch.cuda.empty_cache()
    sim_score = float(np.mean(similarities))
    return sim_score

def calc_prop_score(data, sim_score):

    theta = sim_score
    proportion = []
    for i, item in enumerate(tqdm(data, desc="Evaluating")):
        prediction = item['prediction']
        target_summary = item['target_summary']

        prediction = sentence_split(prediction)
        target_summary = sentence_split(target_summary)

        similar_count = 0
        d_rep, q_rep = get_embeddings(prediction=prediction, target_summary=target_summary, model=model)

        for d in range(len(d_rep)):
            for q in range(len(q_rep)):
                sim = 1 - cosine(q_rep[q], d_rep[d])
                if sim > theta:
                    similar_count += 1
        proportion.append(similar_count/len(prediction))
    prop_score = float(np.mean(proportion))
    return prop_score

def get_gritlm_score(data):
    sim_score = calc_sim_score(data)
    prop_score = calc_prop_score(data, sim_score)

    gritlm_score = {
        'similarity_score': sim_score * 100,
        'proportion_score': prop_score * 100
    }

    return gritlm_score
"""
if "mini" in args.output:
    model_name = '4o-mini'
else:
    model_name = '4o'

with open(f"./{args.output}", "w") as f:
    json.dump({'avg_sim_score': float(np.mean(similarities) * 100), 'proportion_score': np.mean(proportion) * 100}, indent=2, fp=f)"""



