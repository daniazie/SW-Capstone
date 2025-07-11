from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from huggingface_hub import login
import torch
from random import sample
from tqdm import tqdm
from random import sample
import os
import gc
import json
import argparse
import re
import time


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--first_gen_file', type=str, default='first_gen.json')
    parser.add_argument('--refinement_model', type=str, help="Model to use: [llama-sft|llama-instruct-sft|llama-instruct|qwen-sft|qwen-instruct-sft|qwen-instruct]", default='gpt-4o-mini')
    parser.add_argument('--range1', type=int, default=0)
    parser.add_argument('--range2', type=int)
    parser.add_argument('--output', type=str)
    parser.add_argument('--iter', type=int, default=1)

 
    return parser

def extract_total_score(feedback: str) -> float:
    # Match numbers followed by optional spaces and then '/10'
    fb = feedback.split('Total Score')[0].strip()
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*/\s*10', fb)

    if len(matches) == 0:
        return 'Error'

    # Sum the first 4 scores only
    
    scores = [float(score) for score in matches[:3]]
    
    return sum(scores)



def feedback_prompt(item, lay_report, examples, refined_prompt=""):
    prompt = f"""### You are an expert medical language reviewer. You are given a radiology report and the full output generated by a language model in response to it. Evaluate the quality of the **entire model output** (not just the lay report section) based on the following 3 criteria.

    For each, provide a **concise explanation (1-2 sentences max)** and a **score in the format x/10**. At the end, provide the total score as the **sum of all three criteria**, formatted as **n/30**.
    1. **Factuality (x/10)**: How factually consistent is the output with the original radiology report? Highlight factually incorrect or inconsistent phrases and penalize accordingly.
    2. **Completeness (x/10)**: Does the output include all important information from the radiology report? Penalize omissions.
    3. **Format (x/10)**: Penalize any commentary or non-report language, such as “Here is your revised report,” “Translation:”, or any explanation of changes. Full marks only if the output **only** contains the lay summary, without extra headers or commentary.
    4. **Total Score (n/30)**: Sum of the seven individual scores.
    """

    if not refined_prompt:
        prompt += f"""Here are some examples of evaluations.
        Original Radiology Report:
        {examples[0]['radiology_report']}

        Lay Report:
        {examples[0]['lay_report']}

        Feedback:
        {examples[0]['feedback']}

        
        Original Radiology Report:
        {examples[1]['radiology_report']}

        Lay Report:
        {examples[1]['lay_report']}

        Feedback:
        {examples[1]['feedback']}


        Original Radiology Report:
        {examples[2]['radiology_report']}

        Lay Report:
        {examples[2]['lay_report']}

        Feedback:
        {examples[2]['feedback']}\n"""

    elif refined_prompt and isinstance(refined_prompt, str):
        if 'past edits' not in prompt:
            prompt += "Here are past edits for your reference:\n"
        prompt += f"{refined}\n\n"

    query = f"""## Original Radiology Report:
    {item}

    ## Lay Report:
    {lay_report}

    ## Feedback:"""

    return [{"role": "system", "content": prompt},
            {"role": "user", "content": query}]

def refinement(feedback, item, lay_report, feedback_query=""):
    prompt = """### You are translating radiology reports into layman's terms. You are given feedback for a lay report. Use the given feedback to improve and rewrite the lay report.
    Do not include any commentary, section titles, or explanation of any changes made. The output should contain only the lay report, written clearly."""

    if feedback_query:
        if 'past feedbacks' not in prompt:
            prompt += "Here are past feedbacks for your reference:\n"
        prompt += f"{feedback_query}\n\n"

    query = f"""### Original Radiology Report:
    {item}
    
    ### Model Output:
    {lay_report}

    ### Feedback:
    {feedback}
    
    ### Use the feedback to improve the lay report. 
    ### Revised Lay Report:"""

    return [{"role": "system", "content": prompt},
            {"role": "user", "content": query}]

gc.collect()
torch.cuda.empty_cache()

parser = init_parser()
args = parser.parse_args()


client = OpenAI(api_key=os.environ['OPENAI_KEY'])

with open(f'../results_test/{args.first_gen_file}', 'r') as f:
    dataset = json.load(f)


if 'gpt' not in args.refinement_model:
    refine_model = f'gpt-{args.refinement_model}'
else:
    refine_model = args.refinement_model

    
first_gen = []
feedbacks = []
refined = []

with open('feedbacks.json', 'r') as file:
    feedback_examples = json.load(file)

for i in tqdm(range(args.range1, args.range2), desc="Generating"):
    
    input_text = dataset[i]['document']
    document = dataset[i]['generated_caption']

    iter_count = 0    
    feedbacks = ""
    refine = ""
    old_feedback = ""
    old_refine = ""
    old_iter = ""
    old_score = 0


    while iter_count < args.iter:
        example = sample(feedback_examples, 3)

        if not feedbacks:
            feedbacks = feedback_prompt(input_text, document, example)

        completion = client.chat.completions.create(
            model=refine_model,
            messages=feedbacks,
            temperature=0.3
        )

        feedback = completion.choices[0].message.content
        del completion
    
        score = extract_total_score(feedback)
        if isinstance(score, str):
            iter_count -= 1
            score = 0

        if score >= 27:
             break
        

        if not refine:
            refine = refinement(feedback, input_text, document)

        completion = client.chat.completions.create(
            model=refine_model,
            messages=refine,
            temperature=0.3,
        )

        document = completion.choices[0].message.content
        del completion

        
        iter_count += 1

        old_refine += refine[1]['content'] + document
        old_feedback += feedbacks[1]['content'] + feedback


        
        feedbacks = feedback_prompt(input_text, document, example, old_refine)
        refine = refinement(feedback, input_text, document, old_feedback)

        iter_count += 1
    
    refined.append({
        'document': input_text,
        'generated_caption': document
    })

    del input_text
    del document
    
os.makedirs('../results_test', exist_ok=True)
dir = os.listdir('../results_test')
if args.output not in dir:
    with open(f'../results_test/{args.output}', 'w') as file:
        json.dump(refined, fp=file, indent=2)
else:
    with open(f'../results_test/{args.output}', 'r') as file:
        refined_results = json.load(file)
    refined_results.extend(refined)

    with open(f'../results_test/{args.output}', 'w') as file:
        json.dump(refined_results, fp=file, indent=2)

