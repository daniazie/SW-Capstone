from openai import OpenAI
from random import sample
from datasets import load_dataset
import os
import json
import argparse
from tqdm import tqdm
import time

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="GPT model to be used: [gpt-4o|gpt-4o-mini|gpt-4.1]", default="gpt-4o-mini")
 #   parser.add_argument('--few_shot', type=bool, help="Whether to use few shot prompting or not", default="True")
    parser.add_argument('--data', type=str, default='sample')
    parser.add_argument("--output", type=str, help="Output file", default="sample100_result.json")
    return parser

parser = init_parser()
args = parser.parse_args()

if 'sample' in args.data:
    with open('sample100.json', 'r') as f:
        dataset = json.load(fp=f)
    with open('examples_3shot.json', 'r') as f:
        examples = json.load(fp=f)
elif 'test' in args.data:
    dataset = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track", split='test')
    example_data = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track", split='validation')
    examples = []

    for example in example_data:
        examples.append({
            'radiology_report': example['radiology_report'],
            'layman_report': example['layman_report']
        })


client = OpenAI(api_key=os.environ['OPENAI_KEY'])

if 'gpt' not in args.model:
    model = f'gpt-{args.model}'
else:
    model = args.model

def my_3shot_prompt(item, example):
    return [{"role": "system", "content": "You are translating professional radiology reports into layman's terms. Do not include any medical jargon. Write concisely. When rewriting the radiology reports, follow these examples:\n"
                                            f"Radiology report: {example[0]['radiology_report']}\n Layman's report: {example[0]['layman_report']}\n"
                                            f"Radiology report: {example[1]['radiology_report']}\n Layman's report: {example[1]['layman_report']}\n"
                                            f"Radiology report: {example[2]['radiology_report']}\n Layman's report: {example[2]['layman_report']}\n"},
            {"role": "user", "content": f"Radiology report: {item['radiology_report']}\n Layman's report: "}
            ]


def XMS_prompt(item, example):
    instruction = """You are a writer of science journalism
    
    Given a radiology report, please finish the following tasks.
    
    Tasks: 1. Translation: Please translate the following report into plain language that is easy to understand (layman's terms). The layman-translated report requires writing factual descriptions, while also paraphrasing complex scientific concepts using a language that is accessible to the general public. Meanwhile, it preserves the details as much as possible. Each translated sentence must correspond to the original sentence. For example, a 4-sentence report should be translated into a 4-sentence layman's termed report. You must translate all the reports.
    
    Here are some examples of layman-version reports:
    """

    query = f"""Report to be translated:\n {item['radiology_report']}"""

    for ex in example:
        instruction += f"{ex['layman_report']}\n\n"

    return [{"role": "system", "content": instruction},
            {"role": "user", "content": query}]

def XMS_0shot_prompt(item):
    instruction = """You are a writer of science journalism
    
    Given a radiology report, please finish the following tasks.
    
    Tasks: 1. Translation: Please translate the following report into plain language that is easy to understand (layman's terms). The layman-translated report requires writing factual descriptions, while also paraphrasing complex scientific concepts using a language that is accessible to the general public. Meanwhile, it preserves the details as much as possible. Each translated sentence must correspond to the original sentence. For example, a 4-sentence report should be translated into a 4-sentence layman's termed report. You must translate all the reports.
    
    """

    query = f"""Report to be translated:\n {item['radiology_report']}"""

    return [{"role": "system", "content": instruction},
            {"role": "user", "content": query}]

data = []
idx = 1
for i, item in enumerate(tqdm(dataset, desc='Generating')):
    ex_3shot = sample(examples, 3)
    if 'XMS' in args.output:
        if '0shot' in args.output:
            completion = client.chat.completions.create(
                model=model,
                messages= XMS_0shot_prompt(item),
                temperature=0.3,
                max_tokens=2048,
            )
        else:
            completion = client.chat.completions.create(
                model=model,
                messages= XMS_prompt(item, ex_3shot),
                temperature=0.3,
                max_tokens=2048
            )

    else:
        completion = client.chat.completions.create(
            model=model,
            messages= my_3shot_prompt(item, ex_3shot),
            temperature=0.3,
            max_tokens=2048
        )

    prediction = completion.choices[0].message.content
    pred = prediction.replace("\n", " ")
    pred = prediction.replace("  ", " ")

    if 'test' in args.data:
        os.makedirs('../results_test', exist_ok=True)
        with open('../results_test/250516_open_rrg.txt', 'a') as file:
            file.write(pred+'\n')
            
    data.append({
        'document': item['radiology_report'],
        'generated_caption': prediction,
        'reference': item['layman_report'],
    })


if 'test' in args.data:
    os.makedirs('../results_test', exist_ok=True)
    with open(f'../results_test/{args.output}', 'w') as file:
        json.dump(data, fp=file, indent=2)
    
else:
    os.makedirs('../gpt_results', exist_ok=True)
    with open(f"../gpt_results/{args.output}", 'w') as f:
        json.dump(data, fp=f, indent=2)