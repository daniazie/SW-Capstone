from transformers import AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from huggingface_hub import login
import torch
from tqdm import tqdm
import os
import json
import argparse


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help="Model to use: [llama|qwen]", default='llama')
    parser.add_argument('--dataset', type=str, help="sample or full dataset: [sample|full]", default='sample')
    return parser

parser = init_parser()
args = parser.parse_args()


login(os.environ['HUGGINGFACE_TOKEN'])

if 'sample' in args.dataset:
    with open('sample100.json', 'r') as f:
        dataset = json.load(f)
else:
    dataset = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track", split='validation')

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_name = args.model_name
model = AutoPeftModelForCausalLM.from_pretrained(f'./models/{model_name.capitalize()}_v1', device_map='auto', trust_remote_code=True, low_cpu_mem_usage=True, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(f'./models/{model_name.capitalize()}_v1', trust_remote_code=True)

data = []
for i, item in enumerate(tqdm(dataset, desc='Predicting...')):
    input_text = item['radiology_report']
    gold = item['layman_report']
    
    text = f"### Rewrite the following radiology report in layman's terms.\n ### Radiology Report: {input_text}\n ### Layman's Report: "
    model_inputs = tokenizer(text, return_tensors='pt')
    model_inputs = {n: m.to('cuda') for n, m in model_inputs.items()}

    output = model.generate(
        **model_inputs,
        max_new_tokens=100,
        top_k=50,
        top_p=0.9,
        temperature=0.5,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.2,
        early_stopping=True
        )
    
    document = tokenizer.batch_decode(output, skip_special_tokens=True)
    document = document[0].split("### Layman's Report: ")[1].strip()
    
    dic = {
        'radiology_report': input_text,
        'target_summary': gold,
        'predicted_summary': document
    }

    data.append(dic)
os.makedirs('./results', exist_ok=True)
with open(f'./results/{args.model_name}_{args.dataset}_results_v1.json', 'a') as f:
    json.dump(data, fp=f, indent=2)


    