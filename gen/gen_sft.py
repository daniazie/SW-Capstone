from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from huggingface_hub import login
import torch
from tqdm import tqdm
from random import sample
import os
import json
import argparse

from torch.amp import autocast


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help="Model to use: [llama-sft|llama-instruct-sft|llama-instruct|qwen-sft|qwen-instruct-sft|qwen-instruct]", default='llama')
    parser.add_argument('--setting', type=str, default='3b')
    parser.add_argument('--enable_thinking', action='store_true')
    parser.add_argument('--no-enable_thinking', dest='enable_thinking', action='store_false')
    parser.add_argument('--dataset', type=str, help="sample or full dataset: [sample|validation|test]", default='sample')
    parser.add_argument('--output', type=str)

    parser.set_defaults(enable_thinking=False)
    return parser


#def XMS_prompt_llama(item, example):
    instruction = """### You are a writer of science journalism
    
    Given a radiology report, please finish the following tasks.
    
    Tasks: 1. Translation: Please translate the following report into plain language that is easy to understand (layman's terms). The layman-translated report requires writing factual descriptions, while also paraphrasing complex scientific concepts using a language that is accessible to the general public. Meanwhile, it preserves the details as much as possible. Each translated sentence must correspond to the original sentence. For example, a 4-sentence report should be translated into a 4-sentence layman's termed report. You must translate all the reports.
    
    Here are some examples of layman-version reports:
    """

    query = f"""### Report to be translated:\n {item['radiology_report']}\n\n ### Layman's report: """

    for ex in example:
        instruction += f"{ex['layman_report']}\n\n"

    prompt = instruction + query
    return prompt

def XMS_prompt(item, example):
    instruction = """You are a writer of science journalism.
    
    Given a radiology report, please finish the following tasks.
    
    Tasks: 1. Translation: Please translate the following report into plain language that is easy to understand (layman's terms). The layman-translated report requires writing factual descriptions, while also paraphrasing complex scientific concepts using a language that is accessible to the general public. Meanwhile, it preserves the details as much as possible. Each translated sentence must correspond to the original sentence. For example, a 4-sentence report should be translated into a 4-sentence layman's termed report. You must translate all the reports.
    
    Here are some examples of layman-version reports:
    """

    query = f"""Report to be translated:\n{item['radiology_report']}\n\n"""

    for ex in example:
        instruction += f"{ex['layman_report']}\n\n"

    return [{"role": "system", "content": instruction},
            {"role": "user", "content": query}]

def few_shot(item, example):
    prompt = f"""### You are translating professional radiology reports into layman's terms. Do not include any medical jargon. Write concisely. When rewriting the radiology reports, follow these examples:
    Radiology Report:\n{example[0]['radiology_report']}\n\nLayman's Report: {example[0]['layman_report']}\n
    Radiology Report:\n{example[1]['radiology_report']}\n\nLayman's Report: {example[1]['layman_report']}\n
    Radiology Report:\n{example[2]['radiology_report']}\n\nLayman's Report: {example[2]['layman_report']}"""

    query = f"### Radiology Report:\n{item['radiology_report']} ### Layman's Report:"

    
    return [{"role": "system", "content": prompt},
            {"role": "user", "content": query}]
    
def zero_shot(item):
    prompt = f"### You are translating professional radiology reports into layman's terms. Do not include any medical jargon. Write concisely."


    query = f"Radiology Report:\n{item['radiology_report']}"

    return [{"role": "system", "content": prompt},
            {"role": "user", "content": query}]
        
        

parser = init_parser()
args = parser.parse_args()


login(os.environ['HUGGINGFACE_TOKEN'])

if 'sample' in args.dataset:
    with open('sample.json', 'r') as f:
        dataset = json.load(f)
    
else:
    dataset = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track", split=args.dataset)
    

with open('examples_3shot.json', 'r') as f:
    examples = json.load(fp=f)


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_name = args.model_name
setting = args.setting
if 'SFT' in model_name:
    if '3b' or '4b' in setting:
        model = AutoModelForCausalLM.from_pretrained(f'../models/{model_name}/merged_final', device_map='auto', trust_remote_code=True, low_cpu_mem_usage=True, local_files_only=True, attn_implementation='flash_attention_2')
        tokenizer = AutoTokenizer.from_pretrained(f'../models/{model_name}/merged_final', trust_remote_code=True)
    elif '7b' in setting or '8b' in setting:
        model = AutoModelForCausalLM.from_pretrained(f'../models_7B/{model_name}/merged_final', device_map='auto', trust_remote_code=True, low_cpu_mem_usage=True, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(f'../models_7B/{model_name}/merged_final', trust_remote_code=True)
else:
    if 'qwen' in model_name:
        if '3b' in setting:
            model = AutoModelForCausalLM.from_pretrained(
                'Qwen/Qwen2.5-3B-Instruct',
                quantization_config=quantization_config,
                device_map='auto',
                attn_implementation="flash_attention_2"
            )
            tokenizer = AutoTokenizer.from_pretrained(f'Qwen/Qwen2.5-3B-Instruct', device_map='auto')
        elif '7b' in setting:
            model = AutoModelForCausalLM.from_pretrained(
                'Qwen/Qwen2.5-7B-Instruct',
                quantization_config=quantization_config,
                device_map='auto',
                attn_implementation="flash_attention_2"
            )
            tokenizer = AutoTokenizer.from_pretrained(f'Qwen/Qwen2.5-7B-Instruct', device_map='auto')
    elif 'llama' in model_name:
        if '3b' in setting:
            model = AutoModelForCausalLM.from_pretrained(
                'meta-llama/Llama-3.2-3B-Instruct',
                quantization_config=quantization_config,
                device_map='auto',
                attn_implementation="flash_attention_2"
            )
            tokenizer = AutoTokenizer.from_pretrained(f'meta-llama/Llama-3.2-3B-Instruct', device_map='auto')
        elif '8b' in setting:
            model = AutoModelForCausalLM.from_pretrained(
                'meta-llama/Llama-3.1-8B-Instruct',
                quantization_config=quantization_config,
                device_map='auto',
                attn_implementation="flash_attention_2"
            )
            tokenizer = AutoTokenizer.from_pretrained(f'meta-llama/Llama-3.1-8B-Instruct', device_map='auto')

    


data = []
for i, item in enumerate(tqdm(dataset, desc='Predicting...')):
    ex_3shot = sample(examples, 3)

    input_text = item['radiology_report']
    gold = item['layman_report']

    if 'XMS' in args.output:
        text = tokenizer.apply_chat_template(
            XMS_prompt(item, ex_3shot),
            tokenize=False,
            add_generation_prompt=True
            )
        model_inputs = tokenizer([text], return_tensors='pt').to('cuda')
        #model_inputs = {n: m.to('cuda') for n, m in model_inputs.items()}
        model_inputs = {
                k: v.to('cuda') if v.dtype in [torch.long, torch.int] else v.to(dtype=torch.float16, device='cuda')
                for k, v in model_inputs.items()
            }

        
    else:
        if '0shot' in args.output:
            text = tokenizer.apply_chat_template(
                zero_shot(item),
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors='pt')
            model_inputs = {n: m.to('cuda') for n, m in model_inputs.items()}
        elif '3shot' in args.output:
            if 'qwen3' in args.model_name:
                text = tokenizer.apply_chat_template(
                    few_shot(item, ex_3shot),
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=args.enable_thinking
                )
            else:
                text = tokenizer.apply_chat_template(
                    few_shot(item, ex_3shot),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            model_inputs = tokenizer([text], return_tensors='pt').to('cuda')
            model_inputs = {
                k: v.to('cuda') if v.dtype in [torch.long, torch.int] else v.to(dtype=torch.float16, device='cuda')
                for k, v in model_inputs.items()
            }

            #model_inputs = {n: m.to('cuda') for n, m in model_inputs.items()}

    input_length = len(model_inputs['input_ids'][0])
    if args.enable_thinking: 
        max_gen_tokens = 1024
    else: 
        max_gen_tokens = 512  # up to 256, scaled to input length

    
    output = model.generate(
        **model_inputs,
        max_new_tokens=max_gen_tokens,
        top_k=50,
        top_p=0.9,
        temperature=0.5,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,
#     early_stopping=True
        )
    
    document = tokenizer.batch_decode(output, skip_special_tokens=True)
    if 'gemma' in args.model_name:
        document = document[0].split('model\n')[1].strip()
    else:
        if 'qwen3' in args.model_name:
            document = document[0].split('\n</think>\n')[1].strip()
        else:
            document = document[0].split('assistant')[1].strip()
    
    
    if 'test' in args.dataset:
        dic = {
            'document': input_text,
            'generated_caption': document
            }
    else:
        dic = {
            'document': input_text,
            'reference': gold,
            'generated_caption': document
        }

    data.append(dic)

if 'test' in args.dataset:
    os.makedirs('../results_test', exist_ok=True)
    with open(f'../results_test/{args.output}', 'w') as file:
        json.dump(data, fp=file, indent=2)
else:
    os.makedirs('../results', exist_ok=True)
    with open(f'../results/{args.output}', 'w') as f:
        json.dump(data, fp=f, indent=2)

    