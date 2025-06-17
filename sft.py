from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from huggingface_hub import login
import torch
import torch.nn as nn
import os
import wandb
import json
import argparse
from random import sample
from tqdm import tqdm
import numpy as np 
from rouge_score.rouge_scorer import RougeScorer

class LayerNorm32(nn.LayerNorm):
    def forward(self, input):
        orig_dtype = input.dtype
        output = super().forward(input.to(torch.float32))
        return output.to(orig_dtype)

def convert_layernorm_to_fp32(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            parent = model
            names = name.split('.') 
            for n in names[:-1]:
                parent = getattr(parent, n)
            setattr(parent, names[-1], LayerNorm32(module.normalized_shape, eps=module.eps, elementwise_affine=module.elementwise_affine))
    return model

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="Model to fine-tuned [qwen | qwen-instruct | llama | llama-instruct]", default='qwen')
    parser.add_argument('--few_shot', action='store_true')
    parser.add_argument('--no-few_shot', dest='few-shot', action='store_false')
    parser.add_argument('--packing', action='store_true')
    parser.add_argument('--no-packing', dest='packing', action='store_false')
    parser.add_argument('--per_device_train_batch_size', type=int, default=5)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--eval_accumulation_steps', type=int, default=16)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--max_seq_length', type=int, default=200)
    parser.add_argument('--evaluation_strategy', type=str, default=None)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--eval_steps', type=int, default=0)
    parser.add_argument('--save_steps', type=int, default=0)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--no-bf16', dest='bf16', action='store_false')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--no-fp16', dest='fp16', action='store_false')
    parser.add_argument('--report_to', type=str, default='wandb')
    
    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=64)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--lora_bias', type=str, default='none')

    parser.set_defaults(packing=True)
    parser.set_defaults(few_shot=False)
    parser.set_defaults(bf16=False)
    parser.set_defaults(fp16=False)
    return parser


def formatting_prompts_func(example):    
    text = f"### Radiology Report: {example['radiology_report']}\n\n### Layman Report: {example['layman_report']}"
    return text

"""def format_prompt(examples):
    reports = examples["radiology_report"]
    laymans = examples["layman_report"]
    prompts = []

    for report, layman in zip(reports, laymans):
        prompt = [
          {"role": "system",
           "content": [
             {"type": "text",
              "text": "You are a radiology report translator who rewrites radiology reports into layman's terms."}
           ]},
          {"role": "user",
           "content": [
             {"type": "text",
              "text": f"Radiology Report: {report}\n\nLayman Report: {layman}"}
           ]}
        ]
        prompts.append(prompt)

    return {"prompt": prompts}"""

def format_prompt(examples):
    instruction = "Rewrite the following radiology report in layman's terms. Avoid using medical jargon, and write concisely."
    prompts = []
    radiology_reports = examples['radiology_report']
    layman_reports = examples['layman_report']
    for radiology, layman in zip(radiology_reports, layman_reports):
        text = f"""<start_of_turn>user\n {instruction}\n\nRadiology report: {radiology}\n\n<end_of_turn>\n<start_of_turn>model\n Layman report:\n{layman} <end_of_turn>"""
        prompts.append(text)
    return {'prompt': prompts}


# If using DataCollatorForCompletionOnlyLM
"""def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['radiology_report'])):
        text = f"### Radiology Report: {example['radiology_report']}\n\n### Layman Report: {example['layman_report']}"
        output_texts.append(text)
    return output_texts"""


def find_target_modules(model):
    unique_layers = set()
    
    for name, module in model.named_modules():
        if "Linear4bit" in str(type(module)):
            layer_type = name.split('.')[-1]
            
            unique_layers.add(layer_type)

    return list(unique_layers)

def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                logits = logits[0]
            return logits.argmax(dim=-1)

scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

  
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(pred.strip()) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.strip()) for label in decoded_labels]
    result = scorer.score(
        predictions=decoded_preds, references=decoded_labels
    )
    return {'ROUGE': (np.mean(result['rouge1'].fmeasure) + np.mean(result['rouge2'].fmeasure) + np.mean(result['rougel'].fmeasure))/3.0}
        
if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()

    login(os.environ['HUGGINGFACE_TOKEN'])
    wandb.login(key=os.environ['WANDB_TOKEN'])


    experiment_name = f"SFT_lr_{args.learning_rate}_ep_{args.num_train_epochs}_wd_{args.weight_decay}_r_{args.lora_r}_alpha_{args.lora_alpha}_dropout_{args.lora_dropout}_bias_{args.lora_bias}"

    wandb.init(
        project=f"BioLaySumm_{args.model}",
        name=experiment_name,
        config = {
            'epochs': args.num_train_epochs,
            'lr': args.learning_rate,
            'weight_decay': args.weight_decay,
            'lora': {
                'r': args.lora_r,
                'alpha': args.lora_alpha,
                'dropout': args.lora_dropout,
                'bias': args.lora_bias
            }
        },
        reinit=True
    )

    dataset = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track", split='train')

    if 'gemma' in args.model.lower():
        dataset = dataset.map(format_prompt, batched=True)

    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_set = dataset['train']
    eval_set = dataset['test']

    

    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

    if args.model.lower() == 'qwen':
        model_name = 'Qwen/Qwen2.5-3B'
    elif 'qwen-instruct' in args.model.lower():
        if '3b' in args.model.lower():
            model_name = 'Qwen/Qwen2.5-3B-Instruct'
        elif '7b' in args.model.lower():
            model_name = 'Qwen/Qwen2.5-7B-Instruct'
    elif 'qwen3' in args.model.lower():
        if '8b' in args.model.lower():
            model_name = 'Qwen/Qwen3-8B'
        else:
            model_name = 'Qwen/Qwen3-4B'
    elif args.model.lower() == 'llama':
        model_name = 'meta-llama/Llama-3.2-3B'
    elif args.model.lower() == 'llama-instruct':
        model_name = 'meta-llama/Llama-3.2-3B-Instruct'
    elif args.model.lower() == 'gemma':
        model_name = 'google/gemma-3-4b-it'
    else:
        raise Exception('Not an applicable model name. Applicable models: [qwen|qwen-instruct|llama|llama-instruct]')

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map='auto',
        attn_implementation="eager" if 'gemma' in model_name else "flash_attention_2",
        torch_dtype=torch.bfloat16
        )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        device_map='auto',
        add_eos_token=True
        )
    base_model.gradient_checkpointing_enable()
    
    base_model = prepare_model_for_kbit_training(base_model)

    if 'gemma' in model_name.lower():
        base_model = convert_layernorm_to_fp32(base_model)

    if tokenizer.eos_token is None:
        tokenizer.eos_token = '</s>'
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.add_special_tokens({'eos_token': tokenizer.eos_token})
    base_model.resize_token_embeddings(len(tokenizer))

    response_template = "### Layman Report:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    os.makedirs('./models', exist_ok=True)
    output_dir = f'./models/{args.model}_{experiment_name}'
    
    training_args = SFTConfig(
        output_dir=output_dir,
      #  dataset_text_field="prompt",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_seq_length=args.max_seq_length,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        packing=args.packing,
        bf16=args.bf16,
        fp16=args.fp16,
        label_names=["labels"],
        optim='adamw_8bit',
        lr_scheduler_type='linear',
    )

    peft_config = LoraConfig(
        r=args.lora_r, 
        lora_alpha=args.lora_alpha, 
        lora_dropout=args.lora_dropout,
        target_modules='all-linear',
        bias=args.lora_bias,
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        base_model,
        train_dataset=train_set,
        eval_dataset=eval_set,
        args=training_args,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func if 'gemma' not in model_name.lower() else None,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
      #  data_collator=collator # use only if packing=False
        )

    torch.cuda.empty_cache()

    trainer.train()

    torch.cuda.empty_cache()

    trainer.save_model(output_dir)

    trainer.model.save_pretrained(f'{output_dir}/final_checkpoint')
    tokenizer.save_pretrained(f'{output_dir}/final_checkpoint')

    wandb.finish()

    model = PeftModel.from_pretrained(base_model, f'{output_dir}/final_checkpoint', device_map='auto', torch_dtype=torch.bfloat16)
    del base_model

    model = model.merge_and_unload()


    model.save_pretrained(f'{output_dir}/merged_final', safe_serialization=True)
    tokenizer.save_pretrained(f'{output_dir}/merged_final', safe_serialization=True)
   
    if args.model == 'qwen-instruct':
        model_name = 'qwen-instruct-7B'

    model.push_to_hub(repo_id=f'daniazie/{model_name}_SFT')

    with open('model_name.txt', 'a') as file:
        file.write(output_dir + '\n')

    torch.cuda.empty_cache()
