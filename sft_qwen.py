from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, AutoPeftModelForCausalLM
import torch
import os
import json

dataset = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track", split='train')

dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_set = dataset['train']
eval_set = dataset['test']

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", quantization_config=bnb_config, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", device_map='auto')

"""with open('sample100.json', 'r') as f:
    data = json.load(f)"""

def formatting_prompts_func(example):
    text = f"### Rewrite the following radiology report in layman's terms.\n ### Radiology Report: {example['radiology_report']}\n ### Layman's Report: {example['layman_report']}"
    return text

def find_target_modules(model):
    unique_layers = set()
    
    for name, module in model.named_modules():
        if "Linear4bit" in str(type(module)):
            layer_type = name.split('.')[-1]
            
            unique_layers.add(layer_type)

    return list(unique_layers)

#response_template = " ### Layman's Report:"
#collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

os.makedirs('./models', exist_ok=True)
training_args = SFTConfig(
    output_dir='./models/Qwen',
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=5,
    weight_decay=0.1,
    learning_rate=5e-4,
    packing=True,
)

peft_config = LoraConfig(
    r=32, 
    lora_alpha=64, 
    lora_dropout=0.1,
    target_modules=find_target_modules(model),
    bias="none",
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model,
    tokenizer=tokenizer,
    train_dataset=train_set,
    eval_dataset=eval_set,
    args=training_args,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
  #  data_collator=collator
)

trainer.train()
trainer.save_model('./models/Qwen')

trainer.model.save_pretrained('./models/Qwen/final_checkpoint')
del model

model = AutoPeftModelForCausalLM.from_pretrained('./models/Qwen/final_checkpoint', device_map='auto', torch_dtype=torch.bfloat16)
model = model.merge_and_unload()

model.save_pretrained('./models/Qwen/merged_final', safe_serialization=True)