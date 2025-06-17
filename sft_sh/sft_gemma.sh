#!/usr/bin/bash

python sft.py \
--model="gemma" \
--packing \
--per_device_train_batch_size=4 \
--per_device_eval_batch_size=4 \
--gradient_accumulation_steps=32 \
--num_train_epochs=5 \
--weight_decay=0.1 \
--learning_rate=5e-4 \
--max_seq_length=1024 \
--logging_steps=500 \
--eval_steps=0 \
--save_steps=0 \
--report_to="wandb" \
--lora_r=64 \
--lora_alpha=128 \
--lora_dropout=0.1 \
--lora_bias="none"


