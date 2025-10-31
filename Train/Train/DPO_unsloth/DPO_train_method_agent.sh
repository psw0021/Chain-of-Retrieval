#!/bin/bash

## Setup WandB Environment Variables, and Cache Directory
export WANDB_LOG_MODEL=""
export WANDB_DIR=""
export WANDB_CACHE_DIR=""
export WANDB_DATA_DIR=""
export CUDA_VISIBLE_DEVICES=6


model_name="meta-llama/Llama-3.2-3B-Instruct"
agent_name="method_agent"
dataset="Jackson0018/Preference_Set_Llama-3.2-3B-Instruct_BGE_ref_as_gt_True_IterRet_individual_recall_True_top_k_30" ## Preference Dataset to train Method-Focused Query Optimizer Agents
output_dir="Trained_Models/${dataset}/${model_name}/${agent_name}"
bits=4
report_to="wandb"
save_strategy="steps"
save_steps=10000
project_name="" # wandb project name
learning_rate=1e-5
weight_decay=0.01
max_grad_norm=0.6
logging_steps=10
epoch=3
gradient_accumulation_steps=32
per_device_train_batch_size=1
beta=0.1
max_length=4000
max_prompt_length=2000
max_completion_length=2000
load_in_4bit=True


python Train/DPO_unsloth/DPO_train_unsloth.py \
    --model_name_or_path ${model_name} \
    --dataset ${dataset}    \
    --agent_name ${agent_name}  \
    --output_dir  ${output_dir}\
    --project_name ${project_name}  \
    --full_finetune ${full_finetune} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --bits ${bits}  \
    --beta ${beta}  \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --weight_decay ${weight_decay}  \
    --max_grad_norm ${max_grad_norm}  \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --save_strategy ${save_strategy} \
    --max_length ${max_length}  \
    --max_prompt_length ${max_prompt_length}  \
    --max_completion_length ${max_completion_length}\
    --save_steps ${save_steps}  \
    --num_train_epochs ${epoch} \
    --learning_rate ${learning_rate} \
    --load_in_4bit ${load_in_4bit}  \
    --logging_steps ${logging_steps} \
    --report_to  ${report_to}\
    --evaluation_strategy no > logs/output_train_${agent_name}_DPO.log 2>&1