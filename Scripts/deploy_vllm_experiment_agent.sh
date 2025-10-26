#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

## write experiment agent model path here
experiment_agent_model_path="/c2/swpark/Process_Code_for_Submission/Chain-of-Retrieval/Models/Preference_Set_Llama-3.2-3B-Instruct_INFV_ref_as_gt_True_IterRet_individual_recall_True_top_k_30/meta-llama/Llama-3.2-3B-Instruct/experiment_agent/merged_model"
gpu_memory_utilization=1.0

port_number=8081
python3 -m vllm.entrypoints.openai.api_server \
    --model ${experiment_agent_model_path} \
    --dtype half \
    --port ${port_number} \
    --gpu-memory-utilization ${gpu_memory_utilization}

