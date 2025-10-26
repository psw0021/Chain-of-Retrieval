#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

## write research question agent model path here(trained with QWEN-2.5-3B-Instruct model)
research_question_agent_model_path=""
gpu_memory_utilization=1.0
max_model_len=131072

port_number=8082
python3 -m vllm.entrypoints.openai.api_server \
    --model ${research_question_agent_model_path} \
    --dtype half \
    --port ${port_number} \
    --gpu-memory-utilization ${gpu_memory_utilization}  \
    --rope_scaling '{"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}' \
    --max_model_len ${max_model_len}

