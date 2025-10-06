#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

## write research question agent model path here
research_question_agent_model_path=""
gpu_memory_utilization=1.0

port_number=8082
python3 -m vllm.entrypoints.openai.api_server \
    --model ${research_question_agent_model_path} \
    --dtype half \
    --port ${port_number} \
    --gpu-memory-utilization ${gpu_memory_utilization}

