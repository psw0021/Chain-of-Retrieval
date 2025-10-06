#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

## write experiment agent model path here
experiment_agent_model_path=""
gpu_memory_utilization=1.0

port_number=8081
python3 -m vllm.entrypoints.openai.api_server \
    --model ${experiment_agent_model_path} \
    --dtype half \
    --port ${port_number} \
    --gpu-memory-utilization ${gpu_memory_utilization}

