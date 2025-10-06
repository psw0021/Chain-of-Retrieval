#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

## write method agent model path here
method_agent_model_path=""
gpu_memory_utilization=1.0

port_number=8083
python3 -m vllm.entrypoints.openai.api_server \
    --model ${method_agent_model_path} \
    --dtype half \
    --port ${port_number} \
    --gpu-memory-utilization ${gpu_memory_utilization}

