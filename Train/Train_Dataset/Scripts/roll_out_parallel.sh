#!/bin/bash

# conda environment
CONDA_ENV_NAME="paper_retrieval"
### Environmental variables

query_optimizer_model="meta-llama/Llama-3.2-3B-Instruct"
use_gpt=False
deploy_llm=True
train_set_directory="Train_Dataset/Final_Train_Set/Papers_and_Candidates"
embedding_model="jina-embeddings-v2-base-en"
top_k=300
max_top_k=300
corpus_directory="Train_Dataset/Final_Train_Set/Target_Corpus/corpus.json"
batch_size=2
include_original_retrieval=False
use_method_agent=True
use_experiment_agent=True
use_research_question_agent=True
query_detailedness=3
use_multi_source=False
use_chunked=True
rollout_number=16
gpu_memory_utilization=0.8
repetition_penalty=1.2
max_tokens=2000
temperature=0.7
top_p=0.8

## absolute path to the parent directory benchmark
absolute_path="/c2/swpark/Process_Code_for_Submission/Chain-of-Retrieval"


if [[ ${use_query_optimizer} == False && ${multi_agent} == False ]]; then
    use_multi_source=False
fi

function run_job() {
    local start_idx=$1
    local end_idx=$2
    local job_num=$3
    echo "Running roll_out.py"
    rm logs/output_${job_num}.log 
    python Train_Dataset/Create_Preference_Dataset/roll_out.py \
    --batch_size ${batch_size} \
    --query_optimizer_model ${query_optimizer_model} \
    --use_gpt ${use_gpt} \
    --deploy_llm ${deploy_llm}  \
    --corpus_directory ${corpus_directory} \
    --start_idx ${start_idx} \
    --end_idx ${end_idx} \
    --include_original_retrieval ${include_original_retrieval} \
    --use_method_agent ${use_method_agent} \
    --use_experiment_agent ${use_experiment_agent} \
    --use_research_question_agent ${use_research_question_agent} \
    --train_set_directory ${train_set_directory} \
    --embedding_model ${embedding_model} \
    --query_detailedness ${query_detailedness} \
    --top_k ${top_k} \
    --max_top_k ${max_top_k} \
    --use_multi_source ${use_multi_source} \
    --use_chunked ${use_chunked} \
    --rollout_number ${rollout_number} \
    --repetition_penalty ${repetition_penalty}    \
    --max_tokens ${max_tokens}   \
    --temperature ${temperature}   \
    --top_p ${top_p} \
    --gpu_memory_utilization ${gpu_memory_utilization} > logs/output_${job_num}.log 2>&1
}

total_indices=15665
indices_per_thread=15665

current_start=0
job_count=0
while [ "$current_start" -lt "$total_indices" ]; do
    current_end=$((current_start + indices_per_thread))
    if [ "$current_end" -gt "$total_indices" ]; then
        current_end=$total_indices
    fi
    ((job_count++))

    if [ "$job_count" -eq 1 ]; then
        export CUDA_VISIBLE_DEVICES=4
        echo $CUDA_VISIBLE_DEVICES
    elif [ "$job_count" -eq 2 ]; then
        export CUDA_VISIBLE_DEVICES=3
        echo $CUDA_VISIBLE_DEVICES
    elif [ "$job_count" -eq 3 ]; then
        export CUDA_VISIBLE_DEVICES=3,4
        echo $CUDA_VISIBLE_DEVICES
    elif [ "$job_count" -eq 4 ]; then
        export CUDA_VISIBLE_DEVICES=1,5
        echo $CUDA_VISIBLE_DEVICES
    elif [ "$job_count" -eq 5 ]; then
        export CUDA_VISIBLE_DEVICES=2,5
        echo $CUDA_VISIBLE_DEVICES
    fi

    # Run the job
    run_job $current_start $current_end $job_count &

    # Increment start index for next job
    current_start=$current_end
done

### Wait for all jobs to complete
wait