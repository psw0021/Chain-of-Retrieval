#!/bin/bash
echo $CUDA_VISIBLE_DEVICES

# conda environment
CONDA_ENV_NAME="paper_retrieval"
### Environmental variables

use_query_optimizer=True
use_full_paper_as_query=False
multi_agent=True
query_optimizer_model="gpt-4o-mini-2024-07-18"
gpu_memory_utilization=0.7
use_gpt=True
benchmark_directory=""
embedding_model="bge-m3"
top_k=300
max_top_k=300
use_RRF_for_chunked_baseline=False
hyperparameter_RRF=60
corpus_directory=""
batch_size=1
include_original_retrieval=False
use_base_agent=False
use_method_agent=True
use_experiment_agent=True
use_research_question_agent=True
use_multi_source=False
use_chunked=True
chunk_unit=3000
use_full_paper_as_corpus=False
embedding_fusion=False
embedding_fuse_method="maximum_similarity"
use_trained_model=False
remove_citations=True 
remove_mentions=True


iterative_retrieval=False
SubTreeSearch=False
Recursive_Merge=False
use_aspect_aware_cache_for_selection=False
total_iterative_retrieval_loop=5
starting_iteration_of_A2A_only=-1
input_top_k_for_verifier=1
selector_starting_idx=1


if [ ${use_trained_model} == True ]; then
    method_agent_model_path=""
    experiment_agent_model_path=""
    research_question_agent_model_path=""

else
    method_agent_model_path=0
    experiment_agent_model_path=0
    research_question_agent_model_path=0
    
    echo ${method_agent_model_path}
fi

if [[ ${use_query_optimizer} == False && ${multi_agent} == False ]]; then
    use_multi_source=False
fi

function run_job() {
    local start_idx=$1
    local end_idx=$2
    local job_num=$3
    local iteration=$4
    echo "Running inference_retrieval.py"
    rm logs/logs/output_${job_num}.log 
    python Inference/inference_retrieval.py \
    --iteration_num ${iteration}    \
    --batch_size ${batch_size} \
    --use_query_optimizer ${use_query_optimizer} \
    --use_full_paper_as_query ${use_full_paper_as_query} \
    --multi_agent ${multi_agent}    \
    --query_optimizer_model ${query_optimizer_model} \
    --gpu_memory_utilization ${gpu_memory_utilization}  \
    --use_gpt ${use_gpt} \
    --corpus_directory ${corpus_directory} \
    --start_idx ${start_idx} \
    --end_idx ${end_idx} \
    --include_original_retrieval ${include_original_retrieval} \
    --use_base_agent ${use_base_agent}  \
    --use_method_agent ${use_method_agent} \
    --use_experiment_agent ${use_experiment_agent} \
    --use_research_question_agent ${use_research_question_agent} \
    --benchmark_directory ${benchmark_directory} \
    --remove_citations ${remove_citations}  \
    --remove_mentions ${remove_mentions}    \
    --embedding_model ${embedding_model} \
    --top_k ${top_k} \
    --max_top_k ${max_top_k} \
    --use_RRF_for_chunked_baseline ${use_RRF_for_chunked_baseline} \
    --hyperparameter_RRF ${hyperparameter_RRF} \
    --use_multi_source ${use_multi_source} \
    --use_chunked ${use_chunked} \
    --chunk_unit ${chunk_unit}  \
    --use_full_paper_as_corpus ${use_full_paper_as_corpus}  \
    --use_trained_model ${use_trained_model} \
    --method_agent_model_path ${method_agent_model_path} \
    --experiment_agent_model_path ${experiment_agent_model_path} \
    --research_question_agent_model_path ${research_question_agent_model_path} \
    --embedding_fusion ${embedding_fusion}  \
    --embedding_fuse_method ${embedding_fuse_method}    \
    --iterative_retrieval ${iterative_retrieval}    \
    --Recursive_Merge ${Recursive_Merge} \
    --use_aspect_aware_cache_for_selection ${use_aspect_aware_cache_for_selection} \
    --SubTreeSearch ${SubTreeSearch} \
    --total_iterative_retrieval_loop ${total_iterative_retrieval_loop}  \
    --starting_iteration_of_A2A_only ${starting_iteration_of_A2A_only} \
    --input_top_k_for_verifier ${input_top_k_for_verifier} \
    --selector_starting_idx ${selector_starting_idx} > logs/logs/output_${job_num}.log 2>&1
}


for i in {1..1}; do
    echo "Iteration $i"
    iteration=$i
    current_start=0  # Reset for each iteration
        
    total_indices=400
    indices_per_thread=400

    job_count=0
    while [ "$current_start" -lt "$total_indices" ]; do
        if [ "$job_count" -eq 0 ]; then
            export CUDA_VISIBLE_DEVICES=3
        elif [ "$job_count" -eq 1 ]; then
            export CUDA_VISIBLE_DEVICES=3
        elif [ "$job_count" -eq 2 ]; then
            export CUDA_VISIBLE_DEVICES=3
        elif [ "$job_count" -eq 3 ]; then
            export CUDA_VISIBLE_DEVICES=3
        elif [ "$job_count" -eq 4 ]; then
            export CUDA_VISIBLE_DEVICES=2
        elif [ "$job_count" -eq 5 ]; then
            export CUDA_VISIBLE_DEVICES=5
        elif [ "$job_count" -eq 6 ]; then
            export CUDA_VISIBLE_DEVICES=5
        elif [ "$job_count" -eq 7 ]; then
            export CUDA_VISIBLE_DEVICES=5
        elif [ "$job_count" -eq 8 ]; then
            export CUDA_VISIBLE_DEVICES=5
        elif [ "$job_count" -eq 9 ]; then
            export CUDA_VISIBLE_DEVICES=5
        else
            echo "Invalid JOB_ID: $job_count"
            exit 1
        fi

        current_end=$((current_start + indices_per_thread))
        if [ "$current_end" -gt "$total_indices" ]; then
            current_end=$total_indices
        fi

        # Run the job in background
        run_job $current_start $current_end $job_count $iteration &

        ((job_count++))
        current_start=$current_end
    done

    # Wait for all background jobs in this iteration to finish
    wait
done
