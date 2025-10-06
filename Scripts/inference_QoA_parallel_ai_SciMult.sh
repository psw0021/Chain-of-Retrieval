#!/bin/bash
use_full_paper_as_query=False
benchmark_directory="LongDocumentBench/ScientificPapers/Final_Dataset_w_citations_mentions_removed/Benchmark/EMNLP/Cited_Papers"
embedding_model="SciMult-MoE"
MoE_task='multi-hop-retrieval'
top_k=300
max_top_k=300
corpus_directory="LongDocumentBench/ScientificPapers/Final_Dataset_w_citations_mentions_removed/Target_Corpus/target_corpus_citations_removed_True_mentions_removed_True/corpus.json"
batch_size=4
use_multi_source=False
use_chunked=False
chunk_unit=3000
use_full_paper_as_corpus=False
remove_citations=True 
remove_mentions=True


function run_job() {
    local start_idx=$1
    local end_idx=$2
    local job_num=$3
    local iteration=$4
    echo "Running inference_scimult_retrieval.py"
    rm logs/logs_scimult/output_${job_num}.log 
    python Inference/inference_scimult_retrieval.py \
    --iteration_num ${iteration}    \
    --batch_size ${batch_size} \
    --use_full_paper_as_query ${use_full_paper_as_query} \
    --corpus_directory ${corpus_directory} \
    --start_idx ${start_idx} \
    --end_idx ${end_idx} \
    --benchmark_directory ${benchmark_directory} \
    --remove_citations ${remove_citations}  \
    --remove_mentions ${remove_mentions}    \
    --embedding_model ${embedding_model} \
    --MoE_task ${MoE_task}  \
    --top_k ${top_k} \
    --max_top_k ${max_top_k} \
    --use_chunked ${use_chunked} \
    --chunk_unit ${chunk_unit}  \
    --use_full_paper_as_corpus ${use_full_paper_as_corpus}  > logs/logs_scimult/output_${job_num}.log 2>&1
}




for i in {1..1}; do
    echo "Iteration $i"
    iteration=$i
    current_start=0  # Reset for each iteration
        
    total_indices=400
    indices_per_thread=100
    export CUDA_VISIBLE_DEVICES=3

    job_count=0
    while [ "$current_start" -lt "$total_indices" ]; do
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