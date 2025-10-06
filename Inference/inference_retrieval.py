import argparse
import os
import sys
import json
from pathlib import Path
import logging
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
from Utils.utils import *
from Retrieval.retriever import Retriever


def DatasetConfig():
    parser = argparse.ArgumentParser(
        description="Configurations related to Dataset"
    )
    parser.add_argument("--benchmark_directory", default="")
    parser.add_argument("--remove_citations", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--remove_mentions", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--corpus_directory", default="Dataset/Ours/Formatted_Candidate_Paper_Corpus/ICLR_2024/Direct_References/corpus.json")

    args, rem = parser.parse_known_args()
    
    return args

def QueryOptimizerConfig():
    parser = argparse.ArgumentParser(
        description="Configurations related to QueryOptimizers"
    )

    parser.add_argument("--use_query_optimizer", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--multi_agent", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--query_optimizer_model", default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--use_gpt", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--use_base_agent", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--use_method_agent", default=True, type=lambda x: x.lower() == "true")
    parser.add_argument("--use_experiment_agent", default=True, type=lambda x: x.lower() == "true")
    parser.add_argument("--use_research_question_agent", default=True, type=lambda x: x.lower() == "true")
    parser.add_argument("--method_agent_model_path", default="", type=str)
    parser.add_argument("--experiment_agent_model_path", default="", type=str)
    parser.add_argument("--research_question_agent_model_path", default="", type=str)
    parser.add_argument("--use_trained_model", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--temperature", default=0, type=str)
    parser.add_argument("--max_tokens", default=2000, type=int)
    parser.add_argument("--gpu_memory_utilization", default=0.7, type=float)
    parser.add_argument("--repetition_penalty", default=1.2, type=float)

    args, rem = parser.parse_known_args()
    
    return args


def RetrievalConfig():
    parser = argparse.ArgumentParser(
        description="Configurations related to retrieval setup"
    )

    parser.add_argument("--iteration_num", default=0, type=int, choices=[1, 2, 3])
    parser.add_argument("--embedding_model", default="jina-embeddings-v2-base-en", choices=["SPECTER", "text-embedding-3-small", \
        "jina-embeddings-v2-base-en", "SPECTER2", "SPECTER2_Base", \
            "SciNCL", "text-embedding-ada-002", "bge-m3", "dewey_en_beta", \
               "Qwen3-Embedding-0.6B", "granite-embedding-english-r2", "inf-retriever-v1-1.5b"])
    parser.add_argument("--embedding_fusion", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--embedding_fuse_method", default="naive_aggregation", choices=["naive_aggregation", "maximum_similarity"])
    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--end_idx", default=500, type=int)
    parser.add_argument("--top_k", default=100, type=int)
    parser.add_argument("--max_top_k", default=100, type=int)
    parser.add_argument("--use_RRF_for_chunked_baseline", default=True, type=lambda x: x.lower() == "true")
    parser.add_argument("--hyperparameter_RRF", default=60, type=int)
    parser.add_argument("--include_original_retrieval", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--use_multi_source", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--use_chunked", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--use_full_paper_as_corpus", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--chunk_unit", default=3000, type=int)
    parser.add_argument("--batch_size", default=500, type=int)
    parser.add_argument("--use_full_paper_as_query", default=False, type=lambda x: x.lower() == "true")


    args, rem = parser.parse_known_args()
    
    return args

def IterativeRetrievalConfig():
    parser = argparse.ArgumentParser(
        description="Configuration for Verifiers"
    )

    parser.add_argument("--iterative_retrieval", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--SubTreeSearch", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--Recursive_Merge", default=True, type=lambda x: x.lower() == "true")
    parser.add_argument("--use_aspect_aware_cache_for_selection", default=True, type=lambda x: x.lower() == "true")
    parser.add_argument("--total_iterative_retrieval_loop", default=3, type=int)
    parser.add_argument("--starting_iteration_of_A2A_only", default=-1, type=int)
    parser.add_argument("--input_top_k_for_verifier", default=1, type=int)
    parser.add_argument("--selector_starting_idx", default=1, type=int)
    
    
    args, rem = parser.parse_known_args()
    
    return args


def merge_configs(dataset_config, retrieval_config, query_optimizer_config, iterative_retrieval_config):
    merged_dict = {**vars(retrieval_config), **vars(dataset_config), **vars(query_optimizer_config), **vars(iterative_retrieval_config)}
    
    return argparse.Namespace(**merged_dict)

def save_config(config, result_path):
    """Save the configuration dictionary as JSON."""
    config_to_save = vars(config)
    config_path = os.path.join(result_path, "config.json")

    with open(config_path, "w") as f:
        json.dump(config_to_save, f, indent=4)

    print(f"Config saved at {config_path}")

def evaluate(args):
    ## Bencharmk directory
    path = Path(args.benchmark_directory)
    last_two_dirs = "/".join(path.parts[-3:])
    result_directory = "Results"

    if args.use_chunked == True and args.use_full_paper_as_corpus == True:
        raise TypeError("Target Corpus cannot be both chunked and not chunked")
    
    if args.iterative_retrieval == False and args.SubTreeSearch == True:
        raise TypeError("SubTreeSearch always refers to Iterative Retrieval")
    
    if args.embedding_model ==  "SPECTER2" or args.embedding_model == "SPECTER2_Base" or args.embedding_model == "SciNCL":
        if args.use_query_optimizer == True:
            raise TypeError("When Embedding models from SPECTER2 are used, you should not experiment with query optimizer")
    
    if args.SubTreeSearch == True:
        if args.starting_iteration_of_A2A_only > 0:
            if args.use_multi_source == False:
                raise TypeError("When trying to use abstract chaining for SubTreeSearch, you must use multiple sources")
            if args.include_original_retrieval == False:
                raise TypeError("When trying to use abstract chaining for SubTreeSearch, you must include original A2A retrieval")
            
    if args.use_base_agent == True:
        if args.use_method_agent == True:
            raise TypeError("Cannot use other aspect-specific agents when using base agents")
        elif args.use_experiment_agent == True:
            raise TypeError("Cannot use other aspect-specific agents when using base agents")
        elif args.use_research_question_agent == True:
            raise TypeError("Cannot use other aspect-specific agents when using base agents")
            
    #if args.use_multi_source == False:
        #if args.include_original_retrieval == True:
            #raise TypeError("When trying to use single source retrieval, you must not include abstract to abstract retrieval")

    if args.embedding_fusion == True:
        if args.use_query_optimizer == False or args.multi_agent == False:
            raise TypeError("When experimenting with embedding fusion, you must always generate multiple queries with query optimizers")
        
    if args.starting_iteration_of_A2A_only > 0:
        if int(args.starting_iteration_of_A2A_only) > args.total_iterative_retrieval_loop:
            raise TypeError("Must set the starting iteration of using A2A only equal or under the total iterative retrieval loops or search depth")

    if args.iterative_retrieval == False:
        if args.use_query_optimizer == True:
            if args.embedding_fusion == True:
                result_folder_name = f"{last_two_dirs}/remove_citations_{args.remove_citations}_remove_mentions_{args.remove_mentions}/Use_QueryOptimizers/use_trained_model_{args.use_trained_model}/{args.query_optimizer_model}/use_multi_source_{args.use_multi_source}/multi_agent_{args.multi_agent}_INCLUDE_ORIGINAL_RETRIEVAL_{args.include_original_retrieval}_BASE_{args.use_base_agent}_METHOD_{args.use_method_agent}_RESEARCH_QUESTION_{args.use_research_question_agent}_EXPERIMENT_{args.use_experiment_agent}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.chunk_unit}_{args.use_chunked}_use_full_paper_as_corpus_{args.use_full_paper_as_corpus}_fuse_embedding_{args.embedding_fusion}_{args.embedding_fuse_method}/iteration_{args.iteration_num}"
            elif args.embedding_fusion == False:
                if args.use_trained_model == False:
                    result_folder_name = f"{last_two_dirs}/remove_citations_{args.remove_citations}_remove_mentions_{args.remove_mentions}/Use_QueryOptimizers/use_trained_model_{args.use_trained_model}/{args.query_optimizer_model}/use_multi_source_{args.use_multi_source}/multi_agent_{args.multi_agent}_INCLUDE_ORIGINAL_RETRIEVAL_{args.include_original_retrieval}_BASE_{args.use_base_agent}_METHOD_{args.use_method_agent}_RESEARCH_QUESTION_{args.use_research_question_agent}_EXPERIMENT_{args.use_experiment_agent}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.chunk_unit}_{args.use_chunked}_use_full_paper_as_corpus_{args.use_full_paper_as_corpus}/iteration_{args.iteration_num}"
                elif args.use_trained_model == True:
                    try:
                        path = Path(args.method_agent_model_path)
                        train_dataset_name = path.parents[3].name
                    except:
                        pass

                    try:
                        path = Path(args.experiment_agent_model_path)
                        train_dataset_name = path.parents[3].name
                    except:
                        pass

                    try:
                        path = Path(args.research_question_agent_model_path)
                        train_dataset_name = path.parents[3].name
                    except:
                        pass

                    result_folder_name = f"{last_two_dirs}/remove_citations_{args.remove_citations}_remove_mentions_{args.remove_mentions}/Use_QueryOptimizers/use_trained_model_{args.use_trained_model}/{train_dataset_name}/{args.query_optimizer_model}/use_multi_source_{args.use_multi_source}/multi_agent_{args.multi_agent}_INCLUDE_ORIGINAL_RETRIEVAL_{args.include_original_retrieval}_BASE_{args.use_base_agent}_METHOD_{args.use_method_agent}_RESEARCH_QUESTION_{args.use_research_question_agent}_EXPERIMENT_{args.use_experiment_agent}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.chunk_unit}_{args.use_chunked}_use_full_paper_as_corpus_{args.use_full_paper_as_corpus}/iteration_{args.iteration_num}"
                
        else:
            if args.use_full_paper_as_query == False:
                if args.use_chunked == False:
                    result_folder_name = f"{last_two_dirs}/remove_citations_{args.remove_citations}_remove_mentions_{args.remove_mentions}/No_QueryOptimizers/use_query_optimizer_{args.use_query_optimizer}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.chunk_unit}_{args.use_chunked}_use_full_paper_as_corpus_{args.use_full_paper_as_corpus}/iteration_{args.iteration_num}"
                elif args.use_chunked == True:
                    if args.use_RRF_for_chunked_baseline == False:
                        result_folder_name = f"{last_two_dirs}/remove_citations_{args.remove_citations}_remove_mentions_{args.remove_mentions}/No_QueryOptimizers/use_query_optimizer_{args.use_query_optimizer}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.chunk_unit}_{args.use_chunked}_use_full_paper_as_corpus_{args.use_full_paper_as_corpus}/iteration_{args.iteration_num}"
                    elif args.use_RRF_for_chunked_baseline == True:
                        result_folder_name = f"{last_two_dirs}/remove_citations_{args.remove_citations}_remove_mentions_{args.remove_mentions}/No_QueryOptimizers/use_query_optimizer_{args.use_query_optimizer}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.chunk_unit}_{args.use_chunked}_use_full_paper_as_corpus_{args.use_full_paper_as_corpus})_use_RRF_for_chunked_baseline_{args.use_RRF_for_chunked_baseline}/iteration_{args.iteration_num}"
            
            elif args.use_full_paper_as_query == True:
                if args.use_chunked == False: 
                    result_folder_name = f"{last_two_dirs}/remove_citations_{args.remove_citations}_remove_mentions_{args.remove_mentions}/No_QueryOptimizers/use_query_optimizer_{args.use_query_optimizer}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.chunk_unit}_{args.use_chunked}_use_full_paper_as_corpus_{args.use_full_paper_as_corpus}_use_full_paper_as_query_{args.use_full_paper_as_query}/iteration_{args.iteration_num}"

                elif args.use_chunked == True:
                    if args.use_RRF_for_chunked_baseline == False:
                        result_folder_name = f"{last_two_dirs}/remove_citations_{args.remove_citations}_remove_mentions_{args.remove_mentions}/No_QueryOptimizers/use_query_optimizer_{args.use_query_optimizer}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.chunk_unit}_{args.use_chunked}_use_full_paper_as_corpus_{args.use_full_paper_as_corpus}_use_full_paper_as_query_{args.use_full_paper_as_query}/iteration_{args.iteration_num}"
                    
                    elif args.use_RRF_for_chunked_baseline == True:
                        result_folder_name = f"{last_two_dirs}/remove_citations_{args.remove_citations}_remove_mentions_{args.remove_mentions}/No_QueryOptimizers/use_query_optimizer_{args.use_query_optimizer}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.chunk_unit}_{args.use_chunked}_use_full_paper_as_corpus_{args.use_full_paper_as_corpus}_use_RRF_for_chunked_baseline_{args.use_RRF_for_chunked_baseline}_use_full_paper_as_query_{args.use_full_paper_as_query}/iteration_{args.iteration_num}"
                        
    elif args.iterative_retrieval == True:
        if args.use_query_optimizer == True:
            if args.embedding_fusion == False:
                if args.use_trained_model == False:
                    if args.SubTreeSearch == True:
                        if args.starting_iteration_of_A2A_only > 0:
                            if args.use_aspect_aware_cache_for_selection == True:
                                result_folder_name = f"{last_two_dirs}/remove_citations_{args.remove_citations}_remove_mentions_{args.remove_mentions}/Use_QueryOptimizers/iterative_retrieval_{args.iterative_retrieval}_total_iterations_{args.total_iterative_retrieval_loop}_SubTreeSearch_{args.SubTreeSearch}_Recursive_Merge_{args.Recursive_Merge}_selector_starting_idx_{args.selector_starting_idx}_starting_iteration_of_A2A_only_{args.starting_iteration_of_A2A_only}/use_trained_model_{args.use_trained_model}/{args.query_optimizer_model}/use_multi_source_{args.use_multi_source}/multi_agent_{args.multi_agent}_INCLUDE_ORIGINAL_RETRIEVAL_{args.include_original_retrieval}_BASE_{args.use_base_agent}_METHOD_{args.use_method_agent}_RESEARCH_QUESTION_{args.use_research_question_agent}_EXPERIMENT_{args.use_experiment_agent}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.chunk_unit}_{args.use_chunked}_use_full_paper_as_corpus_{args.use_full_paper_as_corpus}/iteration_{args.iteration_num}"
                            elif args.use_aspect_aware_cache_for_selection == False:
                                result_folder_name = f"{last_two_dirs}/remove_citations_{args.remove_citations}_remove_mentions_{args.remove_mentions}/Use_QueryOptimizers/iterative_retrieval_{args.iterative_retrieval}_total_iterations_{args.total_iterative_retrieval_loop}_SubTreeSearch_{args.SubTreeSearch}_Recursive_Merge_{args.Recursive_Merge}_selector_starting_idx_{args.selector_starting_idx}_starting_iteration_of_A2A_only_{args.starting_iteration_of_A2A_only}_use_aspect_aware_cache_for_selection_{args.use_aspect_aware_cache_for_selection}/use_trained_model_{args.use_trained_model}/{args.query_optimizer_model}/use_multi_source_{args.use_multi_source}/multi_agent_{args.multi_agent}_INCLUDE_ORIGINAL_RETRIEVAL_{args.include_original_retrieval}_BASE_{args.use_base_agent}_METHOD_{args.use_method_agent}_RESEARCH_QUESTION_{args.use_research_question_agent}_EXPERIMENT_{args.use_experiment_agent}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.chunk_unit}_{args.use_chunked}_use_full_paper_as_corpus_{args.use_full_paper_as_corpus}/iteration_{args.iteration_num}"
                        else:
                            if args.use_aspect_aware_cache_for_selection == True:
                                result_folder_name = f"{last_two_dirs}/remove_citations_{args.remove_citations}_remove_mentions_{args.remove_mentions}/Use_QueryOptimizers/iterative_retrieval_{args.iterative_retrieval}_total_iterations_{args.total_iterative_retrieval_loop}_SubTreeSearch_{args.SubTreeSearch}_Recursive_Merge_{args.Recursive_Merge}_selector_starting_idx_{args.selector_starting_idx}/use_trained_model_{args.use_trained_model}/{args.query_optimizer_model}/use_multi_source_{args.use_multi_source}/multi_agent_{args.multi_agent}_INCLUDE_ORIGINAL_RETRIEVAL_{args.include_original_retrieval}_BASE_{args.use_base_agent}_METHOD_{args.use_method_agent}_RESEARCH_QUESTION_{args.use_research_question_agent}_EXPERIMENT_{args.use_experiment_agent}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.chunk_unit}_{args.use_chunked}_use_full_paper_as_corpus_{args.use_full_paper_as_corpus}/iteration_{args.iteration_num}"
                            elif args.use_aspect_aware_cache_for_selection == False:
                                result_folder_name = f"{last_two_dirs}/remove_citations_{args.remove_citations}_remove_mentions_{args.remove_mentions}/Use_QueryOptimizers/iterative_retrieval_{args.iterative_retrieval}_total_iterations_{args.total_iterative_retrieval_loop}_SubTreeSearch_{args.SubTreeSearch}_Recursive_Merge_{args.Recursive_Merge}_selector_starting_idx_{args.selector_starting_idx}_use_aspect_aware_cache_for_selection_{args.use_aspect_aware_cache_for_selection}/use_trained_model_{args.use_trained_model}/{args.query_optimizer_model}/use_multi_source_{args.use_multi_source}/multi_agent_{args.multi_agent}_INCLUDE_ORIGINAL_RETRIEVAL_{args.include_original_retrieval}_BASE_{args.use_base_agent}_METHOD_{args.use_method_agent}_RESEARCH_QUESTION_{args.use_research_question_agent}_EXPERIMENT_{args.use_experiment_agent}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.chunk_unit}_{args.use_chunked}_use_full_paper_as_corpus_{args.use_full_paper_as_corpus}/iteration_{args.iteration_num}"
                                
                elif args.use_trained_model == True:
                    if args.SubTreeSearch == True:
                        try:
                            path = Path(args.method_agent_model_path)
                            train_dataset_name = path.parents[3].name
                        except:
                            pass

                        try:
                            path = Path(args.experiment_agent_model_path)
                            train_dataset_name = path.parents[3].name
                        except:
                            pass

                        try:
                            path = Path(args.research_question_agent_model_path)
                            train_dataset_name = path.parents[3].name
                        except:
                            pass
                        
                        
                        if args.starting_iteration_of_A2A_only > 0:
                            if args.use_aspect_aware_cache_for_selection == True:
                                result_folder_name = f"{last_two_dirs}/remove_citations_{args.remove_citations}_remove_mentions_{args.remove_mentions}/Use_QueryOptimizers/use_trained_model_{args.use_trained_model}/iterative_retrieval_{args.iterative_retrieval}_total_iterations_{args.total_iterative_retrieval_loop}_SubTreeSearch_{args.SubTreeSearch}_Recursive_Merge_{args.Recursive_Merge}_selector_starting_idx_{args.selector_starting_idx}_starting_iteration_of_A2A_only_{args.starting_iteration_of_A2A_only}/{train_dataset_name}/{args.query_optimizer_model}/use_multi_source_{args.use_multi_source}/multi_agent_{args.multi_agent}_INCLUDE_ORIGINAL_RETRIEVAL_{args.include_original_retrieval}_BASE_{args.use_base_agent}_METHOD_{args.use_method_agent}_RESEARCH_QUESTION_{args.use_research_question_agent}_EXPERIMENT_{args.use_experiment_agent}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.chunk_unit}_{args.use_chunked}_use_full_paper_as_corpus_{args.use_full_paper_as_corpus}/iteration_{args.iteration_num}"               
                            elif args.use_aspect_aware_cache_for_selection == False:
                                result_folder_name = f"{last_two_dirs}/remove_citations_{args.remove_citations}_remove_mentions_{args.remove_mentions}/Use_QueryOptimizers/use_trained_model_{args.use_trained_model}/iterative_retrieval_{args.iterative_retrieval}_total_iterations_{args.total_iterative_retrieval_loop}_SubTreeSearch_{args.SubTreeSearch}_Recursive_Merge_{args.Recursive_Merge}_selector_starting_idx_{args.selector_starting_idx}_starting_iteration_of_A2A_only_{args.starting_iteration_of_A2A_only}_use_aspect_aware_cache_for_selection_{args.use_aspect_aware_cache_for_selection}/{train_dataset_name}/{args.query_optimizer_model}/use_multi_source_{args.use_multi_source}/multi_agent_{args.multi_agent}_INCLUDE_ORIGINAL_RETRIEVAL_{args.include_original_retrieval}_BASE_{args.use_base_agent}_METHOD_{args.use_method_agent}_RESEARCH_QUESTION_{args.use_research_question_agent}_EXPERIMENT_{args.use_experiment_agent}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.chunk_unit}_{args.use_chunked}_use_full_paper_as_corpus_{args.use_full_paper_as_corpus}/iteration_{args.iteration_num}"               
                        else:
                            if args.use_aspect_aware_cache_for_selection == True:
                                result_folder_name = f"{last_two_dirs}/remove_citations_{args.remove_citations}_remove_mentions_{args.remove_mentions}/Use_QueryOptimizers/use_trained_model_{args.use_trained_model}/iterative_retrieval_{args.iterative_retrieval}_total_iterations_{args.total_iterative_retrieval_loop}_SubTreeSearch_{args.SubTreeSearch}_Recursive_Merge_{args.Recursive_Merge}_selector_starting_idx_{args.selector_starting_idx}/{train_dataset_name}/{args.query_optimizer_model}/use_multi_source_{args.use_multi_source}/multi_agent_{args.multi_agent}_INCLUDE_ORIGINAL_RETRIEVAL_{args.include_original_retrieval}_BASE_{args.use_base_agent}_METHOD_{args.use_method_agent}_RESEARCH_QUESTION_{args.use_research_question_agent}_EXPERIMENT_{args.use_experiment_agent}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.chunk_unit}_{args.use_chunked}_use_full_paper_as_corpus_{args.use_full_paper_as_corpus}/iteration_{args.iteration_num}"
                            elif args.use_aspect_aware_cache_for_selection == False:
                                result_folder_name = f"{last_two_dirs}/remove_citations_{args.remove_citations}_remove_mentions_{args.remove_mentions}/Use_QueryOptimizers/use_trained_model_{args.use_trained_model}/iterative_retrieval_{args.iterative_retrieval}_total_iterations_{args.total_iterative_retrieval_loop}_SubTreeSearch_{args.SubTreeSearch}_Recursive_Merge_{args.Recursive_Merge}_selector_starting_idx_{args.selector_starting_idx}_use_aspect_aware_cache_for_selection_{args.use_aspect_aware_cache_for_selection}/{train_dataset_name}/{args.query_optimizer_model}/use_multi_source_{args.use_multi_source}/multi_agent_{args.multi_agent}_INCLUDE_ORIGINAL_RETRIEVAL_{args.include_original_retrieval}_BASE_{args.use_base_agent}_METHOD_{args.use_method_agent}_RESEARCH_QUESTION_{args.use_research_question_agent}_EXPERIMENT_{args.use_experiment_agent}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.chunk_unit}_{args.use_chunked}_use_full_paper_as_corpus_{args.use_full_paper_as_corpus}/iteration_{args.iteration_num}"
                                 
        elif args.use_query_optimizer == False:
            ## only for cases without abstract chaining
            if args.use_chunked == False:
                if args.use_full_paper_as_query == False:
                    result_folder_name = f"{last_two_dirs}/remove_citations_{args.remove_citations}_remove_mentions_{args.remove_mentions}/No_QueryOptimizers/use_query_optimizer_{args.use_query_optimizer}/iterative_retrieval_{args.iterative_retrieval}_total_iterations_{args.total_iterative_retrieval_loop}_selector_starting_idx_{args.selector_starting_idx}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.chunk_unit}_{args.use_chunked}_use_full_paper_as_corpus_{args.use_full_paper_as_corpus}/iteration_{args.iteration_num}"
            
                elif args.use_full_paper_as_query == True:
                    result_folder_name = f"{last_two_dirs}/remove_citations_{args.remove_citations}_remove_mentions_{args.remove_mentions}/No_QueryOptimizers/use_query_optimizer_{args.use_query_optimizer}/iterative_retrieval_{args.iterative_retrieval}_total_iterations_{args.total_iterative_retrieval_loop}_selector_starting_idx_{args.selector_starting_idx}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.chunk_unit}_{args.use_chunked}_use_full_paper_as_corpus_{args.use_full_paper_as_corpus}_use_full_paper_as_query_{args.use_full_paper_as_query}/iteration_{args.iteration_num}"
            
            elif args.use_chunked == True:
                if args.use_RRF_for_chunked_baseline == False:
                    if args.use_full_paper_as_query == False:
                        result_folder_name = f"{last_two_dirs}/remove_citations_{args.remove_citations}_remove_mentions_{args.remove_mentions}/No_QueryOptimizers/use_query_optimizer_{args.use_query_optimizer}/iterative_retrieval_{args.iterative_retrieval}_total_iterations_{args.total_iterative_retrieval_loop}_selector_starting_idx_{args.selector_starting_idx}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.chunk_unit}_{args.use_chunked}_use_full_paper_as_corpus_{args.use_full_paper_as_corpus}/iteration_{args.iteration_num}"
                    elif args.use_full_paper_as_query == True:
                        result_folder_name = f"{last_two_dirs}/remove_citations_{args.remove_citations}_remove_mentions_{args.remove_mentions}/No_QueryOptimizers/use_query_optimizer_{args.use_query_optimizer}/iterative_retrieval_{args.iterative_retrieval}_total_iterations_{args.total_iterative_retrieval_loop}_selector_starting_idx_{args.selector_starting_idx}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.chunk_unit}_{args.use_chunked}_use_full_paper_as_corpus_{args.use_full_paper_as_corpus}_use_full_paper_as_query_{args.use_full_paper_as_query}/iteration_{args.iteration_num}"
                elif args.use_RRF_for_chunked_baseline == True:
                    if args.use_full_paper_as_query == False:
                        result_folder_name = f"{last_two_dirs}/remove_citations_{args.remove_citations}_remove_mentions_{args.remove_mentions}/No_QueryOptimizers/use_query_optimizer_{args.use_query_optimizer}/iterative_retrieval_{args.iterative_retrieval}_total_iterations_{args.total_iterative_retrieval_loop}_selector_starting_idx_{args.selector_starting_idx}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.chunk_unit}_{args.use_chunked}_use_full_paper_as_corpus_{args.use_full_paper_as_corpus}_use_RRF_for_chunked_baseline_{args.use_RRF_for_chunked_baseline}/iteration_{args.iteration_num}"
                    elif args.use_full_paper_as_query == True:
                        result_folder_name = f"{last_two_dirs}/remove_citations_{args.remove_citations}_remove_mentions_{args.remove_mentions}/No_QueryOptimizers/use_query_optimizer_{args.use_query_optimizer}/iterative_retrieval_{args.iterative_retrieval}_total_iterations_{args.total_iterative_retrieval_loop}_selector_starting_idx_{args.selector_starting_idx}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.chunk_unit}_{args.use_chunked}_use_full_paper_as_corpus_{args.use_full_paper_as_corpus}_use_full_paper_as_query_{args.use_full_paper_as_query}_use_RRF_for_chunked_baseline_{args.use_RRF_for_chunked_baseline}/iteration_{args.iteration_num}"
            
    result_folder_path = os.path.join(result_directory, result_folder_name)
    os.makedirs(result_folder_path, exist_ok=True)
    
    save_config(args, result_folder_path)
    

    test_file_directory = []
    # Walk through the directory tree
    for root, _, files in os.walk(args.benchmark_directory):
        for file in files:
            if file.endswith('.json'):
                target_result_file = os.path.join(result_folder_path, file)
                test_file_directory.append(os.path.join(target_result_file))
    
    current_test_file_directory = test_file_directory[args.start_idx:args.end_idx]
    filtered_test_file_directory = []            
    for target_result_file in current_test_file_directory:       
        if not os.path.exists(target_result_file):
            filename = os.path.basename(target_result_file)
            filtered_test_file_directory.append(os.path.join(args.benchmark_directory, filename))
        
    test_file_directory = filtered_test_file_directory
    logging.info(f"Number of files left to run is {len(test_file_directory)}")
    ## Initialize Retrieval Module
    retriever = Retriever(args, result_folder_path)
    retriever.initialize()
    retriever.evaluate(args, test_file_directory, result_folder_path)

# Example usage
if __name__ == "__main__":
    # Abstract to compare
    dataset_config = DatasetConfig()
    query_optimizer_config = QueryOptimizerConfig()
    retrieval_config = RetrievalConfig()
    IterativeRetrieval_config = IterativeRetrievalConfig()
    
    args = merge_configs(dataset_config, retrieval_config, query_optimizer_config, IterativeRetrieval_config)
    evaluate(args)
