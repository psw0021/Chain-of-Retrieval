import argparse
import os
import sys
import json
from pathlib import Path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
# Add the root directory to Python's module search path
sys.path.append(root_dir)
from Utils.utils import *
from Retrieval.retriever import Retriever
import logging
import re


def config():
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    
    parser.add_argument("--train_set_directory", default="")
    parser.add_argument("--query_optimizer_model", default="")
    parser.add_argument("--deploy_llm", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--use_gpt", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--embedding_model", default="jina-embeddings-v2-base-en", choices=["jina-embeddings-v2-base-en", "bge-m3", "inf-retriever-v1-1.5b"])
    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--end_idx", default=500, type=int)
    parser.add_argument("--top_k", default=100, type=int)
    parser.add_argument("--query_detailedness", default=3, type=int)
    parser.add_argument("--max_top_k", default=100, type=int)
    parser.add_argument("--include_original_retrieval", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--use_method_agent", default=True, type=lambda x: x.lower() == "true")
    parser.add_argument("--use_experiment_agent", default=True, type=lambda x: x.lower() == "true")
    parser.add_argument("--use_research_question_agent", default=True, type=lambda x: x.lower() == "true")
    parser.add_argument("--corpus_directory", default="")
    parser.add_argument("--use_multi_source", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--use_chunked", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--batch_size", default=500, type=int)
    parser.add_argument("--rollout_number", default=5, type=int)
    parser.add_argument("--gpu_memory_utilization", default=0.6, type=float)
    parser.add_argument("--repetition_penalty", default=1.2, type=float)
    parser.add_argument("--max_tokens", default=2000, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--top_p", default=0.8, type=float)
    parser.add_argument("--absolute_path", default="/c2/swpark/Chain-of-Retrieval", help="Absolute path to the parent directory of Paper2PaperRetrievalBench(SciFullBench + PatentFullBench)")

    args = parser.parse_args()
    
    return args


def save_config(config, result_path):
    """Save the configuration dictionary as JSON."""
    config_to_save = vars(config)
    config_path = os.path.join(result_path, "config.json")

    with open(config_path, "w") as f:
        json.dump(config_to_save, f, indent=4)

    print(f"Config saved at {config_path}")


def traverse_folder(folder_path):
    """
    Traverse through the folder and retrieve file names.

    Args:
        folder_path (str): Path to the folder to traverse.

    Returns:
        list: A list of file paths.
    """
    file_list = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_list.append(file)
    return file_list


def traverse_folder_w_full_directory(folder_path):
    """
    Traverse through the folder and retrieve file names.

    Args:
        folder_path (str): Path to the folder to traverse.

    Returns:
        list: A list of file paths.
    """
    file_list = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def format_paper_content(content):
    paper_content = content.replace("\n", " ")
    split_paper_content = paper_content.split(" ")
        
    parsed_paper_content_list = []
    for word in split_paper_content:
        word = word.replace(" ", "")
        if word != "":
            parsed_paper_content_list.append(word)
            
    parsed_paper_content = " ".join(parsed_paper_content_list)
    parsed_paper_content = parsed_paper_content.replace("-", " ")

    return parsed_paper_content.lower()


def filter_unnecessary_query(args, train_file_directories: list) -> list:    
    ## Format our test set(benchmark input paper information)
    final_benchmark_root_directory = f"{args.absolute_path}/Paper2PaperRetrievalBench/SciFullBench/Final_Dataset_w_citations_mentions_removed/Benchmark"
    venues = ["ACL", "EMNLP", "ICLR", "NeurIPS"]
    relations = ["Cited_Papers", "Direct_References"]
    
    benchmark_existing_dictionary = {}
    for venue in venues:
        for relation in relations:
            benchmark_path = os.path.join(final_benchmark_root_directory, f"{venue}/{relation}")
            benchmark_file_list = traverse_folder_w_full_directory(benchmark_path)
            
            for file in benchmark_file_list:
                with open(file, "r") as json_file:
                    content = json.load(json_file)
                    
                query_content = content["Query"]
                
                query_title = query_content["title"]
                query_abstract = query_content['abstract']
                
                if query_title != None:
                    try:
                        benchmark_existing_dictionary[format_paper_content(query_title)]
                    except KeyError:
                        benchmark_existing_dictionary[format_paper_content(query_title)] = True
    
    final_corpus_directory = f"{args.absolute_path}/Paper2PaperRetrievalBench/SciFullBench/Final_Dataset_w_citations_mentions_removed/Target_Corpus/target_corpus_citations_removed_True_mentions_removed_True/corpus.json"
    with open(final_corpus_directory, "r") as json_file:
        final_corpus = json.load(json_file)

    corpus_existing_dictionary = {}
    for corpus in final_corpus:
            corpus_title = corpus["title"]
            if corpus_title != None:
                try:
                    corpus_existing_dictionary[format_paper_content(corpus_title)]
                except KeyError:
                    corpus_existing_dictionary[format_paper_content(corpus_title)] = True
    
    filtered_train_file_directories = []                
    for file in train_file_directories:
        with open(file, "r") as json_file:
            train_data = json.load(json_file)
            
        train_data_title = train_data["Query"]["title"]
        
        try:
            benchmark_existing_dictionary[format_paper_content(train_data_title)]
            continue
        except:
            pass

        try:
            corpus_existing_dictionary[format_paper_content(train_data_title)]
            continue
        except:
            pass

        filtered_train_file_directories.append(file)
        
    
    return filtered_train_file_directories


def filter_only_reference(train_file_directories: list) -> list:
    filtered_train_file_directories = []
    for file in train_file_directories:
            with open(file, "r") as json_file:
                evaluation_data = json.load(json_file)
            
            candidate_papers = evaluation_data["Total_Candidate"]
            reference_plus_cited = []
            for candidate in candidate_papers:
                candidate_title = candidate["title"]
                candidate_abstract = candidate["abstract"]
            
                formatted_candidate = f"Title: {candidate_title}\nAbstract: {candidate_abstract}"

                reference_plus_cited.append(formatted_candidate)

            basename = os.path.basename(file)
            citation_stored_info_root_directory = "Raw_Train_Dataset/semantic_scholar_citation_information"
            citation_stored_info_target_directory = os.path.join(citation_stored_info_root_directory, basename)
            try:
                    with open(citation_stored_info_target_directory, "r") as json_file:
                        cited_content = json.load(json_file)

                    semantic_scholar_cited_list = cited_content["Semantic Scholar Cited Papers"]

                    semantic_scholar_cited_formatted_title_list = []
                    for citation in semantic_scholar_cited_list:
                        raw_citation_title = citation["title"]

                        formatted_paper_title = format_paper_content(raw_citation_title)

                        semantic_scholar_cited_formatted_title_list.append(formatted_paper_title)

                    correct_candidates_citation_split = []
                    for gt in reference_plus_cited:
                        match = re.search(r"Title:\s*(.*?)\s*Abstract:", gt, re.DOTALL)
                        if match:
                            title_content = match.group(1).replace('\n', ' ').strip()
                        formatted_gt_title = format_paper_content(title_content)
                        if formatted_gt_title in semantic_scholar_cited_formatted_title_list:
                            correct_candidates_citation_split.append(gt)

                    correct_candidates_reference_split = []
                    for gt in reference_plus_cited:
                        if gt not in correct_candidates_citation_split:
                            correct_candidates_reference_split.append(gt)

                    correct_candidates = correct_candidates_reference_split

                    if len(correct_candidates) < 5:
                        continue
                    
                    filtered_train_file_directories.append(file)
                    
            except:
                pass
    
    return filtered_train_file_directories
    
def evaluate(args):
    ## Bencharmk directory
    path = Path(args.train_set_directory)
    last_two_dirs = "/".join(path.parts[-3:])
    result_directory = "Train_Dataset/Rollout_Results_Final"
    result_folder_name = f"{last_two_dirs}/{args.query_optimizer_model}/use_multi_source_{args.use_multi_source}/INCLUDE_ORIGINAl_RETRIEVAL_{args.include_original_retrieval}_METHOD_{args.use_method_agent}_RESEARCH_QUESTION_{args.use_research_question_agent}_EXPERIMENT_{args.use_experiment_agent}_{args.query_detailedness}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.use_chunked}"

    result_folder_path = os.path.join(result_directory, result_folder_name)
    
    finished_results = traverse_folder(result_folder_path)
    os.makedirs(result_folder_path, exist_ok=True)
    
    test_file_directory = []
    # Walk through the directory tree
    for root, _, files in os.walk(args.train_set_directory):
        for file in files:
            if file.endswith('.json'):
                test_file_directory.append(os.path.join(root, file))
    logging.info(f"Number of original files to rollout is {len(test_file_directory)}")
    
    save_config(args, result_folder_path)
    test_file_directory = filter_unnecessary_query(args, test_file_directory)
    logging.info(f"Number of files to rollout after filtering ones existing in our benchmark and corpus is {len(test_file_directory)}")
    
    test_file_directory = filter_only_reference(test_file_directory)
    logging.info(f"Number of files to rollout after filtering queries with reference candidates more than equal to 5 {len(test_file_directory)}")
    
    test_file_directory = test_file_directory[args.start_idx:args.end_idx]
    logging.info(f"Number of files to rollout per thread is {len(test_file_directory)}")
    
    
    
    unfinished_test_file_directory = []
    for file in test_file_directory:
        filename = os.path.basename(file)
        if filename not in finished_results:
            unfinished_test_file_directory.append(file)
    
    print(finished_results)
    test_file_directory = unfinished_test_file_directory
    
    logging.info(f"Number of unfinished files is {len(unfinished_test_file_directory)}")
    
    ## Initialize Retrieval Module
    retriever = Retriever(args, result_folder_path)
    retriever.initialize()
    retriever.evaluate(args, test_file_directory, result_folder_path)

if __name__ == "__main__":
    args = config()
    evaluate(args)
