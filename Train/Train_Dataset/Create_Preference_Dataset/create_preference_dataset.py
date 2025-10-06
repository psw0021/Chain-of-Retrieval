import os
import json
import yaml
from datasets import DatasetDict, load_dataset, Dataset
from huggingface_hub import login
import argparse
import sys
import random
import re

def config():
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    
    parser.add_argument("--use_group_reward", default=False)
    parser.add_argument("--use_recall_as_reward", default=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", choices=["meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct"])
    parser.add_argument("--original_retrieval", default=False)
    parser.add_argument("--multi_source", default=False)
    parser.add_argument("--retriever", default="inf-retriever-v1-1.5b", choices=["jina-embeddings-v2-base-en", "bge-m3", "inf-retriever-v1-1.5b"])
    parser.add_argument("--use_overlap_ratio_as_reward", default=False)
    parser.add_argument("--top_k", default=30)
    parser.add_argument("--weight_alpha", default=0.6, type=int)
    parser.add_argument("--overlap_threshold", default=0.35)
    parser.add_argument("--margin_threshold", default=0.03)
    parser.add_argument("--huggingface_directory", default="")
    parser.add_argument("--training_algorithm", default="DPO", choices=["DPO", "ORPO"])
    parser.add_argument("--use_reference_as_gt", default=True, type=bool)
    parser.add_argument("--dataset_for_iterative_retrieval", default=True, type=bool)
    parser.add_argument("--upload_to_huggingface", default=True)
    parser.add_argument("--absolute_path", default="", help="absolute path for parent directory of dataset")

    args = parser.parse_args()
    
    return args

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
            file_list.append(os.path.join(root, file))
    return file_list


def recall_at_k(retrieved, relevant, k):
    retrieved_at_k = retrieved[:k]
    relevant_at_k = set(relevant).intersection(set(retrieved_at_k))
    
    return len(relevant_at_k) / len(relevant) if relevant else 0

def Compute_Reciprocal_Rank_Fusion(agent1_retrieved_result, agent2_retrieved_result, agent3_retrieved_result, original_retrieved_result, top_k):
    agent1_retrieved_result = agent1_retrieved_result[:top_k]
    agent2_retrieved_result = agent2_retrieved_result[:top_k]
    agent3_retrieved_result = agent3_retrieved_result[:top_k]
    original_retrieved_result = original_retrieved_result[:top_k]
    retrieved_results_from_agents = [agent1_retrieved_result, agent2_retrieved_result, agent3_retrieved_result, original_retrieved_result]
    fused_scores = {}
    for top_k_corpus in retrieved_results_from_agents:
        for rank, corpus in enumerate(top_k_corpus):
            fused_scores[corpus] = fused_scores.get(corpus, 0) + 1 / (60 + rank + 1)
                            
    sorted_docs = [doc for doc, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
    
    return sorted_docs


def Average_Recall_at_K_Group_Reward(retrieved, relevant, other_agent1_retrieved_results, other_agent2_retrieved_results, original_retrieved_result, top_k):
    fused_results_with_RRF = []
    for retrieved_result_from_agent1 in other_agent1_retrieved_results:
        for retrieved_result_from_agent2 in other_agent2_retrieved_results:
            rrf_sorted_docs = Compute_Reciprocal_Rank_Fusion(retrieved, retrieved_result_from_agent1, retrieved_result_from_agent2, original_retrieved_result, top_k)
            fused_results_with_RRF.append(rrf_sorted_docs)
            
    total_recall_at_k = 0
    for fused_retrieved_result in fused_results_with_RRF:
        total_recall_at_k = total_recall_at_k + recall_at_k(fused_retrieved_result, relevant, top_k)
        
    return total_recall_at_k / len(fused_results_with_RRF)
            

def average_overlap_ratio_at_k(retrieved_at_k, other_agent_retrieved_results, k):
    total_overlap_ratio_at_k = 0
    number_of_total_comparison = 0
    for other_agent_retrieved_result in other_agent_retrieved_results:
        other_agent_retrieved_at_k = other_agent_retrieved_result[:k]
        overlap_samples_at_k = set(other_agent_retrieved_at_k).intersection(set(retrieved_at_k))
        overlap_ratio_at_k = len(overlap_samples_at_k) / k
        total_overlap_ratio_at_k = total_overlap_ratio_at_k + overlap_ratio_at_k
        
        number_of_total_comparison = number_of_total_comparison + 1
        
    average_overlap_ratio_at_k = total_overlap_ratio_at_k / number_of_total_comparison
    
    return average_overlap_ratio_at_k


def Rank_Queries(retrieved_results, other_agent_retrieved_results, original_retrieved_result, correct_candidates, use_recall_as_reward, use_overlap_ratio_as_reward, top_k, alpha, overlap_threshold):
    queries = list(retrieved_results.keys())
    if use_recall_as_reward == True and use_overlap_ratio_as_reward == False:
        query_and_rewards = {}
        for query in queries:
            retrieved_result_for_each_query = retrieved_results[query]
            sorted_contents = [v["Content"] for k, v in sorted(retrieved_result_for_each_query.items(), key=lambda item: item[1]["Score"])]

            top_k_sorted_contents = sorted_contents[:top_k]

            if args.use_group_reward == False:
                reward = recall_at_k(top_k_sorted_contents, correct_candidates, top_k)
                
            elif args.use_group_reward == True:
                reward = Average_Recall_at_K_Group_Reward(top_k_sorted_contents, correct_candidates, other_agent_retrieved_results[0], other_agent_retrieved_results[1], original_retrieved_result, top_k)

            query_and_rewards[query] = reward
        
        best_query = max(query_and_rewards, key=query_and_rewards.get)
        worst_query = min(query_and_rewards, key=query_and_rewards.get)
        
        reward_for_best_query = query_and_rewards[best_query]
        reward_for_worst_query = query_and_rewards[worst_query]
        
        margin_of_reward = reward_for_best_query - reward_for_worst_query
    
    elif use_recall_as_reward == False and use_overlap_ratio_as_reward == True:
        print(f"Don't use recall for reward")
        print(f"Use 1 - overlap ratio between agent retrieved results as reward")
        query_and_rewards = {}
        for query in queries:
            retrieved_result_for_each_query = retrieved_results[query]
            sorted_contents = [v["Content"] for k, v in sorted(retrieved_result_for_each_query.items(), key=lambda item: item[1]["Score"])]
            
            top_k_sorted_contents = sorted_contents[:top_k]
            
            total_other_agent_retrieved_results = []
            for other_agent_retrieved_result in other_agent_retrieved_results[0]:
                total_other_agent_retrieved_results.append(other_agent_retrieved_result)
            for other_agent_retrieved_result in other_agent_retrieved_results[1]:
                total_other_agent_retrieved_results.append(other_agent_retrieved_result)
            
            average_overlap_ratio_k = average_overlap_ratio_at_k(top_k_sorted_contents, total_other_agent_retrieved_results, top_k)
            
            reward = (1 - average_overlap_ratio_k)
            
            print(args.overlap_threshold)
            if average_overlap_ratio_k > args.overlap_threshold:
                query_and_rewards[query] = reward
                
        
        query_list = query_and_rewards.keys()
        if len(query_list) > 2:
            best_query = max(query_and_rewards, key=query_and_rewards.get)
            worst_query = min(query_and_rewards, key=query_and_rewards.get)
        
            reward_for_best_query = query_and_rewards[best_query]
            reward_for_worst_query = query_and_rewards[worst_query]
        
            margin_of_reward = reward_for_best_query - reward_for_worst_query
        
        else:
            best_query = ""
            worst_query = ""
            
            reward_for_best_query = 0
            reward_for_worst_query = 0
            
            margin_of_reward = 0
        
    elif use_recall_as_reward == True and use_overlap_ratio_as_reward == True:
        query_and_rewards = {}
        for query in queries:
            retrieved_result_for_each_query = retrieved_results[query]
            sorted_contents = [v["Content"] for k, v in sorted(retrieved_result_for_each_query.items(), key=lambda item: item[1]["Score"])]
            
            top_k_sorted_contents = sorted_contents[:top_k]
            recall_k = recall_at_k(top_k_sorted_contents, correct_candidates, top_k)
            
            total_other_agent_retrieved_results = []
            for other_agent_retrieved_result in other_agent_retrieved_results[0]:
                total_other_agent_retrieved_results.append(other_agent_retrieved_result)
            for other_agent_retrieved_result in other_agent_retrieved_results[1]:
                total_other_agent_retrieved_results.append(other_agent_retrieved_result)
                
            average_overlap_ratio_k = average_overlap_ratio_at_k(top_k_sorted_contents, total_other_agent_retrieved_results, top_k)
            
            reward = (1 - alpha) * (1 - average_overlap_ratio_k) + (recall_k * alpha)
            
            if average_overlap_ratio_k > args.overlap_threshold:
                query_and_rewards[query] = reward
        
        query_list = query_and_rewards.keys()
        if len(query_list) > 2:
            best_query = max(query_and_rewards, key=query_and_rewards.get)
            worst_query = min(query_and_rewards, key=query_and_rewards.get)
        
            reward_for_best_query = query_and_rewards[best_query]
            reward_for_worst_query = query_and_rewards[worst_query]
        
            margin_of_reward = reward_for_best_query - reward_for_worst_query
        
        else:
            best_query = ""
            worst_query = ""
            
            reward_for_best_query = 0
            reward_for_worst_query = 0
            
            margin_of_reward = 0
    
              
    return best_query, worst_query, margin_of_reward, query_and_rewards


def collect_other_agent_retrieved_result(agent1_retrieved_result, agent2_retrieved_result):
    # combine experiment based agent, research question based agent results into one.
    agent1_query_list = list(agent1_retrieved_result.keys())
    agent2_query_list = list(agent2_retrieved_result.keys())
            
    other_agent1_retrieved_result = []
    for query in agent1_query_list:
        agent1_retrieved_results = agent1_retrieved_result[query]
        agent1_retrieved_results_list = [v["Content"] for k, v in sorted(agent1_retrieved_results.items(), key=lambda item: item[1]["Score"])]
        other_agent1_retrieved_result.append(agent1_retrieved_results_list)
    
    other_agent2_retrieved_result = []
    for query in agent2_query_list:
        agent2_retrieved_results = agent2_retrieved_result[query]
        agent2_retrieved_results_list = [v["Content"] for k, v in sorted(agent2_retrieved_results.items(), key=lambda item: item[1]["Score"])]
        other_agent2_retrieved_result.append(agent2_retrieved_results_list)
        
    return [other_agent1_retrieved_result, other_agent2_retrieved_result]


def format_dataset(full_paper_path, agent_name, query, training_algorithm):
    with open(full_paper_path, "r") as file:
        full_paper_content = file.read()

    if agent_name == "method":
        prompt_path = "Train_Dataset/QueryOptimizer/Prompts/method_focused_query_optimizer_agent_prompt_3.yaml"
        with open(prompt_path, "r") as file:
            data = yaml.safe_load(file)
                
        user_prompt = data["user_prompt"]
        system_prompt = data["system_prompt"]
            
        user_prompt = user_prompt.format(paper=full_paper_content)

        if training_algorithm == "DPO":
            formatted_dataset = f"Human: {system_prompt}\n\n{user_prompt}\nAssistant: {query}"
        elif training_algorithm == "ORPO":
            formatted_dataset = f"{system_prompt}\n\n{user_prompt}"
    
    elif agent_name == "experiment":
        prompt_path = "Train_Dataset/QueryOptimizer/Prompts/experiment_focused_query_optimizer_agent_prompt_3.yaml"
        with open(prompt_path, "r") as file:
            data = yaml.safe_load(file)
                
        user_prompt = data["user_prompt"]
        system_prompt = data["system_prompt"]
            
        user_prompt = user_prompt.format(paper=full_paper_content)

        if training_algorithm == "DPO":
            formatted_dataset = f"Human: {system_prompt}\n\n{user_prompt}\nAssistant: {query}"
        elif training_algorithm == "ORPO":
            formatted_dataset = f"{system_prompt}\n\n{user_prompt}"

    elif agent_name == "research_question":
        prompt_path = "Train_Dataset/QueryOptimizer/Prompts/research_question_focused_query_optimizer_agent_prompt_3.yaml"
        with open(prompt_path, "r") as file:
            data = yaml.safe_load(file)
                
        user_prompt = data["user_prompt"]
        system_prompt = data["system_prompt"]
            
        user_prompt = user_prompt.format(paper=full_paper_content)

        if training_algorithm == "DPO":
            formatted_dataset = f"Human: {system_prompt}\n\n{user_prompt}\nAssistant: {query}"
        elif training_algorithm == "ORPO":
            formatted_dataset = f"{system_prompt}\n\n{user_prompt}"

    
    return formatted_dataset


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
    

def main(args):
    ## Format our test set(benchmark input paper information)
    final_benchmark_root_directory = f"{args.absolute_path}/LongDocumentRetrieval/LongDocumentBench/ScientificPapers/Final_Dataset_w_citations_mentions_removed/Benchmark"
    venues = ["ACL", "EMNLP", "ICLR", "NeurIPS"]
    relations = ["Cited_Papers", "Direct_References"]
    
    benchmark_existing_dictionary = {}
    for venue in venues:
        for relation in relations:
            benchmark_path = os.path.join(final_benchmark_root_directory, f"{venue}/{relation}")
            benchmark_file_list = traverse_folder(benchmark_path)
            
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
    
    if args.dataset_for_iterative_retrieval == True:
        final_corpus_directory = f"{args.absolute_path}/LongDocumentRetrieval/LongDocumentBench/ScientificPapers/Final_Dataset_w_citations_mentions_removed/Target_Corpus/target_corpus_citations_removed_True_mentions_removed_True/corpus.json"
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
                    
    benchmark_existing_list = list(benchmark_existing_dictionary.keys())
    print(f"The number of existing papers without duplicates in our benchmark are {len(benchmark_existing_list)}")

    folder_path = f"Train_Dataset/Rollout_Results_Final/Train_Dataset/Final_Train_Set/Papers_and_Candidates/{args.model}/use_multi_source_{args.multi_source}/INCLUDE_ORIGINAl_RETRIEVAL_{args.original_retrieval}_METHOD_True_RESEARCH_QUESTION_True_EXPERIMENT_True_3/{args.retriever}_top_300_use_chunked_True"
    print(f"Current preference creation folder is is {folder_path}")
    roll_out_results = traverse_folder(folder_path)

    method_based_agent_data = []
    experiment_based_agent_data = []
    research_question_based_agent_data = []
    
    method_query_and_rewards_list = []
    experiment_query_and_rewards_list = []
    research_question_query_and_rewards_list = []

    best_worst_query_pair = {"method": [], "experiment": [], "research_question": []}

    original_train_set_root_directory = "Train_Dataset/Final_Train_Set/Papers_and_Candidates"

    for result in roll_out_results:
            try:
                with open(result, "r") as json_file:
                    roll_out_result = json.load(json_file)
            except:
                continue
        
            basename = os.path.basename(result)

            if basename == "prompts.json":
                continue

            if basename == "config.json":
                continue

            original_train_set_directory = os.path.join(original_train_set_root_directory, basename)
            with open(original_train_set_directory, "r") as json_file:
                train_data = json.load(json_file)

            train_data_title = train_data["Query"]["title"]

            try:
                benchmark_existing_dictionary[format_paper_content(train_data_title)]
                continue
            except:
                pass

            if args.dataset_for_iterative_retrieval == True:
                try:
                    corpus_existing_dictionary[format_paper_content(train_data_title)]
                    continue
                except:
                    pass

            paper_id = basename.removesuffix(".json")

            full_paper_root_directory = "Train_Dataset/Final_Train_Set/Full_Paper"

            full_paper_directory = os.path.join(full_paper_root_directory, f"{paper_id}.mmd")
            
            try:
                query_and_retrieved_candidates = roll_out_result["Retrieved_Candidates"]
            except KeyError:
                continue
            
            if args.use_reference_as_gt == False:
                reference_plus_cited = roll_out_result["Correct_Candidates"]

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

                    correct_candidates = correct_candidates_citation_split

                    if len(correct_candidates) < 5:
                        continue
                except:
                    pass
                
            elif args.use_reference_as_gt == True:
                reference_plus_cited = roll_out_result["Correct_Candidates"]

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
                except:
                    pass
                    
            if args.original_retrieval == True:    
                original_retrieval_result_dictionary = query_and_retrieved_candidates["original_retrieval"]
                original_abstract = list(original_retrieval_result_dictionary.keys())[0]
                original_retrieved_results_dict = original_retrieval_result_dictionary[original_abstract]
                original_retrieval_result = [v["Content"] for k, v in sorted(original_retrieved_results_dict.items(), key=lambda item: item[1]["Score"])]
            else:
                original_retrieval_result = ""
                
            method_based_agent_results = query_and_retrieved_candidates["METHOD FOCUSED AGENT"]
            experiment_based_agent_results = query_and_retrieved_candidates["EXPERIMENT Focused AGENT"]
            research_question_based_agent_results = query_and_retrieved_candidates["RESEARCH QUESTION FOCUSED AGENT"]

            #### Rank queries for method based agent results.
            other_agent_retrieved_result = collect_other_agent_retrieved_result(experiment_based_agent_results, research_question_based_agent_results)
                            
            best_query_for_method_based_agent, worst_query_for_method_based_agent, method_margin_of_reward, method_query_and_rewards = Rank_Queries(method_based_agent_results, other_agent_retrieved_result, original_retrieval_result, correct_candidates, args.use_recall_as_reward, args.use_overlap_ratio_as_reward, args.top_k, args.weight_alpha, args.overlap_threshold)
            
            if args.training_algorithm == "DPO":
                formatted_data_for_method_based_agent_chosen = format_dataset(full_paper_directory, "method", best_query_for_method_based_agent, args.training_algorithm)
                formatted_data_for_method_based_agent_rejected = format_dataset(full_paper_directory, "method", worst_query_for_method_based_agent, args.training_algorithm)

                chosen_and_rejected_for_method_based_agent = {"chosen": formatted_data_for_method_based_agent_chosen, "rejected": formatted_data_for_method_based_agent_rejected}
            
            elif args.training_algorithm == "ORPO":
                input_prompt = format_dataset(full_paper_directory, "method", best_query_for_method_based_agent, args.training_algorithm)

                chosen_and_rejected_for_method_based_agent = {"prompt":input_prompt, "chosen": best_query_for_method_based_agent, "rejected": worst_query_for_method_based_agent}
            
            current_index_for_method = len(best_worst_query_pair["method"])
            if method_margin_of_reward > args.margin_threshold:
                if best_query_for_method_based_agent != worst_query_for_method_based_agent:
                    method_based_agent_data.append(chosen_and_rejected_for_method_based_agent)
                    best_worst_query_pair["method"].append({"id": paper_id, "index": current_index_for_method, "Queries": {"BEST": best_query_for_method_based_agent, "WORST": worst_query_for_method_based_agent}, "REWARD MARGIN": method_margin_of_reward})
                    method_query_and_rewards_list.append(method_query_and_rewards)
            
            #### Rank queries for experiment based agent results.
            other_agent_retrieved_result = collect_other_agent_retrieved_result(method_based_agent_results, research_question_based_agent_results)
            
            best_query_for_experiment_based_agent, worst_query_for_experiment_based_agent, experiment_margin_of_reward, experiment_query_and_rewards = Rank_Queries(experiment_based_agent_results, other_agent_retrieved_result, original_retrieval_result, correct_candidates, args.use_recall_as_reward, args.use_overlap_ratio_as_reward, args.top_k, args.weight_alpha, args.overlap_threshold)
            
            if args.training_algorithm == "DPO":
                formatted_data_for_experiment_based_agent_chosen = format_dataset(full_paper_directory, "experiment", best_query_for_experiment_based_agent, args.training_algorithm)
                formatted_data_for_experiment_based_agent_rejected = format_dataset(full_paper_directory, "experiment", worst_query_for_experiment_based_agent, args.training_algorithm)

                chosen_and_rejected_for_experiment_based_agent = {"chosen": formatted_data_for_experiment_based_agent_chosen, "rejected": formatted_data_for_experiment_based_agent_rejected}
            
            elif args.training_algorithm == "ORPO":
                input_prompt = format_dataset(full_paper_directory, "experiment", best_query_for_experiment_based_agent, args.training_algorithm)

                chosen_and_rejected_for_experiment_based_agent = {"prompt":input_prompt, "chosen": best_query_for_experiment_based_agent, "rejected": worst_query_for_experiment_based_agent}

            current_index_for_experiment = len(best_worst_query_pair["experiment"])
            if experiment_margin_of_reward > args.margin_threshold:
                if best_query_for_experiment_based_agent != worst_query_for_experiment_based_agent:
                    experiment_based_agent_data.append(chosen_and_rejected_for_experiment_based_agent)
                    best_worst_query_pair["experiment"].append({"id": paper_id, "index": current_index_for_experiment, "Queries": {"BEST": best_query_for_experiment_based_agent, "WORST": worst_query_for_experiment_based_agent, "REWARD MARGIN": experiment_margin_of_reward}})
                    experiment_query_and_rewards_list.append(experiment_query_and_rewards)
            
            #### Rank queries for research question based agent results.
            other_agent_retrieved_result = collect_other_agent_retrieved_result(method_based_agent_results, experiment_based_agent_results)
            
            best_query_for_research_question_based_agent, worst_query_for_research_question_based_agent, research_question_margin_of_reward, research_question_query_and_rewards = Rank_Queries(research_question_based_agent_results, other_agent_retrieved_result, original_retrieval_result, correct_candidates, args.use_recall_as_reward, args.use_overlap_ratio_as_reward, args.top_k, args.weight_alpha, args.overlap_threshold)
            
            if args.training_algorithm == "DPO":
                formatted_data_for_research_question_based_agent_chosen = format_dataset(full_paper_directory, "research_question", best_query_for_research_question_based_agent, args.training_algorithm)
                formatted_data_for_research_question_based_agent_rejected = format_dataset(full_paper_directory, "research_question", worst_query_for_research_question_based_agent, args.training_algorithm)
        
                chosen_and_rejected_for_research_question_based_agent = {"chosen": formatted_data_for_research_question_based_agent_chosen, "rejected": formatted_data_for_research_question_based_agent_rejected}
            
            elif args.training_algorithm == "ORPO":
                input_prompt = format_dataset(full_paper_directory, "research_question", best_query_for_research_question_based_agent, args.training_algorithm)

                chosen_and_rejected_for_research_question_based_agent = {"prompt":input_prompt, "chosen": best_query_for_research_question_based_agent, "rejected": worst_query_for_research_question_based_agent}

            current_index_for_research_question = len(best_worst_query_pair["research_question"])

            if research_question_margin_of_reward > args.margin_threshold:
                if best_query_for_research_question_based_agent != worst_query_for_research_question_based_agent:
                    research_question_based_agent_data.append(chosen_and_rejected_for_research_question_based_agent)
                    best_worst_query_pair["research_question"].append({"id": paper_id, "index": current_index_for_research_question, "Queries": {"BEST": best_query_for_research_question_based_agent, "WORST": worst_query_for_research_question_based_agent}, "REWARD MARGIN": research_question_margin_of_reward})
                    research_question_query_and_rewards_list.append(research_question_query_and_rewards)
    

    method_based_agent_data_length = len(method_based_agent_data)
    experiment_based_agent_data_length = len(experiment_based_agent_data)
    research_question_based_agent_data_length = len(research_question_based_agent_data)
    
    print(f"Number of data for method based agent is {method_based_agent_data_length}")
    print(f"Number of data for experiment based agent is {experiment_based_agent_data_length}")
    print(f"Number of data for research question based agent is {research_question_based_agent_data_length}")
    
    num_samples = min(method_based_agent_data_length, experiment_based_agent_data_length, research_question_based_agent_data_length)
    print(f"Number of Samples for Training Set is {num_samples}")
    
    method_based_agent_data = random.sample(method_based_agent_data, num_samples)
    experiment_based_agent_data = random.sample(experiment_based_agent_data, num_samples)
    research_question_based_agent_data = random.sample(research_question_based_agent_data, num_samples)
    
    if args.upload_to_huggingface == True:
        dataset_dict = DatasetDict({
            "method_agent": Dataset.from_list(method_based_agent_data),
            "experiment_agent": Dataset.from_list(experiment_based_agent_data),
            "research_question_agent": Dataset.from_list(research_question_based_agent_data),
        })
        
        original_model_name = args.model
        model_name = original_model_name.split('/')[1]
        
        if args.retriever == "jina-embeddings-v2-base-en":
            retriever_name = "JEmb"
        
        elif args.retriever == "bge-m3":
            retriever_name = "BGE"
            
        elif args.retriever == "inf-retriever-v1-1.5b":
            retriever_name = "INFV"
            
        if args.use_overlap_ratio_as_reward == True:
            if args.use_group_reward == False:
                directory = f"{args.huggingface_directory}_{model_name}_{retriever_name}_use_reference_as_gt_{args.use_reference_as_gt}_use_individual_recall_{args.use_recall_as_reward}_overlap_{args.use_overlap_ratio_as_reward}_top_k_{args.top_k}_overlap_thresh_{args.overlap_threshold}_margin_thresh_{args.margin_threshold}"
            elif args.use_group_reward == True:
                directory = f"{args.huggingface_directory}_{model_name}_{retriever_name}_use_reference_as_gt_{args.use_reference_as_gt}_use_group_recall_{args.use_recall_as_reward}_overlap_{args.use_overlap_ratio_as_reward}_top_k_{args.top_k}_overlap_thresh_{args.overlap_threshold}_margin_thresh_{args.margin_threshold}"
        
        elif args.use_overlap_ratio_as_reward == False:
            if args.use_group_reward == False:
                if args.dataset_for_iterative_retrieval == False:
                    directory = f"{args.huggingface_directory}_{model_name}_{retriever_name}_reference_as_gt_{args.use_reference_as_gt}_individual_recall_{args.use_recall_as_reward}_top_k_{args.top_k}"
                elif args.dataset_for_iterative_retrieval == True:
                    directory = f"{args.huggingface_directory}_{model_name}_{retriever_name}_ref_as_gt_{args.use_reference_as_gt}_IterRet_individual_recall_{args.use_recall_as_reward}_top_k_{args.top_k}"            
            elif args.use_group_reward == True:
                directory = f"{args.huggingface_directory}_{model_name}_{retriever_name}_reference_as_gt_{args.use_reference_as_gt}_group_recall_{args.use_recall_as_reward}_top_k_{args.top_k}"
        
        dataset_dict.push_to_hub(
            directory,
            private=True
        )

# Example usage
if __name__ == "__main__":
    # Abstract to compare
    args = config()
    main(args)
