import os
import json
import numpy as np
import scipy.stats as stats
import sys

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

def return_immediate_subdirectory(directory):
    subdirs = [entry.path for entry in os.scandir(directory) if entry.is_dir()]

    return subdirs

def compute_statistics(data):
    N = len(data)
    mean = np.mean(data)                         # Compute mean
    std_dev = np.std(data, ddof=1)               # Compute sample standard deviation

    return mean, std_dev

def compute_mrr(ranked_lists, ground_truths, top_k):
    """
    Compute Mean Reciprocal Rank (MRR).

    Args:
        ranked_lists (List[Any]]]): A list of ranked retrieval results
        ground_truths (List[Any]): A list of the correct item (or set of correct items) for each query.

    Returns:
        float: The MRR score.
    """

    ranked_lists = ranked_lists[:top_k]

    rank = 0
    
    for i, item in enumerate(ranked_lists, start=1):
        if item in ground_truths:
            rank = i
            rank_inverse = 1/rank
            
            return rank_inverse
    
    return 0

def precision_at_k(retrieved, relevant, k):
    retrieved_at_k = retrieved[:k]
    relevant_at_k = set(relevant).intersection(set(retrieved_at_k))
    
    return len(relevant_at_k) / k

def compute_MAP(retrieved, relevant, max_k):
    num_relevant = 0
    precisions = []

    for i in range(1, max_k + 1):
        if retrieved[i-1] in relevant:
            num_relevant += 1
            precision_at_i = num_relevant / i
            precisions.append(precision_at_i)

    if len(precisions) == 0:
        return 0.0  

    mean_average_precision = sum(precisions) / len(relevant)
    return mean_average_precision

import math

def compute_NDCG_binary(retrieved, relevant, max_k):
    """
    retrieved: list of retrieved document IDs
    relevant: set of relevant document IDs
    max_k: evaluate at top-k
    """
    dcg = 0.0
    for i in range(1, max_k + 1):
        if i-1 >= len(retrieved):
            break
        doc_id = retrieved[i-1]
        rel_i = 1 if doc_id in relevant else 0
        dcg += (rel_i) / math.log2(i + 1)

    # Ideal DCG 
    ideal_relevances = [1] * min(len(relevant), max_k) 
    idcg = 0.0
    for i in range(1, len(ideal_relevances) + 1):
        idcg += (1) / math.log2(i + 1)

    if idcg == 0.0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg


def compute_recall(retrieved, relevant, k):
    retrieved_at_k = retrieved[:k]
    relevant_at_k = set(relevant).intersection(set(retrieved_at_k))
    return len(relevant_at_k) / len(relevant) if relevant else 0

   
recall_1000 = []
recall_500 = []
recall_300 = []
recall_200 = []
recall_100 = []
recall_50 = []
mrr_50 = []
map_30 = []
ndcg_50 = []
ndcg_100 = []
ndcg_200 = []
ndcg_300 = []
ndcg_500 = []
ndcg_1000 = []


total_test_sets = 0
total_recall_1000 = 0
total_recall_500 = 0
total_recall_300 = 0
total_recall_200 = 0
total_recall_100 = 0
total_recall_50 = 0
total_Mrr_50 = 0
total_map_30 = 0
total_ndcg_50 = 0
total_ndcg_100 = 0
total_ndcg_200 = 0
total_ndcg_300 = 0
total_ndcg_500 = 0
total_ndcg_1000 = 0
number_of_test_sets = 0


current = ""

file_list = traverse_folder(current)
print(current)
for file in file_list:
    with open(file, "r") as json_file:
        try:
                data = json.load(json_file)
        except:
                print(file)
                sys.exit()
    try:
            current_results = data["Current Result"]
    except:
            continue
    try:
            final_retrieved_result = data["Final_Ranked_Results"]
    except:
            retrieved_result = data["Retrieved_Candidates"]
            retrieved_result_keys = retrieved_result.keys()
            final_retrieved_result = []
            for result_key in retrieved_result_keys:
                final_retrieved_result.append(retrieved_result[result_key]["Content"])
                
        
    correct_candidates = data["Correct_Candidates"]
        

    total_test_sets = total_test_sets + 1
    
    recall300 = current_results["Recall@300"]
    total_recall_300 = total_recall_300 + recall300
    
    recall200 = current_results["Recall@200"]
    total_recall_200 = total_recall_200 + recall200
    
    recall100 = current_results["Recall@100"]
    total_recall_100 = total_recall_100 + recall100

    current_mrr_50 = compute_mrr(final_retrieved_result, correct_candidates, 50)
    total_Mrr_50 = total_Mrr_50 + current_mrr_50

    current_map_30 = compute_MAP(final_retrieved_result, correct_candidates, 30)
    total_map_30 = total_map_30 + current_map_30
        
    if len(final_retrieved_result) >= 1000:
            current_recall_1000 = compute_recall(final_retrieved_result, correct_candidates, 1000)
            total_recall_1000 = total_recall_1000 + current_recall_1000

    if len(final_retrieved_result) >= 500:
            current_recall_500 = compute_recall(final_retrieved_result, correct_candidates, 500)
            total_recall_500 = total_recall_500 + current_recall_500
            
    current_recall_50 = compute_recall(final_retrieved_result, correct_candidates, 50)
    total_recall_50 = total_recall_50 + current_recall_50

    current_ndcg_50 = compute_NDCG_binary(final_retrieved_result, correct_candidates, 50)
    total_ndcg_50 = total_ndcg_50 + current_ndcg_50

    current_ndcg_100 = compute_NDCG_binary(final_retrieved_result, correct_candidates, 100)
    total_ndcg_100 = total_ndcg_100 + current_ndcg_100

    current_ndcg_200 = compute_NDCG_binary(final_retrieved_result, correct_candidates, 200)
    total_ndcg_200 = total_ndcg_200 + current_ndcg_200
    
    current_ndcg_300 = compute_NDCG_binary(final_retrieved_result, correct_candidates, 300)
    total_ndcg_300 = total_ndcg_300 + current_ndcg_300
    
    if len(final_retrieved_result) >= 500:
        current_ndcg_500 = compute_NDCG_binary(final_retrieved_result, correct_candidates, 500)
        total_ndcg_500 = total_ndcg_500 + current_ndcg_500
        
    if len(final_retrieved_result) >= 1000:
        current_ndcg_1000 = compute_NDCG_binary(final_retrieved_result, correct_candidates, 1000)
        total_ndcg_1000 = total_ndcg_1000 + current_ndcg_1000
        
    number_of_test_sets = number_of_test_sets + 1

    
#if number_of_test_sets != 400:
    #raise ValueError("Less than 400 samples")
average_recall_1000 = total_recall_1000 / number_of_test_sets
average_recall_500 = total_recall_500 / number_of_test_sets
average_recall_300 = total_recall_300 / number_of_test_sets
average_recall_200 = total_recall_200 / number_of_test_sets
average_recall_100 = total_recall_100 / number_of_test_sets
average_recall_50 = total_recall_50 / number_of_test_sets
average_mrr_50 = total_Mrr_50 / number_of_test_sets
average_map_30 = total_map_30 / number_of_test_sets
average_ndcg_50 = total_ndcg_50 / number_of_test_sets
average_ndcg_100 = total_ndcg_100 / number_of_test_sets
average_ndcg_200 = total_ndcg_200 / number_of_test_sets
average_ndcg_300 = total_ndcg_300 / number_of_test_sets
average_ndcg_500 = total_ndcg_500 / number_of_test_sets
average_ndcg_1000 = total_ndcg_1000 / number_of_test_sets

recall_1000.append(average_recall_1000)
recall_500.append(average_recall_500)
recall_300.append(average_recall_300)
recall_200.append(average_recall_200)
recall_100.append(average_recall_100)
recall_50.append(average_recall_50)
mrr_50.append(average_mrr_50)
map_30.append(average_map_30)
ndcg_50.append(average_ndcg_50)
ndcg_100.append(average_ndcg_100)
ndcg_200.append(average_ndcg_200)
ndcg_300.append(average_ndcg_300)
ndcg_500.append(average_ndcg_500)
ndcg_1000.append(average_ndcg_1000)

recall_1000_mean, recall_1000_std = compute_statistics(recall_1000)
recall_500_mean, recall_500_std = compute_statistics(recall_500)
recall_300_mean, recall_300_std = compute_statistics(recall_300)
recall_200_mean, recall_200_std = compute_statistics(recall_200)
recall_100_mean, recall_100_std = compute_statistics(recall_100)
recall_50_mean, recall_50_std = compute_statistics(recall_50)
mrr_50_mean, mrr_50_std = compute_statistics(mrr_50)
map_30_mean, map_30_std = compute_statistics(map_30)
ndcg_50_mean, ndcg_50_std = compute_statistics(ndcg_50)
ndcg_100_mean, ndcg_100_std = compute_statistics(ndcg_100)
ndcg_200_mean, ndcg_200_std = compute_statistics(ndcg_200)
ndcg_300_mean, ndcg_300_std = compute_statistics(ndcg_300)
ndcg_500_mean, ndcg_500_std = compute_statistics(ndcg_500)
ndcg_1000_mean, ndcg_1000_std = compute_statistics(ndcg_1000)

print(f"Number of Total Test sets is {total_test_sets}")
print(f"Average Recall@1000 is {recall_1000_mean}, with standard deviation of {recall_1000_std}")
print(f"Average Recall@500 is {recall_500_mean}, with standard deviation of {recall_500_std}")
print(f"Average Recall@300 is {recall_300_mean}, with standard deviation of {recall_300_std}")
print(f"Average Recall@200 is {recall_200_mean}, with standard deviation of {recall_200_std}")
print(f"Average Recall@100 is {recall_100_mean}, with standard deviation of {recall_100_std}")
print(f"Average Recall@50 is {recall_50_mean}, with standard deviation of {recall_50_std}")
print(f"Average MRR@50 is {mrr_50_mean}, with standard deviation of {mrr_50_std}")
print(f"Average MAP@30 is {map_30_mean}, with standard deviation of {map_30_std}")
print(f"Average ndcg@50 is {ndcg_50_mean}, with standard deviation of {ndcg_50_std}")
print(f"Average ndcg@100 is {ndcg_100_mean}, with standard deviation of {ndcg_100_std}")
print(f"Average ndcg@200 is {ndcg_200_mean}, with standard deviation of {ndcg_200_std}")
print(f"Average ndcg@300 is {ndcg_300_mean}, with standard deviation of {ndcg_300_std}")
print(f"Average ndcg@500 is {ndcg_500_mean}, with standard deviation of {ndcg_500_std}")
print(f"Average ndcg@1000 is {ndcg_1000_mean}, with standard deviation of {ndcg_1000_std}")