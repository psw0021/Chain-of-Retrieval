import torch
import faiss
import os
import sys
from collections import namedtuple
import numpy as np
import torch
from torch import nn as nn
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRContextEncoder, AutoTokenizer,DPRContextEncoderTokenizer, \
DPRQuestionEncoder, DPRQuestionEncoderTokenizer, AutoModel, AutoConfig
from adapters import AutoAdapterModel
from sentence_transformers import SentenceTransformer
import json
import torch
from torch import nn as nn
from pathlib import Path
import logging
from openai import OpenAI
import tiktoken
import re
from vllm import LLM, SamplingParams

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
# Add the root directory to Python's module search path
sys.path.append(root_dir)
from Utils.utils import split_paragraph, reformat_to_string
from QueryOptimizer.agents import QueryOptimizer
from Retrieval.metrics import evaluate_retrieval, average_evaluation_metrics

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your GPU setup.")

logging.basicConfig(level=logging.INFO)



class Retriever:
    def __init__(self, args, result_folder_path):
        """
        Our overall retriever that retrieves input paper from massive target paper corpus.
        """
        self.embedding_model = args.embedding_model
        self.use_gpu = torch.cuda.is_available()
        self.embedding_model_max_length = None
        
        self.use_multi_source = args.use_multi_source
        self.use_chunked = args.use_chunked

        self.include_original_retrieval = args.include_original_retrieval

        self.method_to_paper_dict = {}
        self.experiment_to_paper_dict = {}
        self.research_question_to_paper_dict = {}
        self.chunked_corpus_to_paper_dict = {}
        
        root_dir = os.path.dirname(args.corpus_directory)
        self.root_dir = os.path.join(root_dir, self.embedding_model)
        os.makedirs(self.root_dir, exist_ok=True)
        
        if self.use_multi_source == False:
            if self.use_chunked == False:
                self.faiss_index_path = os.path.join(self.root_dir, "DB.faiss")
            elif args.use_chunked == True:
                self.faiss_index_path = os.path.join(self.root_dir, "DB_chunked.faiss")
            
        elif self.use_multi_source == True:
            if args.use_chunked == False:
                self.faiss_index_path = os.path.join(self.root_dir, "DB.faiss")
            elif args.use_chunked == True:
                self.chunked_faiss_index_path = os.path.join(self.root_dir, "DB_chunked.faiss")
                self.faiss_index_path = os.path.join(self.root_dir, "DB.faiss")
            
        
        self.faiss_index = None
        self.candidate_embeddings = None
        
        self.embedding_dimension = None
        self.batch_size = args.batch_size

        self.method_index = None
        self.research_question_index = None
        self.experiment_index = None
        self.chunked_index = None
        
        num_devices = torch.cuda.device_count()
        device_list = []
        for device_number in range(num_devices):
            device_list.append(f"cuda:{device_number}")
        
        if args.deploy_llm == True:
            if args.query_optimizer_model != "Qwen/Qwen2.5-3B-Instruct":
                self.vllm_model = LLM(model=args.query_optimizer_model, tensor_parallel_size=1, max_model_len=60000, gpu_memory_utilization=args.gpu_memory_utilization, dtype="half", device=device_list[0])
            elif args.query_optimizer_model == "Qwen/Qwen2.5-3B-Instruct":
                self.vllm_model = LLM(model=args.query_optimizer_model, tensor_parallel_size=1, max_model_len=60000, rope_scaling={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}, gpu_memory_utilization=args.gpu_memory_utilization, dtype="half", device=device_list[0])
        
        else:
            self.vllm_model = {}
            
        self.device = device_list[-1]

        self.QueryOptimizer = QueryOptimizer(args, result_folder_path, self.vllm_model)

        with open(args.corpus_directory, "r") as json_file:
            self.corpus = json.load(json_file)
        
        if self.embedding_model == "jina-embeddings-v2-base-en":
            self.question_encoder = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True, max_length=8192).to(self.device)
            self.context_encoder = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True, max_length=8192).to(self.device)
            
            self.embedding_model_max_length=8192
            self.embedding_dimension = 768
            
        elif self.embedding_model == "bge-m3":
            embedding_model = SentenceTransformer("BAAI/bge-m3")
            embedding_model.tokenizer.model_max_length = 8192

            self.question_encoder = embedding_model
            self.context_encoder = embedding_model

            logging.info(f"{embedding_model.tokenizer.model_max_length}")
            self.embedding_model_max_length = 8192
            self.embedding_dimension = 1024
            
        elif self.embedding_model == "inf-retriever-v1-1.5b":
            embedding_model = SentenceTransformer("infly/inf-retriever-v1-1.5b" , trust_remote_code=True)
            embedding_model.tokenizer.model_max_length = 32768

            self.question_encoder = embedding_model
            self.context_encoder = embedding_model

            logging.info(f"{embedding_model.tokenizer.model_max_length}")
            self.embedding_model_max_length = 32768
            self.embedding_dimension = 1536
            
      
    def initialize(self):
        self.format_corpus()
        os.makedirs(self.root_dir, exist_ok=True)
        if self.use_multi_source == False:
            self.faiss_index = self.build_faiss_index()
        elif self.use_multi_source == True:
            if self.use_chunked == True:
                self.faiss_index, self.chunked_index = self.build_faiss_index()
    
    
    def format_corpus(self): 
        if self.use_multi_source == True:
            total_chunked_corpus = []
            for paper in self.corpus:
                paper_title = paper["title"]
                paper_abstract = paper["abstract"]
                if self.use_chunked == True:     
                    chunked_sections = paper["chunked_sections"]
                    for chunked_section in chunked_sections:
                        try:
                            self.chunked_corpus_to_paper_dict[chunked_section]
                        except KeyError:
                            total_chunked_corpus.append(chunked_section)
                            self.chunked_corpus_to_paper_dict[chunked_section] = f"Title: {paper_title}\nAbstract: {paper_abstract}"
                    
            self.chunked_corpus = total_chunked_corpus

            formatted_total_corpus = []
            formatted_total_corpus_dict = {}
            for paper in self.corpus:
                paper_title = paper["title"]
                paper_abstract = paper["abstract"]
                formatted_total_candidate = f"Title: {paper_title}\nAbstract: {paper_abstract}"
                
                try:
                    formatted_total_corpus_dict[formatted_total_candidate]
                except:
                    formatted_total_corpus.append(formatted_total_candidate)
                    formatted_total_corpus_dict[formatted_total_candidate] = formatted_total_candidate
                    
            self.corpus = formatted_total_corpus
        
        elif self.use_multi_source == False and self.use_chunked == True:
            total_chunked_corpus = []
            for paper in self.corpus:
                paper_title = paper["title"]
                paper_abstract = paper["abstract"]
                
                chunked_sections = paper["chunked_sections"]
                for chunked_section in chunked_sections:
                    try:
                        self.chunked_corpus_to_paper_dict[chunked_section]
                    except KeyError:
                        total_chunked_corpus.append(chunked_section)
                        self.chunked_corpus_to_paper_dict[chunked_section] = f"Title: {paper_title}\nAbstract: {paper_abstract}"

            self.corpus = total_chunked_corpus

        elif self.use_multi_source == False and self.use_chunked == False:
            formatted_total_corpus = []
            formatted_total_corpus_dict = {}
            for paper in self.corpus:
                paper_title = paper["title"]
                paper_abstract = paper["abstract"]
                    
                formatted_total_candidate = f"Title: {paper_title}\nAbstract: {paper_abstract}"
                
                try:
                    formatted_total_corpus_dict[formatted_total_candidate] = formatted_total_candidate
                except KeyError:
                    formatted_total_corpus.append(formatted_total_candidate)
                    formatted_total_corpus_dict[formatted_total_candidate] = formatted_total_candidate
                    
            self.corpus = formatted_total_corpus
        
        
    def format_query_candidates(self, paper):
        query_paper = paper["Query"]
        query_title = query_paper["title"]
        query_abstract = query_paper["abstract"]
        optimized_queries = query_paper["optimized_queries"]
        formatted_query = []

        if self.include_original_retrieval == True:
            formatted_original_retrieval_query = f"Title: {query_title}\nAbstract: {query_abstract}"
            formatted_query.append(("original_retrieval", [formatted_original_retrieval_query]))
            
        for agent_name, queries in optimized_queries:
            formatted_query.append((agent_name, queries))
        
        formatted_candidates = []
        candidate_papers = paper["Total_Candidate"]
        for candidate in candidate_papers:
            candidate_title = candidate["title"]
            candidate_abstract = candidate["abstract"]
            
            formatted_candidate = f"Title: {candidate_title}\nAbstract: {candidate_abstract}"

            formatted_candidates.append(formatted_candidate)
        
        return formatted_query, formatted_candidates  
        
    # Function to encode passages in batches
    def encode_passages(self, passages):
        if self.embedding_model == "jina-embeddings-v2-base-en":
            encoded_embeddings = []
            with torch.no_grad():
                embeddings = self.context_encoder.encode(passages, device=self.device)
            encoded_embeddings.append(embeddings)
            return np.vstack(encoded_embeddings)

        elif self.embedding_model == "bge-m3":
            encoded_embeddings = []
            with torch.no_grad():
                embeddings = self.context_encoder.encode(passages, device=self.device)
            encoded_embeddings.append(embeddings)
            return np.vstack(encoded_embeddings)

        elif self.embedding_model == "inf-retriever-v1-1.5b":
            encoded_embeddings = []
            with torch.no_grad():
                ## format passages
                embeddings = self.context_encoder.encode(passages, device=self.device)
            encoded_embeddings.append(embeddings)
            
            return np.vstack(encoded_embeddings)


    # Function to encode a query
    def encode_query(self, query):
        def clean_text(text):
            """Remove special tokens and non-printable characters."""
            text = re.sub(r"[^\x20-\x7E]", "", text)  # Keep only printable characters
            text = text.strip()  # Remove leading/trailing spaces
            return text
        
        def truncate(text, number=8192):
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(clean_text(text), disallowed_special=())[:number]

            truncated_text = encoding.decode(tokens)

            return truncated_text
        
        if self.embedding_model == "jina-embeddings-v2-base-en":
            with torch.no_grad():
                embedding = self.question_encoder.encode(query, device=self.device)  # Encode on GPU
            
            embedding = embedding.reshape(1,-1)
            return embedding
        
        elif self.embedding_model == "bge-m3":
            with torch.no_grad():
                embedding = self.question_encoder.encode(query, device=self.device)
            embedding = embedding.reshape(1, -1)
            
            return embedding

        elif self.embedding_model == "inf-retriever-v1-1.5b":
            with torch.no_grad():
                embedding = self.question_encoder.encode(query, device=self.device, prompt_name="query")
            embedding = embedding.reshape(1, -1)
            
            return embedding
    
    # Step 1: Preprocess and index the corpus
    def build_faiss_index(self):
        if self.use_multi_source == False:
            if os.path.exists(self.faiss_index_path):
                logging.info("Loading existing FAISS index...")
                index = faiss.read_index(self.faiss_index_path)
                return index

            logging.info("Building FAISS index...")
            
            # Initialize a FAISS index for L2 distance
            index = faiss.IndexFlatL2(self.embedding_dimension)

            # Process passages in chunks to avoid memory issues
            for i in range(0, len(self.corpus), self.batch_size):
                batch = self.corpus[i:i + self.batch_size]
                logging.info(f"Processing batch {i // self.batch_size + 1} of {len(self.corpus) // self.batch_size + 1}")
                embeddings = self.encode_passages(batch)
                index.add(embeddings)  

            # Save the index to disk
            faiss.write_index(index, self.faiss_index_path)

            logging.info("FAISS index built and saved.")
            return index
        
        elif self.use_multi_source == True:
            if self.use_chunked == True:
                if os.path.exists(self.faiss_index_path):
                    logging.info("Loading existing FAISS index...")
                    index = faiss.read_index(self.faiss_index_path)

                else:
                    # Initialize a FAISS index for L2 distance
                    index = faiss.IndexFlatL2(self.embedding_dimension)

                    # Process passages in chunks to avoid memory issues
                    for i in range(0, len(self.corpus), self.batch_size):
                        batch = self.corpus[i:i + self.batch_size]
                        logging.info(f"Processing batch {i // self.batch_size + 1} of {len(self.corpus) // self.batch_size + 1}")
                        embeddings = self.encode_passages(batch)
                        index.add(embeddings)  

                    # Save the index to disk
                    faiss.write_index(index, self.faiss_index_path)
                    logging.info("FAISS index built and saved.")

                if os.path.exists(self.chunked_faiss_index_path):
                    logging.info("Loading chunked existing FAISS corpus index...")
                    chunked_index = faiss.read_index(self.chunked_faiss_index_path)
                else:
                    chunked_index = faiss.IndexFlatL2(self.embedding_dimension)

                    # Process passages in chunks to avoid memory issues
                    for i in range(0, len(self.chunked_corpus), self.batch_size):
                        batch = self.chunked_corpus[i:i + self.batch_size]
                        logging.info(f"Processing batch {i // self.batch_size + 1} of {len(self.chunked_corpus) // self.batch_size + 1}")

                        embeddings = self.encode_passages(batch)
                        chunked_index.add(embeddings)  

                    # Save the index to disk
                    faiss.write_index(chunked_index, self.chunked_faiss_index_path)
                    logging.info("FAISS index built and saved.")
                    
                return index, chunked_index
    
    # Calculate similarities and retrieve top-k abstracts
    def retrieve_top_k(self, query, top_k, agent_name):        
        query_embedding = self.encode_query(query)  # Encode the query
        if self.use_multi_source == False:
            if self.use_chunked == False:
                distances, indices = self.faiss_index.search(query_embedding, top_k)
                top_k_corpus = [(self.corpus[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
            elif self.use_chunked == True:
                distances, indices = self.faiss_index.search(query_embedding, top_k)
                top_k_corpus = [(self.chunked_corpus_to_paper_dict[self.corpus[idx]], distances[0][i]) for i, idx in enumerate(indices[0])]

                
        elif self.use_multi_source == True:
            if self.use_chunked == True:
                if agent_name != "original_retrieval":
                    distances, indices = self.chunked_index.search(query_embedding, top_k)
                    top_k_corpus = [(self.chunked_corpus_to_paper_dict[self.chunked_corpus[idx]], distances[0][i]) for i, idx in enumerate(indices[0])]
                        
                elif agent_name == "original_retrieval":
                    distances, indices = self.faiss_index.search(query_embedding, top_k)
                    top_k_corpus = [(self.corpus[idx], distances[0][i]) for i, idx in enumerate(indices[0])] 

                    
        return top_k_corpus
    
    def evaluate(self, args, test_file_directory, result_folder_path):
        """
        Evaluate Retrieval Performance on scientific papers submitted to recent venues, such as ICLR 2024, ICLR 2025.
        You can either use query optimizer or not. 
        """
   
        for files in test_file_directory:
            with open(files, "r") as json_file:
                evaluation_data = json.load(json_file)
            
            full_paper_path = evaluation_data["Query"]["full_paper_directory"]
            
            evaluation_data, optimized_queries = self.QueryOptimizer.forward(full_paper_path, evaluation_data)
                
            formatted_query, formatted_correct_candidates = self.format_query_candidates(evaluation_data)
                    
            organized_retrieved_top_corpus = {}
            for agent_name, queries in formatted_query:
                organized_retrieved_top_corpus[agent_name] = {}
                for current_query in queries:
                    top_k_corpus = self.retrieve_top_k(current_query, args.top_k, agent_name)
                    organized_retrieved_top_corpus[agent_name][current_query] = {}
                    if self.use_multi_source == True:
                        for idx, (corpus, score) in enumerate(top_k_corpus):
                            organized_retrieved_top_corpus[agent_name][current_query][f"\nRank {idx + 1}"] = {"Score": f"Score: {score:.4f}", "Content": f"{corpus}"}
                                
                    elif self.use_multi_source == False:
                        if self.use_chunked == False:
                            for idx, (corpus, score) in enumerate(top_k_corpus):
                                organized_retrieved_top_corpus[agent_name][current_query][f"\nRank {idx + 1}"] = {"Score": f"Score: {score:.4f}", "Content": f"{corpus}"}
                        elif self.use_chunked == True:
                            for idx, (corpus, score) in enumerate(top_k_corpus):
                                organized_retrieved_top_corpus[agent_name][current_query][f"\nRank {idx + 1}"] = {"Score": f"Score: {score:.4f}", "Content": f"{corpus}"}
                    
            queries = []
            for current_query in formatted_query:
                query = {'agent': current_query[0], "content": current_query[1]}
                queries.append(query)
                        
            organized_results = {"query": {"id": f"{evaluation_data['id']}", "content": queries}, 
                                     "Retrieved_Candidates": organized_retrieved_top_corpus, "Correct_Candidates": formatted_correct_candidates}
                    
            current_result_folder_path = f"{result_folder_path}/{evaluation_data['id']}.json"
            with open(current_result_folder_path, "w") as json_file:
                    json.dump(organized_results, json_file, indent=4)