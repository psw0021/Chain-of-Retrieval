import torch
import faiss
import os
import sys
from collections import namedtuple
import numpy as np
import torch
from torch import nn as nn
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from adapters import AutoAdapterModel
import json
import torch
from torch import nn as nn
from pathlib import Path
import logging
from openai import OpenAI
import tiktoken
import re
from vllm import LLM
from scipy.spatial.distance import cdist

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
# Add the root directory to Python's module search path
sys.path.append(root_dir)
from Utils.utils import split_paragraph, reformat_to_string
from Agents.QueryOptimizer_for_patents import QueryOptimizer
from Agents.Selector_for_patents import Selector
from Retrieval.metrics import evaluate_retrieval

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your GPU setup.")

logging.basicConfig(level=logging.INFO)


class Tree_Node:
    def __init__(self, current_node_name, parent_node_name, retrieved_result):
        """
        Tree Node for saving parents, children, queries, and their retrieved result
        """
        self.current_node_name = current_node_name
        self.parent = parent_node_name
        self.retrieved_result = retrieved_result
        self.method_wise_child = None
        self.experiment_wise_child = None
        self.research_question_wise_child = None


class RetrieverPatents:
    def __init__(self, args, result_folder_path):
        """
        Our overall retriever class that retrieves input paper from massive target paper corpus.
        """
        self.benchmark_directory = args.benchmark_directory
        self.embedding_model = args.embedding_model
        self.use_gpu = torch.cuda.is_available()
        self.embedding_model_max_length = None

        self.iterative_retrieval = args.iterative_retrieval
        self.SubTreeSearch = args.SubTreeSearch
        
        self.total_iterative_retrieval_loop = args.total_iterative_retrieval_loop
        
        self.use_multi_source = args.use_multi_source

        self.use_chunked = args.use_chunked
        self.use_full_paper_as_corpus = args.use_full_paper_as_corpus
        self.chunk_unit = args.chunk_unit
    
        self.use_gpt = args.use_gpt

        self.include_original_retrieval = args.include_original_retrieval
        self.use_full_paper_as_query = args.use_full_paper_as_query

        self.number_of_total_queries = 0
        
        if args.use_background_agent == True:
            self.number_of_total_queries += 1
            
        if args.use_claim_agent == True:
            self.number_of_total_queries += 1
            
        if args.use_method_agent == True:
            self.number_of_total_queries += 1
        
        if self.include_original_retrieval == True:
            self.number_of_total_queries += 1

        self.chunked_corpus_to_paper_dict = {}
        self.chunked_corpus_to_full_context_dict = {}
        self.full_paper_corpus_to_paper_dict = {}
        
        self.abstract_to_full_context_dict = {}
        
        root_dir = os.path.dirname(args.corpus_directory)
        self.root_dir = os.path.join(root_dir, self.embedding_model)
        os.makedirs(self.root_dir, exist_ok=True)
        

        if self.use_multi_source == False:
            if self.use_chunked == False and self.use_full_paper_as_corpus == False:
                self.faiss_index_path = os.path.join(self.root_dir, "DB.faiss")
            elif args.use_chunked == True and self.use_full_paper_as_corpus == False:
                self.faiss_index_path = os.path.join(self.root_dir, f"DB_chunked_{self.chunk_unit}.faiss")
            elif args.use_chunked == False and self.use_full_paper_as_corpus == True:
                self.faiss_index_path = os.path.join(self.root_dir, f"DB_full_paper.faiss")
                
        elif self.use_multi_source == True:
            if args.use_chunked == False and self.use_full_paper_as_corpus == False:
                self.faiss_index_path = os.path.join(self.root_dir, "DB.faiss")
            elif args.use_chunked == True and self.use_full_paper_as_corpus == False:
                self.chunked_faiss_index_path = os.path.join(self.root_dir, f"DB_chunked_{self.chunk_unit}.faiss")
                self.faiss_index_path = os.path.join(self.root_dir, "DB.faiss")
            elif args.use_chunked == False and self.use_full_paper_as_corpus == True:
                self.full_paper_faiss_index_path = os.path.join(self.root_dir, f"DB_full_paper.faiss")
                self.faiss_index_path = os.path.join(self.root_dir, "DB.faiss")
        
        self.candidate_embeddings = None
        
        self.embedding_dimension = None
        self.use_query_optimizer = args.use_query_optimizer
        self.batch_size = args.batch_size
        self.only_A2A = False

        self.faiss_index = None
        self.chunked_index = None
        self.full_paper_index = None
        
        self.vllm_dict_for_query_optimizer = {}
        num_devices = torch.cuda.device_count()
        
        device_list = []
        for device_number in range(num_devices):
            device_list.append(f"cuda:{device_number}")

        assert len(device_list) > 0, f"Expected at least 1 devices, but found {len(device_list)}"
        if self.use_gpt == False and self.use_query_optimizer == True: 
            assert len(device_list) >= 1, f"Expected at least 1 device, but found {len(device_list)}"
            if args.query_optimizer_model != "Qwen/Qwen2.5-3B-Instruct":
                    base_model = LLM(model=args.query_optimizer_model, tensor_parallel_size=1, gpu_memory_utilization=args.gpu_memory_utilization, dtype="half", device=device_list[0])
            elif args.query_optimizer_model == "Qwen/Qwen2.5-3B-Instruct":
                    base_model = LLM(model=args.query_optimizer_model, tensor_parallel_size=1, max_model_len=131072, rope_scaling={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}, gpu_memory_utilization=args.gpu_memory_utilization, dtype="half", device=device_list[0])
                
            self.vllm_dict_for_query_optimizer["METHOD"] = {"agent": base_model, "device": device_list[0]}
            self.vllm_dict_for_query_optimizer["CLAIM"] = {"agent": base_model, "device": device_list[0]}
            self.vllm_dict_for_query_optimizer["BACKGROUND"] = {"agent": base_model, "device": device_list[0]}
            
        self.cuda_embedding_model_device = device_list[-1]

        if self.use_query_optimizer == True:
            self.QueryOptimizer = QueryOptimizer(args, result_folder_path, self.vllm_dict_for_query_optimizer)
            
        if self.iterative_retrieval == True and self.SubTreeSearch == True:
            self.Selector = Selector(args)

        with open(args.corpus_directory, "r") as json_file:
            self.corpus = json.load(json_file)
        
        if self.embedding_model == "jina-embeddings-v2-base-en":
            self.question_encoder = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True, max_length=8192).to(self.cuda_embedding_model_device)
            self.context_encoder = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True, max_length=8192).to(self.cuda_embedding_model_device)
            
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
            
        elif self.embedding_model == "paecter":
            embedding_model = SentenceTransformer("mpi-inno-comp/paecter" , trust_remote_code=True)
            embedding_model.tokenizer.model_max_length = 512

            self.question_encoder = embedding_model
            self.context_encoder = embedding_model

            logging.info(f"{embedding_model.tokenizer.model_max_length}")
            self.embedding_model_max_length = 512
            self.embedding_dimension = 1024
            
        elif self.embedding_model == "pat_specter":
            embedding_model = SentenceTransformer("mpi-inno-comp/pat_specter" , trust_remote_code=True)
            embedding_model.tokenizer.model_max_length = 512

            self.question_encoder = embedding_model
            self.context_encoder = embedding_model

            logging.info(f"{embedding_model.tokenizer.model_max_length}")
            self.embedding_model_max_length = 512
            self.embedding_dimension = 768
            
     
    def initialize(self):
        self.format_corpus()
        os.makedirs(self.root_dir, exist_ok=True)
        if self.use_multi_source == False:
            self.faiss_index = self.build_faiss_index()
        elif self.use_multi_source == True:
            if self.use_chunked == False and self.use_full_paper_as_corpus == True:
                self.faiss_index, self.full_paper_index = self.build_faiss_index()
            elif self.use_chunked == True and self.use_full_paper_as_corpus == False:
                self.faiss_index, self.chunked_index = self.build_faiss_index()


    def format_corpus(self):
        if self.use_multi_source == True:
            total_chunked_corpus = []
            total_full_paper_corpus = []
            for paper in self.corpus:
                paper_title = paper["title"]
                paper_abstract = paper["abstract"]
                
                if self.use_chunked == False and self.use_full_paper_as_corpus == True:
                    full_paper_directory = paper["full_paper_directory"]
                    full_paper_root = "LongDocumentBench/Patents"
                    full_paper_directory = os.path.join(full_paper_root, full_paper_directory)
                    with open(full_paper_directory, "r") as file:
                        full_paper_content = file.read()

                    try:
                        self.full_paper_corpus_to_paper_dict[full_paper_content]
                    except KeyError:
                        total_full_paper_corpus.append(full_paper_content)
                        self.full_paper_corpus_to_paper_dict[full_paper_content] = f"Title: {paper_title}\nAbstract: {paper_abstract}"

                elif self.use_chunked == True and self.use_full_paper_as_corpus == False:     
                    chunked_sections = paper[f"chunked_sections_{self.chunk_unit}"]
                    full_paper_directory = paper["full_paper_directory"]
                    full_paper_root = "LongDocumentBench/Patents"
                    full_paper_directory = os.path.join(full_paper_root, full_paper_directory)
                    with open(full_paper_directory, "r") as file:
                        full_paper_content = file.read()
                    for chunked_section in chunked_sections:
                        try:
                            self.chunked_corpus_to_paper_dict[chunked_section]
                        except KeyError:
                            total_chunked_corpus.append(chunked_section)
                            self.chunked_corpus_to_paper_dict[chunked_section] = f"Title: {paper_title}\nAbstract: {paper_abstract}"
                            self.chunked_corpus_to_full_context_dict[chunked_section] = full_paper_content
                    

            self.chunked_corpus = total_chunked_corpus
            self.full_paper_corpus = total_full_paper_corpus

            formatted_total_corpus = []
            formatted_total_corpus_dict = {}
            for paper in self.corpus:
                paper_title = paper["title"]
                paper_abstract = paper["abstract"]
                
                full_paper_directory = paper["full_paper_directory"]
                full_paper_root = "LongDocumentBench/Patents"
                full_paper_directory = os.path.join(full_paper_root, full_paper_directory)
                with open(full_paper_directory, "r") as file:
                    full_paper_content = file.read()
                    
                formatted_total_candidate = f"Title: {paper_title}\nAbstract: {paper_abstract}"
                
                try:
                    formatted_total_corpus_dict[formatted_total_candidate]
                except:
                    formatted_total_corpus.append(formatted_total_candidate)
                    formatted_total_corpus_dict[formatted_total_candidate] = formatted_total_candidate
                    self.abstract_to_full_context_dict[formatted_total_candidate] = full_paper_content
                    
            self.corpus = formatted_total_corpus
        
        elif self.use_multi_source == False and self.use_chunked == True and self.use_full_paper_as_corpus == False:
            total_chunked_corpus = []
            for paper in self.corpus:
                paper_title = paper["title"]
                paper_abstract = paper["abstract"]
                
                full_paper_directory = paper["full_paper_directory"]
                full_paper_root = "LongDocumentBench/Patents"
                full_paper_directory = os.path.join(full_paper_root, full_paper_directory)
                with open(full_paper_directory, "r") as file:
                    full_paper_content = file.read()
                
                chunked_sections = paper[f"chunked_sections_{self.chunk_unit}"]
                for chunked_section in chunked_sections:
                    try:
                        self.chunked_corpus_to_paper_dict[chunked_section]
                    except KeyError:
                        total_chunked_corpus.append(chunked_section)
                        self.chunked_corpus_to_paper_dict[chunked_section] = f"Title: {paper_title}\nAbstract: {paper_abstract}"
                        self.chunked_corpus_to_full_context_dict[chunked_section] = full_paper_content

            self.corpus = total_chunked_corpus
        
        elif self.use_multi_source == False and self.use_chunked == False and self.use_full_paper_as_corpus == True:
            print("Formatting corpus for full paper for single source")
            total_full_paper_corpus = []
            for paper in self.corpus:
                paper_title = paper["title"]
                paper_abstract = paper['abstract']
                
                full_paper_directory = paper["full_paper_directory"]
                full_paper_root = "LongDocumentBench/Patents"
                
                full_paper_directory = os.path.join(full_paper_root, full_paper_directory)
                with open(full_paper_directory, "r") as file:
                    full_paper_content = file.read()
                    
                try:
                    self.full_paper_corpus_to_paper_dict[full_paper_content]
                except KeyError:
                    total_full_paper_corpus.append(full_paper_content)
                    self.full_paper_corpus_to_paper_dict[full_paper_content] = f"Title: {paper_title}\nAbstract: {paper_abstract}"
            
            self.corpus = total_full_paper_corpus

        elif self.use_multi_source == False and self.use_chunked == False and self.use_full_paper_as_corpus == False:
            formatted_total_corpus = []
            formatted_total_corpus_dict = {}
            for paper in self.corpus:
                paper_title = paper["title"]
                paper_abstract = paper["abstract"]
                    
                formatted_total_candidate = f"Title: {paper_title}\nAbstract: {paper_abstract}"
                try:
                    formatted_total_corpus_dict[formatted_total_candidate]
                except KeyError:   
                    formatted_total_corpus.append(formatted_total_candidate)
                    formatted_total_corpus_dict[formatted_total_candidate] = formatted_total_candidate
                    
            self.corpus = formatted_total_corpus
        
        
    def format_query_candidates(self, paper):
        if self.use_query_optimizer == False:
            if self.use_full_paper_as_query == False:
                query_paper = paper["Query"]
                query_title = query_paper["title"]
                query_abstract = query_paper["abstract"]
            
                formatted_query = f"Title: {query_title}\nAbstract: {query_abstract}"

            elif self.use_full_paper_as_query == True:
                paper_id = paper["id"]
                paper_filename = f"{paper_id}.mmd"
                
                benchmark_root_dir = os.path.dirname(self.benchmark_directory)
                root_folder_name = os.path.join(benchmark_root_dir, "Full_Paper") 

                full_paper_path = os.path.join(root_folder_name, paper_filename)

                logging.info(f"{full_paper_path}")

                with open(full_paper_path, "r") as file:
                    paper_content = file.read()

                formatted_query = paper_content
        
            
        elif self.use_query_optimizer == True:
            query_paper = paper["Query"]
            query_title = query_paper["title"]
            optimized_queries = query_paper["optimized_queries"]
            formatted_query = []

            if self.include_original_retrieval == True:
                query_abstract = query_paper["abstract"]
                formatted_query.append(("Original Retrieval", query_abstract))

            for agent_name, query in optimized_queries:
                formatted_optimized_query = f"{query}"
                formatted_query.append((agent_name, formatted_optimized_query))
        
        formatted_candidates = []
        candidate_papers = paper["Candidate"]
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
                embeddings = self.context_encoder.encode(passages, device=self.cuda_embedding_model_device)
            encoded_embeddings.append(embeddings)
            return np.vstack(encoded_embeddings)
        
        elif self.embedding_model == "bge-m3":
            encoded_embeddings = []
            with torch.no_grad():
                embeddings = self.context_encoder.encode(passages, device=self.cuda_embedding_model_device)
            encoded_embeddings.append(embeddings)
            return np.vstack(encoded_embeddings)
        
        elif self.embedding_model == "inf-retriever-v1-1.5b":
            encoded_embeddings = []
            with torch.no_grad():
                ## format passages
                embeddings = self.context_encoder.encode(passages, device=self.cuda_embedding_model_device)
            encoded_embeddings.append(embeddings)
            
            return np.vstack(encoded_embeddings)
        
        elif self.embedding_model == "paecter":
            encoded_embeddings = []
            with torch.no_grad():
                ## format passages
                embeddings = self.context_encoder.encode(passages, device=self.cuda_embedding_model_device)
            encoded_embeddings.append(embeddings)
            
            return np.vstack(encoded_embeddings)
        
        elif self.embedding_model == "pat_specter":
            encoded_embeddings = []
            with torch.no_grad():
                ## format passages
                embeddings = self.context_encoder.encode(passages, device=self.cuda_embedding_model_device)
            encoded_embeddings.append(embeddings)
            
            return np.vstack(encoded_embeddings)

    # Function to encode a query
    def encode_query(self, query):
        if self.embedding_model == "jina-embeddings-v2-base-en":
            with torch.no_grad():
                embedding = self.question_encoder.encode(query, device=self.cuda_embedding_model_device)  # Encode on GPU
            
            embedding = embedding.reshape(1,-1)
            return embedding
        
        elif self.embedding_model == "bge-m3":
            with torch.no_grad():
                embedding = self.question_encoder.encode(query, device=self.cuda_embedding_model_device)
            embedding = embedding.reshape(1, -1)
            
            return embedding
        
        elif self.embedding_model == "inf-retriever-v1-1.5b":
            with torch.no_grad():
                embedding = self.question_encoder.encode(query, device=self.cuda_embedding_model_device, prompt_name="query")
            embedding = embedding.reshape(1, -1)
            
            return embedding
        
        elif self.embedding_model == "paecter":
            with torch.no_grad():
                embedding = self.question_encoder.encode(query, device=self.cuda_embedding_model_device)
            embedding = embedding.reshape(1, -1)
            
            return embedding
        
        elif self.embedding_model == "pat_specter":
            with torch.no_grad():
                embedding = self.question_encoder.encode(query, device=self.cuda_embedding_model_device)
            embedding = embedding.reshape(1, -1)
            
            return embedding
      
    
    # Step 1: Preprocess and index the corpus
    def build_faiss_index(self):
        if self.use_multi_source == False:
            if os.path.exists(self.faiss_index_path):
                logging.info(f"Loading existing FAISS index...{self.faiss_index_path}")
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
            if self.use_chunked == False and self.use_full_paper_as_corpus == True:
                if os.path.exists(self.faiss_index_path):
                    logging.info("Loading existing FAISS index...")
                    index = faiss.read_index(self.faiss_index_path)

                else:
                    logging.info("Creating base FAISS corpus index...")
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

                if os.path.exists(self.full_paper_faiss_index_path):
                    logging.info("Loading existing FAISS corpus index with full papers...")
                    full_paper_index = faiss.read_index(self.full_paper_faiss_index_path)
                else:
                    logging.info("Creating full paper FAISS corpus index...")
                    full_paper_index = faiss.IndexFlatL2(self.embedding_dimension)

                    # Process passages in chunks to avoid memory issues
                    for i in range(0, len(self.full_paper_corpus), self.batch_size):
                        batch = self.full_paper_corpus[i:i + self.batch_size]
                        logging.info(f"Processing batch {i // self.batch_size + 1} of {len(self.full_paper_corpus) // self.batch_size + 1}")

                        embeddings = self.encode_passages(batch)
                        full_paper_index.add(embeddings)  

                    # Save the index to disk
                    faiss.write_index(full_paper_index, self.full_paper_faiss_index_path)
                    logging.info("FAISS index built and saved.")
                    
                return index, full_paper_index
            
            elif self.use_chunked == True and self.use_full_paper_as_corpus == False:
                if os.path.exists(self.faiss_index_path):
                    logging.info("Loading existing FAISS index...")
                    index = faiss.read_index(self.faiss_index_path)

                else:
                    logging.info("Creating base corpus index...")
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
                    logging.info("Creating chunked FAISS corpus index...")
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
    def retrieve_top_k(self, query, top_k, agent_name) -> list:
            """
            Retrieve top k papers for given input query from target corpus
            """     
            query_embedding = self.encode_query(query)  # Encode the query
            if self.use_multi_source == False:
                if self.use_chunked == False and self.use_full_paper_as_corpus == False:
                    distances, indices = self.faiss_index.search(query_embedding, top_k)
                    top_k_corpus = [(self.corpus[idx], distances[0][i]) for i, idx in enumerate(indices[0])]

                elif self.use_chunked == True and self.use_full_paper_as_corpus == False:
                    distances, indices = self.faiss_index.search(query_embedding, top_k)
                    top_k_corpus = [(self.chunked_corpus_to_paper_dict[self.corpus[idx]], distances[0][i], self.corpus[idx]) for i, idx in enumerate(indices[0])]

                elif self.use_chunked == False and self.use_full_paper_as_corpus == True:
                    distances, indices = self.faiss_index.search(query_embedding, top_k)
                    top_k_corpus = [(self.full_paper_corpus_to_paper_dict[self.corpus[idx]], distances[0][i], self.corpus[idx]) for i, idx in enumerate(indices[0])]

                    
            elif self.use_multi_source == True:                        
                if agent_name == "Original Retrieval":
                    logging.info("original retrieval")
                    distances, indices = self.faiss_index.search(query_embedding, top_k)
                    top_k_corpus = [(self.corpus[idx], distances[0][i], self.corpus[idx]) for i, idx in enumerate(indices[0])]
                else:
                    logging.info("currently searching using agent optimized queries")
                        
                    if self.use_chunked == True and self.use_full_paper_as_corpus == False:
                        distances, indices = self.chunked_index.search(query_embedding, top_k)
                        top_k_corpus = [(self.chunked_corpus_to_paper_dict[self.chunked_corpus[idx]], distances[0][i], self.chunked_corpus[idx]) for i, idx in enumerate(indices[0])]
                    elif self.use_chunked == False and self.use_full_paper_as_corpus == True:
                        distances, indices = self.full_paper_index.search(query_embedding, top_k)
                        top_k_corpus = [(self.full_paper_corpus_to_paper_dict[self.full_paper_corpus[idx]], distances[0][i], self.full_paper_corpus[idx]) for i, idx in enumerate(indices[0])]

            return top_k_corpus    
    
    def Total_Merge_Results(self, args, organized_retrieved_top_corpus_per_agent) -> list:
        """
        Merge Each aspect-related results as a whole using Reciprocal Rank Fusion subsequent to chain of retrieval.
        """
        def compute_RRF(args, organized_retrieved_top_corpus_list) -> list:
            """
            Compute RRF between Organized Retrieved Results without differentiating the depth and origin of its retrieved result.
            """
            fused_scores = {}
            for top_k_corpus in organized_retrieved_top_corpus_list:
                for rank, corpus in enumerate(top_k_corpus):
                    fused_scores[corpus] = fused_scores.get(corpus, 0) + 1 / (args.hyperparameter_RRF + rank + 1)
                                
            sorted_docs = [doc for doc, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
            parsed_retrieved_top_corpus = sorted_docs[:args.top_k]
                        
            return parsed_retrieved_top_corpus
        
        per_agent_list = list(organized_retrieved_top_corpus_per_agent.keys())
        total_retrieved_result = []
        for agent_name in per_agent_list:
            if agent_name != "Original Retrieval":
                tree_structure_previous_memory = organized_retrieved_top_corpus_per_agent[agent_name]
                total_agent_depth_merged_results = []
                for iteration in range(0, self.total_iterative_retrieval_loop):
                    per_depth_saved_result_nodes = tree_structure_previous_memory[iteration]

                    for node in per_depth_saved_result_nodes:
                        total_agent_depth_merged_results.append(node.retrieved_result)

                logging.info(f"Number of Total corpus to merge for {agent_name} is {len(total_agent_depth_merged_results)}")
                per_agent_merged_result = compute_RRF(args, total_agent_depth_merged_results)

                total_retrieved_result.append(per_agent_merged_result)

        if self.use_multi_source == True and self.include_original_retrieval == True:
            total_retrieved_result.append(organized_retrieved_top_corpus_per_agent["Original Retrieval"][0][0].retrieved_result)

        logging.info(f"Number of final corpus to merge is {len(total_retrieved_result)}")
        return total_retrieved_result


    
    def Recursive_Merge_Results(self, args, organized_retrieved_top_corpus_per_agent) -> list:
        """
        Recursively Merge Results for Iterative Tree Search subsequent to chain of retrieval
        """
        def compute_RRF(args, organized_tree_nodes_per_agent) -> list:
            """
            Compute RRF between Organized Nodes
            """
            fused_scores = {}
            organized_retrieved_top_corpus_list = []
            for node in organized_tree_nodes_per_agent:
                if len(node.retrieved_result) != 0:
                    logging.info("#####################")
                    logging.info(f"The current node name is {node.current_node_name}")
                    logging.info(f"The parent node of this current node is {node.parent}")
                    logging.info("#####################")
                organized_retrieved_top_corpus_list.append(node.retrieved_result)
                
            for top_k_corpus in organized_retrieved_top_corpus_list:
                for rank, corpus in enumerate(top_k_corpus):
                    fused_scores[corpus] = fused_scores.get(corpus, 0) + 1 / (args.hyperparameter_RRF + rank + 1)
                                
            sorted_docs = [doc for doc, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
            parsed_retrieved_top_corpus = sorted_docs[:args.top_k]
                        
            return parsed_retrieved_top_corpus
        
        def compute_RRF_between_parent_child(args, organized_retrieved_top_corpus_list) -> list:
            """
            Compute RRF between neighboring parent and child
            """
            fused_scores = {}
            for rank, corpus in enumerate(organized_retrieved_top_corpus_list[0]):
                fused_scores[corpus] = fused_scores.get(corpus, 0) + 1 / (args.hyperparameter_RRF + rank + 1)

            for rank, corpus in enumerate(organized_retrieved_top_corpus_list[1]):
                fused_scores[corpus] = fused_scores.get(corpus, 0) + 1 / (args.hyperparameter_RRF + rank + 1)

            sorted_docs = [doc for doc, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
            parsed_retrieved_top_corpus = sorted_docs[:args.top_k]
                        
            return parsed_retrieved_top_corpus

        per_agent_list = list(organized_retrieved_top_corpus_per_agent.keys())
        total_retrieved_result = []
        for agent_name in per_agent_list:
            if agent_name != "Original Retrieval":
                tree_structure_previous_memory = organized_retrieved_top_corpus_per_agent[agent_name]
                previous_depth_merged_results = []
                for iteration in range(0, self.total_iterative_retrieval_loop):
                    if iteration == 0:
                        previous_depth_merged_results.append([])
                        results_from_depth = tree_structure_previous_memory[self.total_iterative_retrieval_loop - 1 - iteration]
                        
                        number_of_results_from_current_depth = len(results_from_depth)
                        
                        logging.info(f"Number of results from current depth is {number_of_results_from_current_depth}")
                        for index in range(0, number_of_results_from_current_depth, self.number_of_total_queries):
                            organized_tree_nodes = results_from_depth[index:index + self.number_of_total_queries]
                            logging.info(f"Length of organized tree nodes is {len(organized_tree_nodes)}")
                            
                            ### Check whether they have same parents
                            current_parent_name = organized_tree_nodes[0].parent
                            for node in organized_tree_nodes:
                                parent_name = node.parent
                                if parent_name != current_parent_name:
                                    raise ValueError("Currently Tree Nodes have different parents")
                                
                            merged_siblings = compute_RRF(args, organized_tree_nodes)
                        
                            previous_depth_merged_results[iteration].append((current_parent_name, merged_siblings))

                    else:
                        previous_depth_merged_results.append([])
                        previous_merged_result = previous_depth_merged_results[iteration - 1]
                        results_from_current_depth = tree_structure_previous_memory[self.total_iterative_retrieval_loop - 1 - iteration]
                        index = 0
                        
                        for idx, tree_node in enumerate(results_from_current_depth):
                            agent_name = tree_node.current_node_name
                            top_k_corpus = tree_node.retrieved_result
                            if agent_name != "Original Retrieval":
                                organized_retrieved_top_corpus_parent_and_child = [top_k_corpus, previous_merged_result[index][1]]
                                if agent_name != previous_merged_result[index][0]:
                                    raise ValueError("Cannot merge results from different children")
                                
                                depth_wise_merged_result = compute_RRF_between_parent_child(args, organized_retrieved_top_corpus_parent_and_child)
                                
                                ## update the retrieved result from higher depth
                                tree_structure_previous_memory[self.total_iterative_retrieval_loop -1 - iteration][idx].retrieved_result = depth_wise_merged_result
                                
                                index = index + 1

                        
                        number_of_results_from_current_depth = len(tree_structure_previous_memory[self.total_iterative_retrieval_loop - 1 - iteration])
                        results_from_depth = tree_structure_previous_memory[self.total_iterative_retrieval_loop - 1 - iteration]
                        for index in range(0, number_of_results_from_current_depth, self.number_of_total_queries):
                            try:
                                organized_tree_nodes = results_from_depth[index:index + self.number_of_total_queries]
                            except:
                                organized_tree_nodes = results_from_depth[index:index + 1]
                            
                            logging.info(f"Length of organized tree nodes is {len(organized_tree_nodes)}")
                            current_parent_name = organized_tree_nodes[0].parent
                            for node in organized_tree_nodes:
                                parent_name = node.parent
                                if parent_name != current_parent_name:
                                    raise ValueError("Currently Tree Nodes have different parents")
                                
                            merged_siblings = compute_RRF(args, organized_tree_nodes)
                        
                            previous_depth_merged_results[iteration].append((current_parent_name, merged_siblings))
    
                ## check before appending the merged result
                if len(previous_depth_merged_results[self.total_iterative_retrieval_loop - 1]) != 1:
                    raise ValueError("Error with Merging Process")
                
                total_retrieved_result.append(previous_depth_merged_results[self.total_iterative_retrieval_loop - 1][0][1])
        
        if self.use_multi_source == True and self.include_original_retrieval == True:
            total_retrieved_result.append(organized_retrieved_top_corpus_per_agent["Original Retrieval"][0][0].retrieved_result)

        logging.info(f"Number of final corpus to merge is {len(total_retrieved_result)}")
        return total_retrieved_result
    
    
    def SubTreeExplore(self, args, query_full_paper, query_abstract, parent_name) -> tuple:
        """
        SubTreeExplore for Multi-Hop document Retrieval with depth-aware reinforced similiarity.
        """        
        if self.only_A2A == False:
            optimized_queries = self.QueryOptimizer.forward_later_rounds(query_full_paper, parent_name)
            formatted_query = []
            if self.include_original_retrieval == True and self.use_multi_source == True:
                formatted_query.append(("Original Retrieval", query_abstract))

            for agent_name, query in optimized_queries:
                formatted_optimized_query = f"{query}"
                formatted_query.append((agent_name, formatted_optimized_query))
                
        elif self.only_A2A == True:
            optimized_queries = self.QueryOptimizer.forward_later_rounds(query_full_paper, parent_name, no_forward=True)
            formatted_query = []
            if self.include_original_retrieval == True and self.use_multi_source == True:
                formatted_query.append((f"{parent_name}_Original Retrieval", query_abstract))

            for agent_name, query in optimized_queries:
                formatted_optimized_query = ""
                formatted_query.append((agent_name, formatted_optimized_query))

        #organized_retrieved_top_corpus_list = []
        organized_retrieved_top_corpus_per_agent = {}
        organized_retrieved_total_corpus_per_agent = {}
        
        for agent_name, current_query in formatted_query:
            if self.only_A2A == False:
                top_k_corpus = self.retrieve_top_k(current_query, args.top_k, agent_name)                
            elif self.only_A2A == True:
                if agent_name == f"{parent_name}_Original Retrieval" and current_query != "":
                    top_k_corpus = self.retrieve_top_k(current_query, args.top_k, "Original Retrieval")
                else:
                    top_k_corpus = []
                    
            #organized_retrieved_top_corpus_list.append(top_k_corpus)
            
            if self.only_A2A == False:
                current_corpus = []
                current_corpus_full_paper = []
                for rank, (corpus, score, original_content) in enumerate(top_k_corpus):
                    current_corpus.append(corpus)
                        
                    if self.use_chunked == True:
                        if agent_name != "Original Retrieval":
                            full_paper_content = self.chunked_corpus_to_full_context_dict[original_content]
                            current_corpus_full_paper.append(full_paper_content)
                    
                if agent_name != "Original Retrieval":        
                    organized_retrieved_top_corpus_per_agent[agent_name] = (current_corpus, current_corpus_full_paper)
                organized_retrieved_total_corpus_per_agent[agent_name] = current_corpus
                
            elif self.only_A2A == True:
                current_corpus = []
                current_corpus_full_paper = []
                for rank, (corpus, score, original_content) in enumerate(top_k_corpus):
                    current_corpus.append(corpus)
                        
                    full_paper_content = self.abstract_to_full_context_dict[original_content]
                    current_corpus_full_paper.append(full_paper_content)
                            
                organized_retrieved_top_corpus_per_agent[agent_name] = (current_corpus, current_corpus_full_paper)
                organized_retrieved_total_corpus_per_agent[agent_name] = current_corpus
        
        if self.only_A2A == True:
            optimized_queries = formatted_query
            
        return organized_retrieved_total_corpus_per_agent, organized_retrieved_top_corpus_per_agent, optimized_queries


    def evaluate(self, args, test_file_directory, result_folder_path):
        """
        Evaluate Retrieval Performance on scientific papers submitted to recent venues, such as ICLR 2024, ICLR 2025.
        You can either use query optimizer or not. 
        """ 
        total_results = []
        for files in test_file_directory:
            with open(files, "r") as json_file:
                evaluation_data = json.load(json_file)

            if self.use_query_optimizer == False:
                formatted_query, formatted_correct_candidates = self.format_query_candidates(evaluation_data)
                    
            elif self.use_query_optimizer == True:
                filename = os.path.basename(files)
                paper_id = filename.removesuffix(".json")
                full_paper_filename = paper_id + ".mmd"

                query_title = evaluation_data["Query"]["title"]
                query_abstract = evaluation_data["Query"]["abstract"]
                original_formatted_query_title_abstract = f"Title: {query_title}\nAbstract: {query_abstract}"
                
                parent_dir = os.path.dirname(args.benchmark_directory)

                root_folder_name = os.path.join(parent_dir, "Full_Paper")
                    
                full_paper_path = os.path.join(root_folder_name, full_paper_filename)

                evaluation_data, optimized_queries = self.QueryOptimizer.forward(full_paper_path, evaluation_data)
                
                formatted_query, formatted_correct_candidates = self.format_query_candidates(evaluation_data)
                            
            if self.use_query_optimizer == False:                   
                if self.iterative_retrieval == False:
                    parsed_retrieved_top_corpus = []
                    agent_name = None
                    
                    if self.use_chunked == True and self.use_full_paper_as_corpus == False:
                        if args.use_RRF_for_chunked_baseline == True:
                            logging.info(f"Currently USING RRF for chunked corpus baseline")
                            top_k_corpus = self.retrieve_top_k(formatted_query, 4 * args.top_k, agent_name)
                            
                            fused_scores = {}
                            for rank, (corpus, score, original_content) in enumerate(top_k_corpus):
                                fused_scores[corpus] = fused_scores.get(corpus, 0) + 1 / (args.hyperparameter_RRF + rank + 1)
                                            
                            sorted_docs = [doc for doc, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
                            parsed_retrieved_top_corpus = sorted_docs[:args.top_k]
                            organized_retrieved_top_corpus = {}
                            
                        elif args.use_RRF_for_chunked_baseline == False:
                            logging.info(f"Currently NOT USING RRF for chunked corpus baseline")
                            top_k_corpus = self.retrieve_top_k(formatted_query, args.top_k, agent_name)
                            organized_retrieved_top_corpus = {}    
                            for idx, (corpus, score, original_content) in enumerate(top_k_corpus):
                                organized_retrieved_top_corpus[f"\nRank {idx + 1}"] = {"Score": f"Score: {score:.4f}", "Content": f"{corpus}"}
                                parsed_retrieved_top_corpus.append(corpus)
                        
                    elif self.use_chunked == False and self.use_full_paper_as_corpus == True:
                        top_k_corpus = self.retrieve_top_k(formatted_query, args.top_k, agent_name)
                        organized_retrieved_top_corpus = {}    
                        for idx, (corpus, score, original_content) in enumerate(top_k_corpus):
                            organized_retrieved_top_corpus[f"\nRank {idx + 1}"] = {"Score": f"Score: {score:.4f}", "Content": f"{corpus}"}
                            parsed_retrieved_top_corpus.append(corpus)
                        
                    elif self.use_chunked == False and self.use_full_paper_as_corpus == False:
                        top_k_corpus = self.retrieve_top_k(formatted_query, args.top_k, agent_name)
                        organized_retrieved_top_corpus = {}
                        for idx, (corpus, score) in enumerate(top_k_corpus):
                            organized_retrieved_top_corpus[f"\nRank {idx + 1}"] = {"Score": f"Score: {score:.4f}", "Content": f"{corpus}"}
                            parsed_retrieved_top_corpus.append(corpus)
                            
                elif self.iterative_retrieval == True:
                    organized_retrieved_top_corpus_list = []
                    previously_chosen_cache = []
                    agent_name = None
                    
                    if self.use_full_paper_as_query == True and self.use_chunked == False and self.use_full_paper_as_corpus == False:
                        raise ValueError("Unsupported Setting when using full paper as query to original abstracts as corpus for iterative retrieval")

                    for iteration in range(0, self.total_iterative_retrieval_loop):
                        logging.info(f"iteration {iteration}")
                        if self.use_chunked == True or self.use_full_paper_as_corpus == True:
                            if self.use_chunked == True and self.use_full_paper_as_corpus == False:
                                if args.use_RRF_for_chunked_baseline == True:
                                    top_k_corpus = self.retrieve_top_k(formatted_query, 4 * args.top_k, agent_name)
                                elif args.use_RRF_for_chunked_baseline == False:
                                    top_k_corpus = self.retrieve_top_k(formatted_query, args.top_k, agent_name)
                                    
                            elif self.use_chunked == False and self.use_full_paper_as_corpus == True:
                                top_k_corpus = self.retrieve_top_k(formatted_query, args.top_k, agent_name)
                                 
                            organized_retrieved_top_corpus_list.append(top_k_corpus)
                            for idx, (corpus, score, original_content) in enumerate(top_k_corpus):
                                if idx < args.selector_starting_idx:
                                    continue
                                
                                if corpus not in previously_chosen_cache:
                                    previously_chosen_cache.append(corpus)
                                    
                                    if self.use_full_paper_as_query == True:
                                        if self.use_full_paper_as_corpus == True and self.use_chunked == False:
                                            formatted_query = original_content
                                        elif self.use_full_paper_as_corpus == False and self.use_chunked == True:
                                            full_paper_content = self.chunked_corpus_to_full_context_dict[original_content]
                                            formatted_query = full_paper_content
                                    
                                    elif self.use_full_paper_as_query == False:
                                        formatted_query = corpus
                                    break
                                
                        elif self.use_chunked == False and self.use_full_paper_as_corpus == False:
                            if self.use_full_paper_as_query == True:
                                raise ValueError("Unsupported Setting when using full paper as query to original abstracts as corpus for iterative retrieval")
                            
                            top_k_corpus = self.retrieve_top_k(formatted_query, args.top_k, agent_name)
                            organized_retrieved_top_corpus_list.append(top_k_corpus)
                            for idx, (corpus, score) in enumerate(top_k_corpus):
                                if idx < args.selector_starting_idx:
                                    continue
                                
                                if corpus not in previously_chosen_cache:
                                    previously_chosen_cache.append(corpus)
                                    formatted_query = corpus
                                    break
                                
                    
                    if self.use_chunked == False and self.use_full_paper_as_corpus == False:
                        ## RRF
                        fused_scores = {}
                        for top_k_corpus in organized_retrieved_top_corpus_list:
                            for rank, (corpus, score) in enumerate(top_k_corpus):
                                fused_scores[corpus] = fused_scores.get(corpus, 0) + 1 / (args.hyperparameter_RRF + rank + 1)
                                        
                        sorted_docs = [doc for doc, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
                        parsed_retrieved_top_corpus = sorted_docs[:args.top_k]
                                    
                    
                    elif self.use_chunked == True or self.use_full_paper_as_corpus == True:
                        ## RRF
                        fused_scores = {}
                        for top_k_corpus in organized_retrieved_top_corpus_list:
                            for rank, (corpus, score, original_content) in enumerate(top_k_corpus):
                                fused_scores[corpus] = fused_scores.get(corpus, 0) + 1 / (args.hyperparameter_RRF + rank + 1)
                                        
                        sorted_docs = [doc for doc, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
                        parsed_retrieved_top_corpus = sorted_docs[:args.top_k]

                    organized_retrieved_top_corpus = []

                                 
            elif self.use_query_optimizer == True and self.SubTreeSearch == False and self.iterative_retrieval == False:
                organized_retrieved_top_corpus = {}
                organized_retrieved_top_corpus_list = []
                for agent_name, current_query in formatted_query:
                    organized_retrieved_top_corpus[agent_name] = {}
                    top_k_corpus = self.retrieve_top_k(current_query, args.top_k, agent_name)
                    organized_retrieved_top_corpus_list.append(top_k_corpus)
                        
                    if self.use_multi_source == True:
                        for idx, (corpus, score, original_content) in enumerate(top_k_corpus):
                            organized_retrieved_top_corpus[agent_name][f"\nRank {idx + 1}"] = {"Score": f"Score: {score:.4f}", "Content": f"{corpus}"}
                                    
                    elif self.use_multi_source == False:
                        if self.use_chunked == False and self.use_full_paper_as_corpus == False:
                            for idx, (corpus, score) in enumerate(top_k_corpus):
                                organized_retrieved_top_corpus[agent_name][f"\nRank {idx + 1}"] = {"Score": f"Score: {score:.4f}", "Content": f"{corpus}"}
                        elif self.use_chunked == True or self.use_full_paper_as_corpus == True:
                            for idx, (corpus, score, original_content) in enumerate(top_k_corpus):
                                organized_retrieved_top_corpus[agent_name][f"\nRank {idx + 1}"] = {"Score": f"Score: {score:.4f}", "Content": f"{corpus}"}
                        
                if self.use_multi_source == False:
                    if self.use_chunked == False and self.use_full_paper_as_corpus == False:
                            ### Fuse rankings using RRF
                            fused_scores = {}
                            for top_k_corpus in organized_retrieved_top_corpus_list:
                                for rank, (corpus, score) in enumerate(top_k_corpus):
                                    fused_scores[corpus] = fused_scores.get(corpus, 0) + 1 / (args.hyperparameter_RRF + rank + 1)
                                
                            sorted_docs = [doc for doc, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
                            parsed_retrieved_top_corpus = sorted_docs[:args.top_k]
                            
                    elif self.use_chunked == True or self.use_full_paper_as_corpus == True:
                            ### Fuse rankings using RRF
                            fused_scores = {}
                            for top_k_corpus in organized_retrieved_top_corpus_list:
                                for rank, (corpus, score, original_content) in enumerate(top_k_corpus):
                                    fused_scores[corpus] = fused_scores.get(corpus, 0) + 1 / (args.hyperparameter_RRF + rank + 1)
                                
                            sorted_docs = [doc for doc, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
                            parsed_retrieved_top_corpus = sorted_docs[:args.top_k]
                        
                elif self.use_multi_source == True:
                    ### Fuse rankings using RRF
                    fused_scores = {}
                    for top_k_corpus in organized_retrieved_top_corpus_list:
                        for rank, (corpus, score, original_content) in enumerate(top_k_corpus):
                            fused_scores[corpus] = fused_scores.get(corpus, 0) + 1 / (args.hyperparameter_RRF + rank + 1)
                            
                    sorted_docs = [doc for doc, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
                    parsed_retrieved_top_corpus = sorted_docs[:args.top_k]
                    ############################
            
            elif self.use_query_optimizer == True and self.SubTreeSearch == True and self.iterative_retrieval == True:
                    organized_retrieved_top_corpus_list = []
                    organized_retrieved_top_corpus = {}
                    organized_retrieved_top_corpus_per_agent = {}
                    
                    previous_organized_retrieved_top_corpus_per_agent_cache = {}

                    for agent, current_query in formatted_query:
                        organized_retrieved_top_corpus_per_agent[agent] = []
                        for iteration in range(0, self.total_iterative_retrieval_loop):
                            organized_retrieved_top_corpus_per_agent[agent].append([])
                                
                        previous_organized_retrieved_top_corpus_per_agent_cache[agent] = []
                    
                    organized_retrieved_top_corpus_for_refinement = {}
                    formatted_query_dictionary = {}
                    for iteration in range(0, self.total_iterative_retrieval_loop):  
                        organized_retrieved_top_corpus_per_iteration_list = []
                        for agent_name, current_query in formatted_query:
                            if iteration == 0:
                                formatted_query_dictionary[agent_name] = current_query
                                top_k_corpus = self.retrieve_top_k(current_query, args.top_k, agent_name)
                                organized_retrieved_top_corpus_per_iteration_list.append((top_k_corpus, iteration))
                            
                                top_k_corpus_for_refinement = []
                                top_k_corpus_for_refinement_abstract_list = []
                                top_k_corpus_for_refinement_full_paper_list = []
                                formatted_top_k_corpus = []
                                for idx, (corpus, score, original_content) in enumerate(top_k_corpus):
                                    top_k_corpus_for_refinement.append(original_content)
                                    formatted_top_k_corpus.append(corpus)
                                    if self.use_chunked == True:
                                        if agent_name != "Original Retrieval":
                                            full_paper_content = self.chunked_corpus_to_full_context_dict[original_content]
                                            top_k_corpus_for_refinement_full_paper_list.append(full_paper_content)
                                        
                                    top_k_corpus_for_refinement_abstract_list.append(corpus)

                                tree_node = Tree_Node(current_node_name=agent_name, parent_node_name=None, retrieved_result=formatted_top_k_corpus)
                                organized_retrieved_top_corpus_per_agent[agent_name][0].append(tree_node)
                                
                                if agent_name != "Original Retrieval":
                                    if args.use_aspect_aware_cache_for_selection == True:
                                        appropriate_corpus, appropriate_full_paper_corpus = self.Selector._find_nonoverlapping_corpus(top_k_corpus_for_refinement_abstract_list, top_k_corpus_for_refinement_full_paper_list, previous_organized_retrieved_top_corpus_per_agent_cache, agent_name, args.input_top_k_for_verifier)
                                    elif args.use_aspect_aware_cache_for_selection == False:
                                        logging.info("Not using Aspect Aware-Cache for Next Query Selection")
                                        appropriate_corpus, appropriate_full_paper_corpus = self.Selector._find_corpus_nearest(top_k_corpus_for_refinement_abstract_list, top_k_corpus_for_refinement_full_paper_list, previous_organized_retrieved_top_corpus_per_agent_cache, agent_name, args.input_top_k_for_verifier)
                                        
                                    for selected_corpus in appropriate_corpus:
                                        previous_organized_retrieved_top_corpus_per_agent_cache[agent_name].append(selected_corpus)
                                    organized_retrieved_top_corpus_for_refinement[agent_name] = (appropriate_corpus, appropriate_full_paper_corpus)
                        
                                                
                        if iteration < self.total_iterative_retrieval_loop - 1:
                            logging.info(f"Currently Refining Query for {iteration + 1} Time.")
                            if (iteration + 1 >= args.starting_iteration_of_A2A_only - 1) and args.starting_iteration_of_A2A_only > 0:
                                self.only_A2A = True
                            
                            ### Verifier Select Appropriate candidates well aligned with Each aspect
                            logging.info(f"{list(organized_retrieved_top_corpus_for_refinement.keys())}")
                            filtered_candidates_dict = self.Selector.forward_for_multirounds(organized_retrieved_top_corpus_for_refinement=organized_retrieved_top_corpus_for_refinement)
                                
                            newly_formatted_query = []
                            for parent_agent, current_query in formatted_query:
                                if self.only_A2A == False:
                                    if parent_agent != "Original Retrieval":
                                        abstract, full_paper = filtered_candidates_dict[parent_agent]
                                        retrieved_total_corpus_per_agent, retrieved_corpus_per_agent, optimized_queries = self.SubTreeExplore(args, full_paper, abstract, parent_agent)

                                        retrieved_corpus_per_agent_list = list(retrieved_corpus_per_agent.keys())

                                        retrieved_total_corpus_per_agent_list = list(retrieved_total_corpus_per_agent.keys())
                                            
                                        logging.info(f"Number of retrieved total corpus per agent list is {len(retrieved_total_corpus_per_agent)}")
                                        for agent in retrieved_total_corpus_per_agent_list:
                                            retrieved_corpus = retrieved_total_corpus_per_agent[agent]
                                                    
                                            tree_node = Tree_Node(current_node_name=agent, parent_node_name=parent_agent, retrieved_result=retrieved_corpus)
                                                    
                                            parent_list = parent_agent.split("_")
                                            root_parent = parent_list[0]
                                            logging.info(f"Root Parent Agent is {root_parent}")
                                            organized_retrieved_top_corpus_per_agent[root_parent][iteration + 1].append(tree_node)
                                            
                                        for agent in retrieved_corpus_per_agent_list:
                                            if self.only_A2A == False:
                                                if agent != "Original Retrieval":
                                                    parent_list = agent.split("_")
                                                    root_parent = parent_list[0]
                                                    
                                                    if args.use_aspect_aware_cache_for_selection == True:
                                                        appropriate_corpus, appropriate_full_paper_corpus = self.Selector._find_nonoverlapping_corpus(retrieved_corpus_per_agent[agent][0], retrieved_corpus_per_agent[agent][1], previous_organized_retrieved_top_corpus_per_agent_cache, root_parent, args.input_top_k_for_verifier)
                                                    elif args.use_aspect_aware_cache_for_selection == False:
                                                        logging.info("Not using Aspect Aware-Cache for Next Query Selection")
                                                        appropriate_corpus, appropriate_full_paper_corpus = self.Selector._find_corpus_nearest(retrieved_corpus_per_agent[agent][0], retrieved_corpus_per_agent[agent][1], previous_organized_retrieved_top_corpus_per_agent_cache, root_parent, args.input_top_k_for_verifier)
                                                    
                                                    for selected_corpus in appropriate_corpus:
                                                        previous_organized_retrieved_top_corpus_per_agent_cache[root_parent].append(selected_corpus)
                                                    organized_retrieved_top_corpus_for_refinement[agent] = (appropriate_corpus, appropriate_full_paper_corpus)
                                                
                                            
                                        newly_formatted_query.extend(optimized_queries)
                                            
                                else:
                                    if parent_agent != "Original Retrieval":
                                        abstract, full_paper = filtered_candidates_dict[parent_agent]
                                        retrieved_total_corpus_per_agent, retrieved_corpus_per_agent, optimized_queries = self.SubTreeExplore(args, full_paper, abstract, parent_agent)

                                        retrieved_corpus_per_agent_list = list(retrieved_corpus_per_agent.keys())

                                        retrieved_total_corpus_per_agent_list = list(retrieved_total_corpus_per_agent.keys())
                                                
                                        logging.info(f"Number of retrieved total corpus per agent list is {len(retrieved_total_corpus_per_agent)}")
                                        for agent in retrieved_total_corpus_per_agent_list:
                                            retrieved_corpus = retrieved_total_corpus_per_agent[agent]
                                                        
                                            tree_node = Tree_Node(current_node_name=agent, parent_node_name=parent_agent, retrieved_result=retrieved_corpus)
                                                        
                                            parent_list = parent_agent.split("_")
                                            root_parent = parent_list[0]
                                            logging.info(f"Root Parent Agent is {root_parent}")
                                            organized_retrieved_top_corpus_per_agent[root_parent][iteration + 1].append(tree_node)
                                                
                                        for agent in retrieved_corpus_per_agent_list:
                                            parent_list = agent.split("_")
                                            root_parent = parent_list[0]
                                            
                                            if args.use_aspect_aware_cache_for_selection == True:
                                                appropriate_corpus, appropriate_full_paper_corpus = self.Selector._find_nonoverlapping_corpus(retrieved_corpus_per_agent[agent][0], retrieved_corpus_per_agent[agent][1], previous_organized_retrieved_top_corpus_per_agent_cache, root_parent, args.input_top_k_for_verifier)
                                            elif args.use_aspect_aware_cache_for_selection == False:
                                                logging.info("Not using Aspect Aware-Cache for Next Query Selection")
                                                appropriate_corpus, appropriate_full_paper_corpus = self.Selector._find_corpus_nearest(retrieved_corpus_per_agent[agent][0], retrieved_corpus_per_agent[agent][1], previous_organized_retrieved_top_corpus_per_agent_cache, root_parent, args.input_top_k_for_verifier)
                                                
                                            for selected_corpus in appropriate_corpus:
                                                    previous_organized_retrieved_top_corpus_per_agent_cache[root_parent].append(selected_corpus)
                                                
                                            organized_retrieved_top_corpus_for_refinement[agent] = (appropriate_corpus, appropriate_full_paper_corpus)
                                                             
                                        newly_formatted_query.extend(optimized_queries)
                                        
                            formatted_query = newly_formatted_query
                            logging.info(f"Number of newly formatted queries are {len(formatted_query)}")
                                                            
                    if args.Recursive_Merge == True:
                        organized_retrieved_top_corpus_list = self.Recursive_Merge_Results(args, organized_retrieved_top_corpus_per_agent)
                    elif args.Recursive_Merge == False:
                        organized_retrieved_top_corpus_list = self.Total_Merge_Results(args, organized_retrieved_top_corpus_per_agent)

                    organized_retrieved_top_corpus = []
                    
                    fused_scores = {}
                    for top_k_corpus in organized_retrieved_top_corpus_list:
                        for rank, corpus in enumerate(top_k_corpus):
                            try:
                                fused_scores[corpus] = fused_scores.get(corpus, 0) + 1 / (args.hyperparameter_RRF + rank + 1)
                            except:
                                print(corpus)
                        
                    sorted_docs = [doc for doc, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
                    parsed_retrieved_top_corpus = sorted_docs[:args.top_k]
 

            current_result = evaluate_retrieval(parsed_retrieved_top_corpus, formatted_correct_candidates, args.top_k, args.max_top_k)
            total_results.append(current_result)
                
            if self.use_query_optimizer == False:
                organized_results = {"query": {"id": f"{evaluation_data['id']}", "content": f"{formatted_query}"}, 
                                     "Retrieved_Candidates": organized_retrieved_top_corpus, "Final_Ranked_Results": parsed_retrieved_top_corpus, "Correct_Candidates": formatted_correct_candidates, "Current Result": current_result}
            
            elif self.use_query_optimizer == True:
                queries = []
                for current_query in formatted_query:
                    query = {'agent': current_query[0], "content": current_query[1]}
                    queries.append(query)
                        
                organized_results = {"query": {"id": f"{evaluation_data['id']}", "content": queries}, 
                                     "Retrieved_Candidates": organized_retrieved_top_corpus, "Final_Ranked_Results": parsed_retrieved_top_corpus, "Correct_Candidates": formatted_correct_candidates, "Current Result": current_result}
                    
            current_result_folder_path = f"{result_folder_path}/{evaluation_data['id']}.json"
            with open(current_result_folder_path, "w") as json_file:
                json.dump(organized_results, json_file, indent=4)