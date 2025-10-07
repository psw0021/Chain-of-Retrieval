import json
import yaml
import openai
from openai import OpenAI
import os
import sys
from pydantic import BaseModel
import re
from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer
import ast
from typing import Tuple, Any
import logging
import tiktoken

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
# Add the root directory to Python's module search path
sys.path.append(root_dir)



class BaseQueryOptimizerAgent:
    """Agent that gets receives full paper of scientific article and generates optimized query for retrieving relevant works
    This optimizer is a general optimizer, that is not tasked with focusing on diverse tasks. """

    def __init__(self, args, vllm_model_dict) -> None:
        self.args = args
        self.model_name = args.query_optimizer_model
        
        self.openai_models = ["gpt-4o-mini-2024-07-18", "gpt-4o-2024-11-20", "gpt-4.1-2025-04-14"]
        self.use_gpt = args.use_gpt
        self.prompt_path = ""
        self.name = "BASE"
        
        if self.use_gpt == False:
            self.vllm_model = vllm_model_dict["METHOD"]["agent"]
            self.vllm_model_device = vllm_model_dict["METHOD"]["device"]

        self.models = ["meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]
        self.multi_agent = args.multi_agent
        self.temperature = args.temperature
        self.max_tokens = args.max_tokens
        self.repetition_penalty = args.repetition_penalty
        
        if self.use_gpt == False:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.eos_token = self.tokenizer.pad_token
        
        if self.use_gpt == True:
            assert self.model_name in self.openai_models, f"Error: {self.model_name} is not a valid model name!"
            if self.model_name != "gpt-4.1-2025-04-14":
                self.context_window = 120000
        elif self.use_gpt == False:
            assert self.model_name in self.models, f"Error: {self.model_name} is not a valid model name!"
            self.context_window = 131920
            
    def truncate_input_for_gpt(self, system_prompt, paper):
        def clean_text(text):
            """Remove special tokens and non-printable characters."""
            text = re.sub(r"[^\x20-\x7E]", "", text)  # Keep only printable characters
            text = text.strip()  # Remove leading/trailing spaces
            return text
        
        def truncate(paper):
            encoding = tiktoken.encoding_for_model(self.model_name)
            
            tokens = encoding.encode(clean_text(paper), disallowed_special=())[:self.context_window]
            truncated_text = encoding.decode(tokens)
            
            final_tokens = encoding.encode(truncated_text)
            logging.info(f"Length of Tokens is {len(final_tokens)}")

            return truncated_text
        
        return truncate(paper)

        
    def open_prompt(self, refinement=False) -> Tuple[str, str]:
        """
        Open prompt for initial optimization
        """
        if refinement == False:
            with open(self.prompt_path, "r") as file:
                data = yaml.safe_load(file)
            
        user_prompt = data["user_prompt"]
        system_prompt = data["system_prompt"]
        
        return system_prompt, user_prompt 


    def call_openai(self, content: str) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise EnvironmentError("VARIABLE_NAME is not set")
        
        system_prompt, user_prompt = self.open_prompt()
        
        if self.model_name != "gpt-4.1-2025-04-14":
            content = self.truncate_input_for_gpt(system_prompt, content)
        else:
            content = content
        
        user_prompt = user_prompt.format(paper=content)
        
        messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
        ]
        
        client = OpenAI()
        
        if self.model_name != "gpt-4.1-2025-04-14":
            response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
            )
            final_response = response.choices[0].message.content

        else:
            response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
            )
            final_response = response.choices[0].message.content
        
        return final_response
    
    
    def call_vllm(self, content: str) -> str:
        """
        Function to use vllm LLM function to generate queries for open-source models
        """
        system_prompt, user_prompt = self.open_prompt()
        user_prompt = user_prompt.format(paper=content)
        
        messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
        ]

        formatted_message = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        sampling_parameters = SamplingParams(temperature=self.temperature, max_tokens=self.max_tokens, repetition_penalty=self.repetition_penalty)

        torch.cuda.set_device(self.vllm_model_device)
        print(self.vllm_model_device)
        
        response = self.vllm_model.generate(formatted_message, sampling_parameters)
        final_response = response[0].outputs[0].text
        
        torch.cuda.empty_cache()
        
        return final_response
    
    
    def forward(self, mmd_file_path: str) -> str:
        """
        Forward generation of output using agents.
        """
        with open(mmd_file_path, 'r') as f:
            paper = f.read()

        if self.use_gpt == True:
            optimized_query = self.call_openai(content=paper)
        elif self.use_gpt == False:
            optimized_query = self.call_vllm(content=paper)
        
        return optimized_query
    

    def forward_for_later_rounds(self, mmd_file_path: str, full_paper : str, paper_opened=False) -> str:
        """
        Forward generation queries for iterative aspect aware chain of retrieval.
        """
        if paper_opened == False:
            with open(mmd_file_path, 'r') as f:
                paper = f.read()
        elif paper_opened == True:
            paper = full_paper

        if self.use_gpt == True:
            optimized_query = self.call_openai(content=paper)
        elif self.use_gpt == False:
            optimized_query = self.call_vllm(content=paper)
        
        return optimized_query
    

class MethodFocusedQueryOptimizerAgent(BaseQueryOptimizerAgent):
    """Agent that receives full patent and generates refined query focused on method of given patent"""
    
    def __init__(self, args, vllm_model_dict):
        super().__init__(args, vllm_model_dict)
        self.prompt_path = "Agents/Prompts/Patents/QueryOptimizer/method_focused_query_optimizer_agents.yaml"
        self.name = "METHOD AGENT"
        if self.use_gpt == False:
            self.vllm_model = vllm_model_dict["METHOD"]["agent"]
            self.vllm_model_device = vllm_model_dict["METHOD"]["device"]
        
    
class ClaimFocusedQueryOptimizerAgent(BaseQueryOptimizerAgent):
    """Agent that receives full patent and generates refined query focused on claims of given patent"""
    
    def __init__(self, args, vllm_model_dict):
        super().__init__(args, vllm_model_dict)
        self.prompt_path = "Agents/Prompts/Patents/QueryOptimizer/claim_focused_query_optimizer_agents.yaml"
        self.name = "CLAIM AGENT"
        if self.use_gpt == False:
            self.vllm_model = vllm_model_dict["CLAIM"]["agent"]
            self.vllm_model_device = vllm_model_dict["CLAIM"]["device"]


class BackgroundFocusedQueryOptimizerAgent(BaseQueryOptimizerAgent):
    """Agent that receives full patent and generates refined query focused on background of given patent"""
    
    def __init__(self, args, vllm_model_dict):
        super().__init__(args, vllm_model_dict)
        self.prompt_path = f"Agents/Prompts/Patents/QueryOptimizer/background_focused_query_optimizer_agent.yaml"
        self.name = "BACKGROUND AGENT"
        if self.use_gpt == False:
            self.vllm_model = vllm_model_dict["BACKGROUND"]["agent"]
            self.vllm_model_device = vllm_model_dict["BACKGROUND"]["device"]
    

class QueryOptimizer:
    """
    Controller that handles multi agent, single agent query optimization process
    """
    def __init__(self, args, result_folder_path, vllm_model_dict):
        self.args = args
        self.multi_agent = args.multi_agent
        
        self.agents = []
        prompts_to_save = {}
        if self.multi_agent == True:
            self.agent1 = MethodFocusedQueryOptimizerAgent(args, vllm_model_dict)
            self.agent2 = ClaimFocusedQueryOptimizerAgent(args, vllm_model_dict)
            self.agent3 = BackgroundFocusedQueryOptimizerAgent(args, vllm_model_dict)
            
            if args.use_method_agent == True:
                self.agents.append(self.agent1)
                system_prompt, user_prompt = self.agent1.open_prompt()
                prompts_to_save["METHOD"] = system_prompt
                
            if args.use_claim_agent == True:
                self.agents.append(self.agent2)
                system_prompt, user_prompt = self.agent2.open_prompt()
                prompts_to_save["CLAIM"] = system_prompt
                
            if args.use_background_agent == True:
                self.agents.append(self.agent3)
                system_prompt, user_prompt = self.agent3.open_prompt()
                prompts_to_save["BACKGROUND"] = system_prompt
                
        prompt_file = os.path.join(result_folder_path, "prompts.json")
        with open(prompt_file, "w") as json_file:
            json.dump(prompts_to_save, json_file, indent=4)
    
    
    def forward(self, paper_path, evaluation_data) -> Tuple[dict, list]:
        """
        Forward multi agent query optimizer
        """
        optimized_queries = []
        for agent in self.agents:
            optimized_query = agent.forward(paper_path)
            agent_name = agent.name
            optimized_queries.append((agent_name, optimized_query))
                
        evaluation_data["Query"]["optimized_queries"] = optimized_queries
        
        return evaluation_data, optimized_queries
    
    
    def forward_later_rounds(self, full_paper, parent_name, no_forward=False) -> Tuple[dict, list]:
        """
        Forward multi agent query optimizer for SUBAGENTS
        """
        logging.info(f"Forwarding Further Exploration for later rounds")
        optimized_queries = []
        for agent in self.agents:
            paper_path = ""
            if no_forward == False:
                optimized_query = agent.forward_for_later_rounds(paper_path, full_paper, paper_opened=True)
                
                agent_name = f"{parent_name}_{agent.name}"
                optimized_queries.append((agent_name, optimized_query))
                    
            elif no_forward == True:
                optimized_query = ""
                agent_name = f"{parent_name}_{agent.name}"
                    
                optimized_queries.append((agent_name, optimized_query))
                
        return optimized_queries
    

    
            
        

        
    