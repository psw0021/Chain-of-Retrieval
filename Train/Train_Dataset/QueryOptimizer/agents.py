import json
import yaml
import os
import sys
from pydantic import BaseModel
import re
from vllm import SamplingParams
import torch
from transformers import AutoTokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
# Add the root directory to Python's module search path
sys.path.append(root_dir)

class Query(BaseModel):
    Aspect: str
    Query: str

class Query_lists(BaseModel):
    list[Query]

class BaseQueryOptimizerAgent:
    """Agent that gets receives full paper of scientific article and generates optimized query for retrieving relevant works
    This optimizer is a general optimizer, that is not tasked with focusing on diverse tasks. """

    def __init__(self, args, vllm_model) -> None:
        self.args = args
        self.model_name = args.query_optimizer_model
        self.openai_models = ["gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06", "o3-mini-2025-01-31"]
        self.use_gpt = args.use_gpt
        self.prompt_path = ""
        self.name = "BASE"
        self.repetition_penalty = args.repetition_penalty
        self.max_tokens = args.max_tokens
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.vllm_model = vllm_model
        self.models = ["meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]
        self.rollout_number = args.rollout_number
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.use_gpt == True:
            assert self.model_name in self.openai_models, f"Error: {self.model_name} is not a valid model name!"
        elif self.use_gpt == False:
            assert self.model_name in self.models, f"Error: {self.model_name} is not a valid model name!"

    def parse_aspect_query(self, text):
        pattern = re.compile(r"##\s*Aspect:\s*(.*?)\n##\s*Query[:;]\s*(.*?)\n", re.DOTALL)
        return pattern.findall(text)
        
    def open_prompt(self):
        with open(self.prompt_path, "r") as file:
            data = yaml.safe_load(file)
            
        user_prompt = data["user_prompt"]
        system_prompt = data["system_prompt"]
        
        return system_prompt, user_prompt 
    
    def call_vllm(self, content):
        system_prompt, user_prompt = self.open_prompt()
        user_prompt = user_prompt.format(paper=content)
        
        messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
        ]
        
        formatted_message = self.tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True)
        
        sampling_parameters = SamplingParams(temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens, n=self.rollout_number, repetition_penalty=self.repetition_penalty)
        response = self.vllm_model.generate(formatted_message, sampling_parameters)
        final_response = []
        for answer in response[0].outputs:
            final_response.append(answer.text)
        
        torch.cuda.empty_cache()
        
        return final_response

    def forward(self, mmd_file_path) -> str:
        with open(mmd_file_path, 'r') as f:
            paper = f.read()
        if self.use_gpt == True:
            optimized_query = self.call_openai(content=paper)
        elif self.use_gpt == False:
            optimized_query = self.call_vllm(content=paper)
        
        return optimized_query
    

class MethodFocusedQueryOptimizerAgent(BaseQueryOptimizerAgent):
    """Agent that receives full paper of scientific paper and generates refined query focused on method of given scientific paper"""
    
    def __init__(self, args, vllm_model):
        super().__init__(args, vllm_model)
        self.prompt_path = f"Train_Dataset/QueryOptimizer/Prompts/method_focused_query_optimizer_agent_prompt_{args.query_detailedness}.yaml"
        self.name = "METHOD FOCUSED AGENT"
        self.vllm_model = vllm_model
        
    
class ExperimentFocusedQueryOptimizerAgent(BaseQueryOptimizerAgent):
    """Agent that receives full paper of scientific paper and generates refined query focused on experiments of given scientific paper"""
    
    def __init__(self, args, vllm_model):
        super().__init__(args, vllm_model)
        self.prompt_path = f"Train_Dataset/QueryOptimizer/Prompts/experiment_focused_query_optimizer_agent_prompt_{args.query_detailedness}.yaml"
        self.name = "EXPERIMENT Focused AGENT"
        self.vllm_model = vllm_model


class ResearchQuestionFocusedQueryOptimizerAgent(BaseQueryOptimizerAgent):
    """Agent that receives full paper of scientific paper and generates refined query focused on research question of given scientific paper"""
    
    def __init__(self, args, vllm_model):
        super().__init__(args, vllm_model)
        self.prompt_path = f"Train_Dataset/QueryOptimizer/Prompts/research_question_focused_query_optimizer_agent_prompt_{args.query_detailedness}.yaml"
        self.name = "RESEARCH QUESTION FOCUSED AGENT"
        self.vllm_model = vllm_model
    

class QueryOptimizer:
    """
    Controller agent that handles multi agent, single agent query optimization process
    """
    def __init__(self, args, result_folder_path, vllm_model):
        self.args = args
        self.agents = []
        prompts_to_save = {}
        self.agent1 = MethodFocusedQueryOptimizerAgent(args, vllm_model)
        self.agent2 = ExperimentFocusedQueryOptimizerAgent(args, vllm_model)
        self.agent3 = ResearchQuestionFocusedQueryOptimizerAgent(args, vllm_model)

        if args.use_method_agent == True:
            self.agents.append(self.agent1)
            system_prompt, user_prompt = self.agent1.open_prompt()
            prompts_to_save["METHOD"] = system_prompt
        if args.use_experiment_agent == True:
            self.agents.append(self.agent2)
            system_prompt, user_prompt = self.agent2.open_prompt()
            prompts_to_save["EXPERIMENT"] = system_prompt
        if args.use_research_question_agent == True:
            self.agents.append(self.agent3)
            system_prompt, user_prompt = self.agent3.open_prompt()
            prompts_to_save["RESEARCH QUESTION"] = system_prompt
            
        prompt_file = os.path.join(result_folder_path, "prompts.json")
        with open(prompt_file, "w") as json_file:
            json.dump(prompts_to_save, json_file, indent=4)
    
    def forward(self, paper_path, evaluation_data) -> str:
        optimized_queries = []
        for agent in self.agents:
            optimized_query = agent.forward(paper_path)
            agent_name = agent.name
            optimized_queries.append((agent_name, optimized_query))
            
        evaluation_data["Query"]["optimized_queries"] = optimized_queries
        
        return evaluation_data, optimized_queries
            
        

        
    