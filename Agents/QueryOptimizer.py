import json
import yaml
import openai
from openai import OpenAI
import os
import sys
from pydantic import BaseModel
import re
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import torch
from transformers import AutoTokenizer
import ast
from typing import Tuple, Any
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
# Add the root directory to Python's module search path
sys.path.append(root_dir)


class Query(BaseModel):
    method_query: str
    experiment_query: str
    research_question_query: str


class BaseQueryOptimizerAgent:
    """Agent that gets receives full paper of scientific article and generates optimized query for retrieving relevant works
    This optimizer is a general optimizer, that is not tasked with focusing on diverse tasks. """

    def __init__(self, args, vllm_model_dict) -> None:
        self.args = args
        self.model_name = args.query_optimizer_model
        self.use_base_agent = args.use_base_agent
        self.openai_models = ["gpt-4o-mini-2024-07-18", "gpt-4o-2024-11-20", "o3-mini-2025-01-31", "o4-mini-2025-04-16", "gpt-5-nano-2025-08-07", "gpt-5-2025-08-07", "gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14"]
        self.use_gpt = args.use_gpt
        self.prompt_path = "Agents/Prompts/ScientificPapers/QueryOptimizer/base_query_optimizer_agent_prompt.yaml"
        self.coverage_critique_prompt_path = ""
        self.name = "BASE"
        #self.vllm_model = vllm_model_dict["BASE"]
        if self.use_gpt == False:
            self.vllm_model = vllm_model_dict["EXPERIMENT"]["agent"]
            self.vllm_model_device = vllm_model_dict["RESEARCH_QUESTION"]["device"]

        self.models = ["meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]
        self.multi_agent = args.multi_agent
        self.use_trained_model = args.use_trained_model
        self.port_number = None
        self.temperature = args.temperature
        self.max_tokens = args.max_tokens
        self.repetition_penalty = args.repetition_penalty
        
        if self.use_gpt == False:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.eos_token = self.tokenizer.pad_token
        
        if self.use_gpt == True:
            assert self.model_name in self.openai_models, f"Error: {self.model_name} is not a valid model name!"
        elif self.use_gpt == False:
            assert self.model_name in self.models, f"Error: {self.model_name} is not a valid model name!"

        
    def open_prompt(self, refinement=False) -> Tuple[str, str]:
        """
        Open prompt for initial optimization and subsequent refinement
        """
        if refinement == False:
            with open(self.prompt_path, "r") as file:
                data = yaml.safe_load(file)
                
        elif refinement == True:
            with open(self.refinement_prompt_path , "r") as file:
                data = yaml.safe_load(file)
            
        user_prompt = data["user_prompt"]
        system_prompt = data["system_prompt"]
        
        return system_prompt, user_prompt 


    def call_openai(self, content: str) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise EnvironmentError("VARIABLE_NAME is not set")
        
        system_prompt, user_prompt = self.open_prompt()
        user_prompt = user_prompt.format(paper=content)
        
        messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
        ]
        
        client = OpenAI()
        
        if self.model_name != "o3-mini-2025-01-31" and self.model_name != "o4-mini-2025-04-16" and self.model_name != "gpt-5-nano-2025-08-07" and self.model_name != "gpt-5-2025-08-07" and self.model_name != "gpt-4.1-2025-04-14" and self.model_name != "gpt-4.1-mini-2025-04-14":
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
    
    
    def call_vllm_api(self, content: str) -> str:
        """
        Function to call vllm API to generate queries for open-source trained query optimizer models        
        """
        openai_api_key = "EMPTY"
        openai_api_base = f"http://localhost:{self.port_number}/v1"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        system_prompt, user_prompt = self.open_prompt()
        user_prompt = user_prompt.format(paper=content)
        
        messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
        ]
        
        ## add repetition penalty
        try:
            response = client.chat.completions.create(model=self.vllm_model, messages=messages,
                                                temperature=self.temperature, max_tokens = self.max_tokens,
                                                  extra_body={"repetition_penalty": self.repetition_penalty})
        except openai.APITimeoutError:
            response = client.chat.completions.create(model=self.vllm_model, messages=messages,
                                                  temperature=self.temperature, max_tokens = self.max_tokens,
                                                  extra_body={"repetition_penalty": self.repetition_penalty, "max_tokens": self.max_tokens}, timeout=600)
        
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
    
    
    def call_openai_structured(self, content: str) -> list:
        """
        Call openai structured to generate multiple queries at once, generating a structured output.
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise EnvironmentError("VARIABLE_NAME is not set")
        
        system_prompt, user_prompt = self.open_prompt()
        user_prompt = user_prompt.format(paper=content)
        
        messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
        ]
        
        client = OpenAI()
        
        if self.model_name != "o3-mini-2025-01-31":
            response = client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    response_format=Query
            )
            completion = response.choices[0].message.parsed

        else:
            response = client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=messages,
                    response_format=Query
            )
            completion = response.choices[0].message.parsed
            

        final_response = []
        final_response.append(completion.method_query)
        final_response.append(completion.experiment_query)
        final_response.append(completion.research_question_query)
        
        return final_response
    
    
    def call_vllm_structured(self, content: str) -> list:
        """
        Call vllm structured to generate multiple queries at once, generating a structured output.
        """
        def clean_str(s):
            s = s.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
            return s
        system_prompt, user_prompt = self.open_prompt()
        user_prompt = user_prompt.format(paper=content)
        
        messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
        ]

        json_schema = Query.model_json_schema()
        formatted_message = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        guided_decoding_params = GuidedDecodingParams(json=json_schema)
        sampling_parameters = SamplingParams(temperature=self.temperature, max_tokens=self.max_tokens, repetition_penalty=self.repetition_penalty, guided_decoding=guided_decoding_params)

        torch.cuda.set_device(self.vllm_model_device)
        
        response = self.vllm_model.generate(formatted_message, sampling_parameters)
        completion = response[0].outputs[0].text
        
        torch.cuda.empty_cache()
        
        try:
            completion_json = ast.literal_eval(clean_str(completion))
            final_response = []
            final_response.append(clean_str(completion_json["method_query"]))
            final_response.append(clean_str(completion_json["experiment_query"]))
            final_response.append(clean_str(completion_json["research_question_query"]))

        except:
            pattern = r'"(method_query|experiment_query|research_question_query)":\s*["\']([^"\']+)'

            result = {
                "method_query": "",
                "experiment_query": "",
                "research_question_query": ""
            }

            matches = re.findall(pattern, completion)
            for key, value in matches:
                result[key] = value

            final_response = []
            final_response.append(clean_str(result["method_query"]))
            final_response.append(clean_str(result["experiment_query"]))
            final_response.append(clean_str(result["research_question_query"]))

        return final_response
    
    
    def forward(self, mmd_file_path: str) -> str:
        """
        Forward generation of output using agents.
        """
        with open(mmd_file_path, 'r') as f:
            paper = f.read()

        if self.use_base_agent == False:
            if self.use_gpt == True:
                optimized_query = self.call_openai(content=paper)
            elif self.use_gpt == False and self.use_trained_model == False:
                optimized_query = self.call_vllm(content=paper)
            elif self.use_gpt == False and self.use_trained_model == True:
                optimized_query = self.call_vllm_api(content=paper)

        elif self.use_base_agent == True:
            if self.use_gpt == True:
                optimized_query = self.call_openai_structured(content=paper)
            elif self.use_gpt == False and self.use_trained_model == False:
                optimized_query = self.call_vllm_structured(content=paper)
        
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

        if self.use_base_agent == False:
            if self.use_gpt == True:
                optimized_query = self.call_openai(content=paper)
            elif self.use_gpt == False and self.use_trained_model == False:
                optimized_query = self.call_vllm(content=paper)
            elif self.use_gpt == False and self.use_trained_model == True:
                optimized_query = self.call_vllm_api(content=paper)

        elif self.use_base_agent == True:
            if self.use_gpt == True:
                optimized_query = self.call_openai_structured(content=paper)
            elif self.use_gpt == False and self.use_trained_model == False:
                optimized_query = self.call_vllm_structured(content=paper)
        
        return optimized_query
    

class MethodFocusedQueryOptimizerAgent(BaseQueryOptimizerAgent):
    """Agent that receives full paper of scientific paper and generates refined query focused on method of given scientific paper"""
    
    def __init__(self, args, vllm_model_dict):
        super().__init__(args, vllm_model_dict)
        self.prompt_path = "Agents/Prompts/ScientificPapers/QueryOptimizer/method_focused_query_optimizer_agent_prompt.yaml"
        self.name = "METHOD FOCUSED AGENT"
        if self.use_gpt == False:
            self.vllm_model = vllm_model_dict["METHOD"]["agent"]
            self.vllm_model_device = vllm_model_dict["METHOD"]["device"]
        self.port_number = 8083
        
    
class ExperimentFocusedQueryOptimizerAgent(BaseQueryOptimizerAgent):
    """Agent that receives full paper of scientific paper and generates refined query focused on experiments of given scientific paper"""
    
    def __init__(self, args, vllm_model_dict):
        super().__init__(args, vllm_model_dict)
        self.prompt_path = "Agents/Prompts/ScientificPapers/QueryOptimizer/experiment_focused_query_optimizer_agent_prompt.yaml"
        self.name = "EXPERIMENT FOCUSED AGENT"
        if self.use_gpt == False:
            self.vllm_model = vllm_model_dict["EXPERIMENT"]["agent"]
            self.vllm_model_device = vllm_model_dict["EXPERIMENT"]["device"]
        self.port_number = 8081


class ResearchQuestionFocusedQueryOptimizerAgent(BaseQueryOptimizerAgent):
    """Agent that receives full paper of scientific paper and generates refined query focused on research question of given scientific paper"""
    
    def __init__(self, args, vllm_model_dict):
        super().__init__(args, vllm_model_dict)
        self.prompt_path = f"Agents/Prompts/ScientificPapers/QueryOptimizer/research_question_focused_query_optimizer_agent_prompt.yaml"
        self.name = "RESEARCH QUESTION FOCUSED AGENT"
        if self.use_gpt == False:
            self.vllm_model = vllm_model_dict["RESEARCH_QUESTION"]["agent"]
            self.vllm_model_device = vllm_model_dict["RESEARCH_QUESTION"]["device"]
        self.port_number = 8082
    

class QueryOptimizer:
    """
    Controller that handles multi agent, single agent query optimization process
    """
    def __init__(self, args, result_folder_path, vllm_model_dict):
        self.args = args
        self.multi_agent = args.multi_agent
        self.use_base_agent = args.use_base_agent
        if self.use_base_agent == True and (args.use_method_agent == True or args.use_experiment_agent == True or args.use_research_question_agent == True):
            raise TypeError("When using base agents, you should not use other aspect-centric agents")
        
        self.agents = []
        prompts_to_save = {}
        if self.multi_agent == True:
            self.agent1 = MethodFocusedQueryOptimizerAgent(args, vllm_model_dict)
            self.agent2 = ExperimentFocusedQueryOptimizerAgent(args, vllm_model_dict)
            self.agent3 = ResearchQuestionFocusedQueryOptimizerAgent(args, vllm_model_dict)
            self.agent4 = BaseQueryOptimizerAgent(args, vllm_model_dict)

            if args.use_base_agent == True:
                self.agents.append(self.agent4)
                system_prompt, user_prompt = self.agent4.open_prompt()
                prompts_to_save["BASE"] = system_prompt
                
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
            
        elif self.multi_agent == False:
            if args.use_base_agent != True:
                raise TypeError("When using Query Optimizer without multi agent setting, you must set use_base_agent to True")
            self.agent1 = BaseQueryOptimizerAgent(args, vllm_model_dict)
            self.agents = [self.agent1]
            
            system_prompt, user_prompt = self.agent1.open_prompt()
            prompts_to_save["BASE"] = system_prompt

        
        prompt_file = os.path.join(result_folder_path, "prompts.json")
        with open(prompt_file, "w") as json_file:
            json.dump(prompts_to_save, json_file, indent=4)
    
    
    def forward(self, paper_path, evaluation_data) -> Tuple[dict, list]:
        """
        Forward multi agent query optimizer
        """
        if self.use_base_agent == False:
            optimized_queries = []
            for agent in self.agents:
                optimized_query = agent.forward(paper_path)
                agent_name = agent.name
                optimized_queries.append((agent_name, optimized_query))
                
        elif self.use_base_agent == True:
            optimized_queries = []
            for agent in self.agents:
                optimized_query = agent.forward(paper_path)
                
            agents = ["METHOD FOCUSED AGENT", "EXPERIMENT FOCUSED AGENT", "RESEARCH QUESTION FOCUSED AGENT"]
            for index, query in enumerate(optimized_query):
                optimized_queries.append((f"{agents[index]}", query))
                
        evaluation_data["Query"]["optimized_queries"] = optimized_queries
        
        return evaluation_data, optimized_queries
    
    def forward_later_rounds(self, full_paper, parent_name, no_forward=False) -> Tuple[dict, list]:
        """
        Forward multi agent query optimizer for SUBAGENTS
        """
        logging.info(f"Forwarding Further Exploration for later rounds")
        if self.use_base_agent == False:
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
        
        elif self.use_base_agent == True:
            paper_path = ""
            for agent in self.agents:
                if no_forward == False:
                    original_optimized_queries = agent.forward_for_later_rounds(paper_path, full_paper, paper_opened=True)

                elif no_forward == True:
                    original_optimzied_queries = []
                
            agents = ["METHOD FOCUSED AGENT", "EXPERIMENT FOCUSED AGENT", "RESEARCH QUESTION FOCUSED AGENT"]
            
            optimized_queries = []
            if no_forward == False:
                for index, query in enumerate(original_optimized_queries):
                    agent_name = f"{parent_name}_{agents[index]}"
                    optimized_queries.append((agent_name, query))
            elif no_forward == True:
                for idx, agent in enumerate(agents):
                    agent_name = f"{parent_name}_{agents[idx]}"
                    optimized_query = ""
                    optimized_queries.append((agent_name, optimized_query))

                    
                
        return optimized_queries
    

    
            
        

        
    