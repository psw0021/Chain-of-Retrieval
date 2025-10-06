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

class BaseSelector:
    """
    Base Selector Module for selecting the neighboring 
    document for subsequent literature search
    """
    def __init__(self, args) -> None:
        self.args = args
        self.name = "BASE"

    

class MethodFocusedSelector(BaseSelector):
    """
    Method-Focused Selector Module for selecting the 
    neighboring document for subsequent literature search
    """
    def __init__(self, args):
        super().__init__(args)
        self.name = "METHOD FOCUSED AGENT"
        
    
    
class ExperimentFocusedSelector(BaseSelector):
    """
    Experiment-Focused Selector Module for selecting the 
    neighboring document for subsesquent literature search
    """
    def __init__(self, args):
        super().__init__(args)
        self.name = "EXPERIMENT FOCUSED AGENT"
        

class ResearchQuestionFocusedSelector(BaseSelector):
    """
    Research-Question-Focused Selector Module for selecting the neighboring 
    document for subsesquent literature search
    """
    
    def __init__(self, args):
        super().__init__(args)
        self.name = "RESEARCH QUESTION FOCUSED AGENT"
    

class Selector:
    """
    Controller that handles aspect-aware selectors.
    """
    def __init__(self, args):
        self.args = args
        self.multi_agent = args.multi_agent
        self.selector_starting_idx = args.selector_starting_idx
        logging.info(f"Starting index for selector is {self.selector_starting_idx}")
        
        self.selector_agents = []
        if self.multi_agent == True:
            self.selector1 = MethodFocusedSelector(args)
            self.selector2 = ExperimentFocusedSelector(args)
            self.selector3 = ResearchQuestionFocusedSelector(args)

            if args.use_method_agent == True:
                self.selector_agents.append(self.selector1)
                
            if args.use_experiment_agent == True:
                self.selector_agents.append(self.selector2)

            if args.use_research_question_agent == True:
                self.selector_agents.append(self.selector3)
    
    
    def forward_for_multirounds(self, organized_retrieved_top_corpus_for_refinement: dict) -> dict:
        """
        Forward Generation of Aspect-Aware Verifiers.
        """
        def concatenate_aspect_specific_top_corpus(aspect_specific_top_corpus_for_selection):
            formatted_aspect_specific_top_corpus_for_selection = ""
            for idx in range(0, len(aspect_specific_top_corpus_for_selection)):
                formatted_aspect_specific_top_corpus_for_selection = formatted_aspect_specific_top_corpus_for_selection + f"{aspect_specific_top_corpus_for_selection[idx]}\n\n"
            
            return formatted_aspect_specific_top_corpus_for_selection
            
        filtered_candidates_dict = {}
        
        subtree_agents = list(organized_retrieved_top_corpus_for_refinement.keys())
        for agent in subtree_agents:
            logging.info(f"Running selector for {agent} agent")
            selector_name = agent
            aspect_specific_top_corpus_for_selection, aspect_specific_top_corpus_for_selection_full_paper = organized_retrieved_top_corpus_for_refinement[selector_name]
            formatted_aspect_specific_top_corpus_for_selection = concatenate_aspect_specific_top_corpus(aspect_specific_top_corpus_for_selection)
            formatted_aspect_specific_top_corpus_for_selection_full_paper =  concatenate_aspect_specific_top_corpus(aspect_specific_top_corpus_for_selection_full_paper)
            
            filtered_candidates_dict[selector_name] = (formatted_aspect_specific_top_corpus_for_selection, formatted_aspect_specific_top_corpus_for_selection_full_paper)
        
        return filtered_candidates_dict
    
    def _find_nonoverlapping_corpus(self, top_k_corpus_for_refinement_abstract_list: list, top_k_corpus_for_refinement_full_paper_list: list, previous_organized_retrieved_top_corpus_per_agent_cache: dict, agent_name: str, input_top_k: int)->tuple:
        """
        Given aspect-aware cache per branch, iterate through retrieved results and select the most similar paper with previous query that retrieved such results,
        that do not exist within the aspect-aware cache.
        """
        previously_used_corpus_cache = previous_organized_retrieved_top_corpus_per_agent_cache[agent_name]
        chosen_corpus = []
        chosen_corpus_full_paper = []
        for index in range(self.selector_starting_idx, len(top_k_corpus_for_refinement_abstract_list)):
            current_corpus = top_k_corpus_for_refinement_abstract_list[index]
            checked_index = 0
            for previous_corpus in previously_used_corpus_cache:
                if current_corpus != previous_corpus:
                    checked_index = checked_index + 1
                
            if checked_index == len(previously_used_corpus_cache):
                chosen_corpus.append(current_corpus)
                chosen_corpus_full_paper.append(top_k_corpus_for_refinement_full_paper_list[index])
                    
            if len(chosen_corpus) == input_top_k:
                return chosen_corpus, chosen_corpus_full_paper        
            
        return chosen_corpus, chosen_corpus_full_paper
    
    def _find_corpus_nearest(self, top_k_corpus_for_refinement_abstract_list: list, top_k_corpus_for_refinement_full_paper_list: list, previous_organized_retrieved_top_corpus_per_agent_cache: dict, agent_name: str, input_top_k: int)->tuple:
        """
        Iterate through retrieved results and select the most similar paper with previous query that retrieved such results,
        without considering aspect aware cache.
        """
        chosen_corpus = []
        chosen_corpus_full_paper = []
        for index in range(self.selector_starting_idx, len(top_k_corpus_for_refinement_abstract_list)):
            chosen_corpus.append(top_k_corpus_for_refinement_abstract_list[index])
            chosen_corpus_full_paper.append(top_k_corpus_for_refinement_full_paper_list[index])
                    
            if len(chosen_corpus) == input_top_k:
                return chosen_corpus, chosen_corpus_full_paper        
            
        return chosen_corpus, chosen_corpus_full_paper