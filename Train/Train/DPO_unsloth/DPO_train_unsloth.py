import os

from dataclasses import dataclass, field
from typing import Optional, Dict

import logging
import torch

from unsloth import is_bfloat16_supported
import transformers
import argparse
import json
from transformers import (
    TrainingArguments,
)

from transformers import StoppingCriteria, TrainerCallback
from datasets import load_dataset

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from trl import DPOTrainer, DPOConfig

from unsloth import FastLanguageModel, PatchDPOTrainer
import wandb

torch.backends.cuda.matmul.allow_tf32 = True


logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

    
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=""
    )
    agent_name: str = field(
        default="method_agent"
    )

@dataclass
class DataArguments:
    dataset: str = field(
        default="",
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )


## 얘는 보기 ##
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(
        default=None
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help":"Lora dropout."}
    )
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    evaluation_strategy: str  = field(
        default = 'no',
        metadata={"help": "check to see whether you are going to use validation set"}
    )
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=2000, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) 
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=40, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})
    load_in_4bit: bool = field(default=False, metadata={"help": "choose to whether use quantized model"})
    project_name: str = field(default="", metadata={"help": "Name of the project this training is going to be run"})
    max_length: Optional[int] = field(default=50000)
    max_prompt_length: int = field(default=40000)
    max_completion_length: int = field(default=5000)
    beta: float = field(default=0.1)
    
        
# Tokenization function
def preprocess_function(examples, tokenizer):
    chosen = tokenizer(examples["chosen"], truncation=True, padding="max_length")
    rejected = tokenizer(examples["rejected"], truncation=True, padding="max_length")

    return {
        "chosen_input_ids": chosen["input_ids"],
        "chosen_attention_mask": chosen["attention_mask"],
        "rejected_input_ids": rejected["input_ids"],
        "rejected_attention_mask": rejected["attention_mask"],
    }
    

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = []):
      StoppingCriteria.__init__(self), 

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stops = []):
      self.stops = stops
      for i in range(len(stops)):
        self.stops = self.stops[i]

class SafeWandbCallback(TrainerCallback):
    def __init__(self, project_name: str):
        self.project_name = project_name  
    
    def on_init_end(self, args, state, control, **kwargs):
        if wandb.run is None:
            wandb.init(name=self.project_name)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if wandb.run is not None and logs is not None:
            wandb.log(logs, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        print("Training finished, but skipped wandb finalization to avoid DeepSpeed error.")

def train():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments,
    ))
    model_args, data_args, training_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    
    dtype=None
    model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = args.model_name_or_path,
            max_seq_length = args.max_length,
            dtype = dtype,
            load_in_4bit = args.load_in_4bit,
        )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = 64, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 64,
        lora_dropout = 0, 
        bias = "none",    
        
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
        use_rslora = False,  
        loftq_config = None, 
    )

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "right"
    tokenizer.max_length = args.max_length
    tokenizer.truncation = True

    training_args_DPO = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,  # Adjust per device
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_length=args.max_length,
        max_prompt_length = args.max_prompt_length,
        max_completion_length = args.max_completion_length,
        max_grad_norm = args.max_grad_norm,
        logging_steps=args.logging_steps,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        beta=args.beta,
        report_to=[]
    )
    
    model_args_dict = vars(model_args)
    training_args_dict = vars(training_args)
    config_DPO = training_args_DPO.to_dict()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    directory_to_save_DPO_config = os.path.join(args.output_dir, "DPO_configs.json")
    model_configs_directory_to_save = os.path.join(args.output_dir, "model_args.json")
    training_configs_directory_to_save = os.path.join(args.output_dir, "train_args.json")
    
    # Save to a JSON file
    with open(directory_to_save_DPO_config, "w") as f:
        json.dump(config_DPO, f, indent=4)

    with open(model_configs_directory_to_save, "w") as f:
        json.dump(model_args_dict, f, indent=4)

    with open(training_configs_directory_to_save, "w") as f:
        additional_training_configs = {}
        additional_training_configs["load_in_4bit"] = args.load_in_4bit
        json.dump(additional_training_configs, f, indent=4)

    print(args.dataset)
    # Load preference dataset
    print(args.agent_name)
    train_dataset = load_dataset(args.dataset, split=args.agent_name)


    PatchDPOTrainer()

    # Initialize DPO Trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args_DPO,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        callbacks=[SafeWandbCallback(args.project_name)]
    )
    
    trainer.train()
    

    
        
    

if __name__ == "__main__":
    train()