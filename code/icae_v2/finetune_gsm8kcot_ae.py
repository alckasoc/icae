import transformers

from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
from peft import (
    LoraConfig,
)
import argparse
import json

from training_utils import pretrain_tokenize_function, DataCollatorForDynamicPadding, train_model
from modeling_icae_multi_span import ICAE

import warnings
warnings.filterwarnings("ignore")
import re


def extract_reasoning_trace(response):
    """
    Extracts the reasoning trace from a response by splitting exactly at "Therefore,".

    Args:
        response (str): The full response including reasoning and the final answer.

    Returns:
        str: The extracted reasoning trace without the final answer.
    """
    # Define the exact split phrase
    split_phrase = "Therefore,"

    # Search for the exact occurrence of "Therefore,"
    match = response.find(split_phrase)

    if match != -1:
        reasoning_trace = response[:match].strip()  # Keep everything before "Therefore,"
    else:
        reasoning_trace = response.strip()  # If not found, return the full response

    return reasoning_trace


def preprocess_function(examples):
    examples["reasoning_trace"] = extract_reasoning_trace(examples["response"])
    return examples

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="meta-llama/Llama-3.2-1B")
    lora_r: int = field(
        default=128,
        metadata={"help": "lora rank"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "lora alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "lora dropout"}
    )
    train: bool = field(
        default=True,
        metadata={"help": "if true, the model ckpt will be initialized for training; else, it's for inference"}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=28000,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    fixed_mem_size: int = field(
        default=1,
        metadata={"help": "Enabling the fixed mem size."},
    )
    mean_compression_rate: int = field(
        default=128*4,
        metadata={"help": "Mean compression rate; default=4"},
    )
    min_tokens_for_lm: int = field(
        default=64,
        metadata={"help": "Minimum tokens for lm objective learning"},
    )
    leave_tokens_for_lm: int = field(
        default=8,
        metadata={"help": "Leave some tokens without loss for lm objective"},
    )
    lm_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio for LM training."},
    )
    add_special_token_for_lm: bool = field(
        default=False,
        metadata={"help": "Add a special token for the prompt of language modeling; default: False"},
    )
    restore_from: str = field(
        default="",
        metadata={"help": "The checkpoint that should be restored from for fine-tuning"}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=1,
        metadata={"help": "The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for evaluation."}
    )
    report_to: str = field(
        default="wandb"
    )
    max_steps: int = field(
        default=5000
    )
    save_strategy: str = field(
        default="no"
    )
    logging_strategy: str = field(
        default="steps"
    )
    logging_steps: int = field(
        default=1
    )
    learning_rate: float = field(
        default=2.5e-5
    )
    lr_scheduler_type: str = field(
        default="cosine"
    )
    lr_scheduler_kwargs: dict = field(default_factory=dict)
    warmup_steps: int = field(
        default=0
    )
    weight_decay: float = field(
        default=0.0
    )
    
    

def main(model_args, training_args, args, notes):    
    print("Loading dataset...")
    ds = load_dataset("ankner/gsm8k-CoT")
    train_dataset = ds["train"]
    eval_dataset = ds["test"]

    # Extract reasoning trace.
    train_dataset = train_dataset.map(preprocess_function)
    eval_dataset = eval_dataset.map(preprocess_function)
    
    train_dataset = train_dataset.map(lambda example: {**example, "text": example['reasoning_trace']}).shuffle(seed=42)
    eval_dataset = eval_dataset.map(lambda example: {**example, "text": example['reasoning_trace']}).shuffle(seed=42)
    print("Dataset loaded successfully...")

    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    print("Loading model...")
    model = ICAE(model_args, training_args, lora_config).to("cuda")
    print("Model loaded successfully...")
    
    memory_size = training_args.fixed_mem_size
    MEM_TOKENS = list(range(model.vocab_size, model.vocab_size + memory_size))

    ### OVERFIT TO 1 EXAMPLE ###
    train_dataset = train_dataset.select([0])
    eval_dataset = eval_dataset.select([0])
    print("train_dataset overfit example: ", train_dataset[0]['question'])

    lines = [
        'Each adult has 32 teeth initially.\n\nFor the first person, 1/4 of 32 teeth were removed: 32 × (1/4) = 8 teeth removed.\n\nFor the second person, 3/8 of 32 teeth were removed: 32 × (3/8) = 12 teeth removed.\n\nFor the third person, 1/2 of 32 teeth were removed: 32 × (1/2) = 16 teeth removed.\n\nThe fourth person had exactly 4 teeth removed.\n\nAdding all teeth removed: 8 + 12 + 16 + 4 = 40 teeth.',
        "Let's find Raymond's jewels first - he has 40 jewels.\n\nHalf of Raymond's jewels is 40 ÷ 2 = 20 jewels.\n\nAaron has 5 more jewels than half of Raymond's jewels, so Aaron has 20 + 5 = 25 jewels.\n\nSiobhan has 2 fewer jewels than Aaron, so she has 25 - 2 = 23 jewels."
    ]
    
    ############################

    print("Tokenizing train/eval datasets...")
    train_dataset = train_dataset.map(pretrain_tokenize_function, batched=True, batch_size=1, fn_kwargs={"model": model, "mem": MEM_TOKENS, "lm_ratio": training_args.lm_ratio})
    eval_dataset = eval_dataset.map(pretrain_tokenize_function, batched=True, fn_kwargs={"model": model, "mem": MEM_TOKENS})
    print("Finished tokenizing train/eval datasets...")

    data_collator = DataCollatorForDynamicPadding(model.pad_token_id)

    print("Training model...")
    train_model(
        args,
        notes,
        model, 
        train_dataset, 
        eval_dataset, 
        model_args,
        training_args, 
        lines,
        data_collator,
    )
    print("Finished training...")

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--lora_r", type=int)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--lora_dropout", type=float)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--lr_scheduler_type", type=str)
    parser.add_argument("--lr_scheduler_kwargs", type=str)
    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument("--optim", type=str)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--notes", type=str)
    args = parser.parse_args()
    
    lr_scheduler_kwargs = json.loads(args.lr_scheduler_kwargs)

    model_args, training_args = ModelArguments(), TrainingArguments(output_dir=args.output_dir)

    model_args.model_name_or_path = args.model_name_or_path
    model_args.lora_r = args.lora_r
    model_args.lora_alpha = args.lora_alpha
    model_args.lora_dropout = args.lora_dropout

    training_args.max_steps = args.max_steps
    training_args.learning_rate = args.learning_rate
    training_args.lr_scheduler_type = args.lr_scheduler_type
    training_args.lr_scheduler_kwargs = lr_scheduler_kwargs
    training_args.warmup_steps = args.warmup_steps
    training_args.optim = args.optim
    training_args.weight_decay = args.weight_decay

    main(model_args, training_args, args, args.notes)