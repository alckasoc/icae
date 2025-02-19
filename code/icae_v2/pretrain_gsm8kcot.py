import transformers

from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
from peft import (
    LoraConfig,
)

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
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    debug_data: bool = field(default=False, metadata={"help": "Enable debug dataset to quickly verify the training process"})

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
        default=20
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
    

def main():    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print("Loading dataset...")
    ds = load_dataset("ankner/gsm8k-CoT")
    train_dataset = ds["train"]
    eval_dataset = ds["test"]

    train_dataset = train_dataset.map(lambda example: {**example, "text": f"{example['question']}\n{example['response']}"}).shuffle(seed=42)
    eval_dataset = eval_dataset.map(lambda example: {**example, "text": f"{example['question']}\n{example['response']}"}).shuffle(seed=42)
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

    train_dataset = train_dataset.map(preprocess_function)
    eval_dataset = eval_dataset.map(preprocess_function)

    print("Tokenizing train/eval datasets...")
    train_dataset = train_dataset.map(pretrain_tokenize_function, batched=True, batch_size=1, fn_kwargs={"model": model, "mem": MEM_TOKENS, "lm_ratio": training_args.lm_ratio})
    eval_dataset = eval_dataset.map(pretrain_tokenize_function, batched=True, fn_kwargs={"model": model, "mem": MEM_TOKENS})
    print("Finished tokenizing train/eval datasets...")

    ### OVERFIT TO 1 EXAMPLE ###
    train_dataset = train_dataset.select([0])
    eval_dataset = eval_dataset.select([0])
    print("train_dataset overfit example: ", train_dataset[0]['question'])

    lines = [
        "Four adults with 32 teeth went to the dentist for a checkup after realizing they were having severe tooth pain. They were found to have different numbers of damaged teeth, and each person had some teeth removed. The first person had 1/4 of all his teeth removed, and the second person had 3/8 of his teeth removed, the third person had half of his teeth removed, while the last person only had 4 teeth removed. What's the total number of teeth removed at the dental clinic?"
    ]
    
    ############################

    data_collator = DataCollatorForDynamicPadding(model.pad_token_id)

    print("Training model...")
    train_model(
        model, 
        train_dataset, 
        eval_dataset, 
        model_args,
        data_args,
        training_args, 
        lines,
        data_collator,
    )
    print("Finished training...")

# model_args.model_name_or_path = "meta-llama/Llama-3.2-1B"
# training_args.bf16 = True
# training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}  # manually add this argument in the code
# training_args.lm_ratio = 0.0

if __name__ == "__main__":
    main()