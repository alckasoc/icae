import transformers

from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
from training_utils import pretrain_tokenize_function
from peft import (
    LoraConfig,
)
from modeling_icae_multi_span import ICAE

import warnings
warnings.filterwarnings("ignore")


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="meta-llama/Llama-3.2-1B")
    lora_r: int = field(
        default=128,
        metadata={"help": "lora rank"}
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
        default=128,
        metadata={"help": "Enalbing the fixed mem size."},
    )
    mean_compression_rate: int = field(
        default=4,
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
        default=0.5,
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


def main():    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print("Loading dataset...")
    ds = load_dataset("ankner/gsm8k-CoT")
    train_dataset = ds["train"]
    eval_dataset = ds["test"]

    train_dataset = train_dataset.map(lambda example: {**example, "text": f"{example['question']}\n{example['response']}"})
    eval_dataset = eval_dataset.map(lambda example: {**example, "text": f"{example['question']}\n{example['response']}"})
    print("Dataset loaded successfully...")
    
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    print("Loading model...")
    model = ICAE(model_args, training_args, lora_config)
    print("Model loaded successfully...")
    
    memory_size = training_args.fixed_mem_size
    MEM_TOKENS = list(range(model.vocab_size, model.vocab_size + memory_size))

    print("Tokenizing train/eval datasets...")
    train_dataset = train_dataset.map(pretrain_tokenize_function, batched=True, batch_size=1, fn_kwargs={"model": model, "mem": MEM_TOKENS, "lm_ratio": training_args.lm_ratio})
    eval_dataset = eval_dataset.map(pretrain_tokenize_function, batched=True, fn_kwargs={"model": model, "mem": MEM_TOKENS})
    print("Finished tokenizing train/eval datasets...")

# model_args.model_name_or_path = "meta-llama/Llama-3.2-1B"
# training_args.bf16 = True
# training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}  # manually add this argument in the code
# training_args.lm_ratio = 0.0

if __name__ == "__main__":
    main()