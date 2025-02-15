import transformers
from modeling_icae_multi_span import ICAE, ModelArguments, DataArguments, TrainingArguments
parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

print(model_args)
