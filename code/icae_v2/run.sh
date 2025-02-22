rm -rf "./output"

python finetune_gsm8kcot_ae.py \
    --output_dir "./output" \
    --model_name_or_path "meta-llama/Llama-3.2-1B-Instruct" \
    --lora_r 1024 \
    --lora_alpha 256 \
    --lora_dropout 0.0 \
    --max_steps 10000 \
    --learning_rate 5e-5 \
    --lr_scheduler_type "cosine" \
    --lr_scheduler_kwargs '{"num_cycles": 1}' \
    --warmup_steps 250 \
    --optim "adamw_torch" \
    --weight_decay 0.01 \
    --notes ""
