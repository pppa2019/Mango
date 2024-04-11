model= # your backbone model path
data_path=train_data/python_neg_contrastive.json
suffix=# assign a name for your output model
deepspeed --master_port=29913  src/train_wizardcoder_lora.py \
        --model_name_or_path $model \
        --data_path $data_path \
        --output_dir "output/"$suffix \
        --num_train_epochs 3 \
        --model_max_length 2048 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 64 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 10 \
        --save_total_limit 5 \
        --learning_rate 2e-5 \
        --warmup_steps 15 \
        --logging_steps 2 \
        --lr_scheduler_type "cosine" \
        --report_to "tensorboard" \
        --gradient_checkpointing True \
        --deepspeed deepspeed_config1.json \
        --fp16 True \
        --overwrite \
        --seed 42 \
        --train_target margin_contrastive \
        --contrastive_train \
        --margin 0.10