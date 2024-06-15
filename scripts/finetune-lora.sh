#!/bin/bash
export HF_HUB_OFFLINE=True
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export https_proxy="http://10.3.73.27:7891"
export WANDB_MODE=dryrun
export WANDB_PROJECT=UHD
# export WANDB_ENTITY=964730078

deepspeed --master_port 29642 llava_uhd/train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /data2/llm_common/vicuna-7b-v1.5 \
    --version plain \
    --data_path /data2/datasets/llava/blip_laion_cc_sbu_558k.json \
    --image_folder /data2/datasets/llava/llava1.5_pretrain/ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none