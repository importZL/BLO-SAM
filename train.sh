python train.py \
    --root_path data_path \
    --output ./output \
    --module sam_lora_mask_decoder\
    --max_epoch 100 \
    --num_data 8 \
    --wandb_mode disabled \
    --batch_size 1 \
    --dataset body \
    --exp_type auto_first \
    --base_lr 5e-3 \
    --prompt_base_lr 5e-3 \
    --gpu_id 1 \
    --num_classes 1 \
    --dice_param 0.8 \
    --rank 4 \
    # --unrolled \