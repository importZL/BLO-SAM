python -W ignore inference.py \
    --volume_path data_path \
    --lora_ckpt model_path \
    --gpu_id 0 \
    --module sam_lora_mask_decoder \
    --dataset body \
    --num_classes 1 \