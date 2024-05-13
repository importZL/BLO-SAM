python -W ignore inference.py \
    --volume_path /data/li/data/CelebAMask-HQ/test/Images \
    --lora_ckpt /data2/li/workspace/BLO-SAM/output/brow4_auto_first_img256_20240513-072623/best.pth \
    --gpu_id 1 \
    --module sam_lora_mask_decoder \
    --dataset brow \
    --num_classes 1 \