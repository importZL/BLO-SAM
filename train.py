import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from importlib import import_module

from sam_lora_image_encoder import LoRA_Sam
from segment_anything import sam_model_registry

from trainer import trainer
import time


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='', help='root dir for data')
parser.add_argument('--output', type=str, default='./output')
parser.add_argument('--dataset', type=str, default='kvasir', help='experiment_name')
parser.add_argument('--num_classes', type=int, default=5, help='output channel of network')
parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
parser.add_argument('--gpu_id', type=str, default='1', help='total gpu')
parser.add_argument('--deterministic', type=bool, default=False, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.005, help='segmentation network learning rate')
parser.add_argument('--prompt_base_lr', type=float, default=0.005, help='prompt learning rate')
parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--vit_name', type=str, default='vit_b', help='select one vit model')
parser.add_argument('--ckpt', type=str, default='/data1/li/Auto_SAMed/checkpoints/sam_vit_b_01ec64.pth', help='Pretrained checkpoint')
parser.add_argument('--lora_ckpt', type=str, default=None, help='Finetuned lora checkpoint')
parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
parser.add_argument('--module', type=str, default='sam_lora_mask_decoder')
parser.add_argument('--dice_param', type=float, default=0.8)

parser.add_argument('--num_data', type=int, default=10, help='batch_size per gpu')
parser.add_argument('--exp_type', type=str, default='vanilla')

parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay')
parser.add_argument('--prompt_weight_decay', type=float, default=0.1, help='weight decay')
parser.add_argument('--unrolled', action='store_true', help='')
parser.add_argument('--wandb_mode', type=str, default='disabled')
args = parser.parse_args()

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    args.exp = args.dataset + str(args.num_data) + '_' + args.exp_type + '_img' + str(args.img_size) 
    snapshot_path = os.path.join(args.output, "{}".format(args.exp))
    snapshot_path = snapshot_path + '_' + time.strftime("%Y%m%d-%H%M%S")

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                num_classes=args.num_classes,
                                                                checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1])

    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()

    # net = LoRA_Sam(sam, args.rank).cuda()
    if args.lora_ckpt is not None:
        net.load_lora_parameters(args.lora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    low_res = img_embedding_size * 4

    config_file = os.path.join(snapshot_path, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)

    trainer(args, net, snapshot_path, multimask_output, low_res)
