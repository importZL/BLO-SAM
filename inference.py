import os
import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import test_single_volume, DiceLoss
from importlib import import_module
from segment_anything import sam_model_registry

from torchvision import transforms
from cal_dice import dice_score


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
parser.add_argument('--volume_path', type=str, default='testset/test_vol_h5/')
parser.add_argument('--dataset', type=str, default='Synapse', help='Experiment name')
parser.add_argument('--num_classes', type=int, default=1)
parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse/', help='list_dir')
parser.add_argument('--output_dir', type=str, default='/output')
parser.add_argument('--img_size', type=int, default=256, help='Input image size of the network')
parser.add_argument('--input_size', type=int, default=256, help='The input size for training SAM model')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--is_savenii', action='store_true', help='Whether to save results during inference')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--ckpt', type=str, default='../Auto_SAMed/checkpoints/sam_vit_b_01ec64.pth', help='Pretrained checkpoint')
parser.add_argument('--lora_ckpt', type=str, default='checkpoints/epoch_159.pth', help='The checkpoint from LoRA')
parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')
parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
parser.add_argument('--gpu_id', type=int, default=0, help='total gpu')
parser.add_argument('--prompt_ckpt', type=str, default='checkpoints/epoch_159.pth', help='The checkpoint for prompt')

# train
parser.add_argument('--Training', default=False, type=bool, help='Training or not')
parser.add_argument('--init_method', default='tcp://127.0.0.1:33111', type=str, help='init_method')
parser.add_argument('--train_steps', default=60000, type=int, help='total training steps')
parser.add_argument('--vst_img_size', default=224, type=int, help='network input size')
parser.add_argument('--lr_decay_gamma', default=0.1, type=int, help='learning rate decay')
parser.add_argument('--lr', default=1e-4, type=int, help='learning rate')
parser.add_argument('--epochs', default=200, type=int, help='epochs')
parser.add_argument('--batch_size', default=11, type=int, help='batch_size')
parser.add_argument('--stepvalue1', default=30000, type=int, help='the step 1 for adjusting lr')
parser.add_argument('--stepvalue2', default=45000, type=int, help='the step 2 for adjusting lr')
parser.add_argument('--trainset', default='DUTS/DUTS-TR', type=str, help='Trainging set')
parser.add_argument('--save_model_dir', default='/home/li/workspace/SAMAug/vst_main/ckpt', type=str, help='save model path')

# test
parser.add_argument('--Testing', default=True, type=bool, help='Testing or not')
# evaluation
parser.add_argument('--Evaluation', default=False, type=bool, help='Evaluation or not')
parser.add_argument('--methods', type=str, default='RGB_VST', help='evaluated method name')
parser.add_argument('--save_dir', type=str, default='./', help='path for saving result.txt')

args = parser.parse_args()


@torch.no_grad()
def inference(args, model, testloader, multimask_output, device):
    score_dice = []
    model.eval()
    for i, sampled_batch in enumerate(tqdm(testloader)):
        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
        low_res_label_batch = sampled_batch['low_res_label']
        image_batch, label_batch = image_batch.to(device=device), label_batch.to(device=device)
        low_res_label_batch = low_res_label_batch.to(device=device)
        try:
            outputs = model(image_batch, multimask_output, args.img_size)
        except:
            print(sampled_batch['path'])
            print(image_batch.shape)
    
        low_res_logits = outputs['low_res_logits']
        dice = dice_score(low_res_logits, low_res_label_batch, bg=False)
        
        score_dice.append(dice.cpu().numpy())

    return np.mean(score_dice)


def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict




if __name__ == '__main__':
    if args.config is not None:
        # overwtite default configurations with config file\
        config_dict = config_to_dict(args.config)
        for key in config_dict:
            setattr(args, key, config_dict[key])

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    
    device = torch.device(f'cuda:{args.gpu_id}')

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                    num_classes=args.num_classes,
                                                                    checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                    pixel_std=[1, 1, 1])
    low_res = img_embedding_size * 4
    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank)
    
    assert args.lora_ckpt is not None
    net.load_lora_parameters(args.lora_ckpt, device)
    net = net.to(device=device)
    
    # net1 = torch.load('/home/li/workspace/SAMed/final.pt', map_location=device)    
    # params1 = net.named_parameters()    
    # params2 = net1.named_parameters()    
    # for p1, p2 in zip(params1, params2):
    #     if p1[0] != p2[0]:
    #         print("Error: models have different parameter names!")
    #         break
    #     if torch.equal(p1[1], p2[1]):
    #         continue
    #     else:
    #         print(f"Parameters: {p1[0]} are different")
    
    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False
    
    testloader = None
    
    if args.dataset == 'kvasir':
        from datasets.dataset_kvasir import Synapse_dataset, RandomGenerator
    elif args.dataset == 'lung':
        from datasets.dataset_lung import Synapse_dataset, RandomGenerator
    elif args.dataset in ['brow', 'eye', 'hair', 'nose', 'mouth', 'celeb']:
        from datasets.dataset_celeb import Synapse_dataset, RandomGenerator
    elif args.dataset in ['car', 'wheel', 'window']:
        from datasets.dataset_car import Synapse_dataset, RandomGenerator
    elif args.dataset == 'teeth':
        from datasets.dataset_teeth import Synapse_dataset, RandomGenerator
    elif args.dataset == 'body':
        from datasets.dataset_body import Synapse_dataset, RandomGenerator
    else:
        print("##### Unimplemented dataset #####")
        sys.exit()
    db_test = Synapse_dataset(train_dir=args.volume_path, 
                              dataset=args.dataset,
                              transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res], split="test")]))
    testloader = DataLoader(db_test, batch_size=1, num_workers=4, pin_memory=True)

    dice_result = inference(args, net, testloader, multimask_output, device)
    print("Test dice score: %.4f" % dice_result)
