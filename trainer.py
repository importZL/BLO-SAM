import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils import DiceLoss, getStat
from torchvision import transforms
from icecream import ic
import wandb

from prompt import Prompt
from medpy import metric
from cal_dice import dice_score


def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice


@torch.no_grad()
def validate(args, model, validloader, multimask_output):
    score_dice = []
    model.eval()
    for _, sampled_batch in enumerate(validloader):
        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
        low_res_label_batch = sampled_batch['low_res_label']
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
        low_res_label_batch = low_res_label_batch.cuda()

        outputs = model(image_batch, multimask_output, args.img_size)

        low_res_logits = outputs['low_res_logits']
        dice = dice_score(low_res_logits, low_res_label_batch)        
        score_dice.append(dice.cpu().numpy())
    model.train()
    return np.mean(score_dice)

def trainer(args, model, snapshot_path, multimask_output, low_res):
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
    
    exp_name = f"{args.dataset}-{args.num_data}-{args.exp_type}-lora_mask"
    logger = wandb.init(project='Auto-sam', name=exp_name, resume='allow', anonymous='must', mode=args.wandb_mode)
    logger.config.update(vars(args))
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    base_lr = args.base_lr
    num_classes = args.num_classes
    
    db_train = Synapse_dataset(
        train_dir=args.root_path, num_data=args.num_data, dataset=args.dataset,
        transform=transforms.Compose([RandomGenerator(
            output_size=[args.img_size, args.img_size], low_res=[low_res, low_res])
        ]),
    )
    
    num_train = int(len(db_train)*0.5)
    num_valid = len(db_train) - num_train
    selector = range(len(db_train))
    logging.info("The length of train set is: {}".format(num_train))
    logging.info("The length of train set is: {}".format(num_valid))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn, sampler=selector[:num_train])
    validloader = DataLoader(db_train, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn, sampler=selector[num_train:])
    
    model.train()
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes + 1)

    optimizer = optim.AdamW(
        list(p for n, p in model.named_parameters() if (p.requires_grad and ("no_mask_embed" not in n))),
        lr=base_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)   
    
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=base_lr, total_steps=max_iterations)
    
    prompt_module = Prompt(model=model, args=args, max_iterations=max_iterations)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    for epoch_num in range(max_epoch):
        for i_batch, sampled_batch in enumerate(trainloader):
            
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
            low_res_label_batch = sampled_batch['low_res_label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            low_res_label_batch = low_res_label_batch.cuda()
            
            outputs = model(image_batch, multimask_output, args.img_size)
            loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, args.dice_param)
            logger.log({'info/stage1_loss': loss})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            valid_batch = next(iter(validloader)) 
            for param_group in optimizer.param_groups:
                eta = param_group['lr']    
                break
            prompt_module.step(sampled_batch, valid_batch, eta, optimizer, unrolled=args.unrolled, cur_iter=iter_num) 
            
            ##### Adjust Learning Rate #####
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1

            if iter_num % 10 == 0:
                logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
            
                image = image_batch[0, :, :, :]
                output_masks = outputs['masks'] 
                labs = label_batch[0, ...].unsqueeze(0)
                    
                ims = {}
                image = (image - image.min()) / (image.max() - image.min())
                image = image.mul(255).permute(1, 2, 0).to('cpu').numpy()
                ims['train/Image'] = wandb.Image(image)
                   
                output_masks = torch.argmax(torch.softmax(output_masks, dim=1), dim=1, keepdim=True)[0, ...]                
                output_masks = output_masks.mul(255).to('cpu').numpy()
                ims['train/Prediction'] = wandb.Image(output_masks)

                labs = labs.mul(255).to('cpu').numpy()
                ims['train/GroundTruth'] = wandb.Image(labs)

                logger.log(ims)

        # validate the model at every epoch ending
        valid_score = validate(args, model, validloader, multimask_output)
        logging.info('Epoch %d : valid score : %f' % (epoch_num + 1, valid_score))
        logger.log({'info/valid_score': valid_score})
        
        if valid_score > best_performance:
            best_performance = valid_score
            save_mode_path = os.path.join(snapshot_path, 'best.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
        
        # test_score = validate(args, model, dice_loss, testloader, multimask_output)
        # logger.log({'info/test_score': test_score})
        
        
    save_mode_path = os.path.join(snapshot_path, 'final.pth')
    try:
        model.save_lora_parameters(save_mode_path)
    except:
        model.module.save_lora_parameters(save_mode_path)
      
    return "Training Finished!"


if __name__ == '__main__':
    print('test')
