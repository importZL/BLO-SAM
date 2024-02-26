import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from utils import DiceLoss
import copy
import math


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss


class Prompt(object):

    def __init__(self, model, args, max_iterations):
        self.model = model
        self.args = args
        if args.num_classes > 1:
            self.multimask_output = True
        else:
            self.multimask_output = False
        self.dice_loss = DiceLoss(args.num_classes + 1)
        self.ce_loss = nn.CrossEntropyLoss()
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.max_iterations = max_iterations
        self.optimizer = torch.optim.AdamW(
            self._get_model_prompt_parameters(self.model),
            lr=args.prompt_base_lr, betas=(self.beta1, self.beta2), weight_decay=args.prompt_weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     self.optimizer, max_lr=args.base_lr, total_steps=self.max_iterations)


    def step(self, sampled_batch, valid_sampled_batch, eta, network_optimizer, unrolled, cur_iter):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(sampled_batch, valid_sampled_batch, eta, network_optimizer)
        else:
            self._backward_step(valid_sampled_batch)
        self.optimizer.step()
        
        # self.scheduler.step()
        lr_ = self.args.base_lr * (1.0 - cur_iter / self.max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_

    def _backward_step(self, sampled_batch):
        image_batch = sampled_batch['image']
        low_res_label_batch = sampled_batch['low_res_label']
        image_batch = image_batch.cuda()
        low_res_label_batch = low_res_label_batch.cuda()
        
        outputs = self.model(image_batch, self.multimask_output, self.args.img_size)
        loss = calc_loss(outputs, low_res_label_batch, self.ce_loss, self.dice_loss, self.args.dice_param)
        
        loss.backward()

    def _backward_step_unrolled(self, sampled_batch, valid_batch, eta, network_optimizer):
        image_batch = sampled_batch['image']
        low_res_label_batch = sampled_batch['low_res_label']
        image_batch = image_batch.cuda()
        low_res_label_batch = low_res_label_batch.cuda()
        
        valid_image_batch = valid_batch['image']
        valid_low_res_label_batch = valid_batch['low_res_label']
        valid_image_batch = valid_image_batch.cuda()
        valid_low_res_label_batch = valid_low_res_label_batch.cuda()
        
        unrolled_model = self._compute_unrolled_model(image_batch, low_res_label_batch, eta, network_optimizer)
        unrolled_outputs = unrolled_model(valid_image_batch, self.multimask_output, self.args.img_size)
        unrolled_loss = \
            calc_loss(unrolled_outputs, valid_low_res_label_batch, self.ce_loss, self.dice_loss, self.args.dice_param)
        
        unrolled_loss.backward()
        dalpha = []
        for v in self._get_model_prompt_parameters(unrolled_model):
            if v.grad == None:
                v.grad = torch.zeros_like(v, memory_format=torch.preserve_format)
            dalpha.append(v.grad)
        vector = []
        for v in self._get_model_parameters(unrolled_model):
            if v.grad == None:
                v.grad = torch.zeros_like(v, memory_format=torch.preserve_format)
            vector.append(v.grad.data)
        implicit_grads = self._hessian_vector_product(vector, image_batch, low_res_label_batch)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(ig.data, alpha=eta)

        for v, g in zip(self._get_model_prompt_parameters(self.model), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)
    
    def _compute_unrolled_model(self, image_batch, low_res_label_batch, eta, network_optimizer):
        
        outputs = self.model(image_batch, self.multimask_output, self.args.img_size)
        loss = calc_loss(outputs, low_res_label_batch, self.ce_loss, self.dice_loss, self.args.dice_param)
        
        theta = _concat(self._get_model_parameters(self.model)).data        
        exp_avg = []
        exp_avg_sq = []
        for v in self._get_model_parameters(self.model):
            state = network_optimizer.state[v]
            if len(state) == 0:
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(v, memory_format=torch.preserve_format)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(v, memory_format=torch.preserve_format)
            exp_avg.append(state['exp_avg'])
            exp_avg_sq.append(state['exp_avg_sq'])
        exp_avg = _concat(exp_avg)
        exp_avg_sq = _concat(exp_avg_sq)
        
        grad = []
        loss.backward()
        for v in self._get_model_parameters(self.model):
            if v.grad == None:
                v.grad = torch.zeros_like(v, memory_format=torch.preserve_format)
            grad.append(v.grad)
        grad = _concat(grad).data
        # Perform stepweight decay
        theta.mul_(1 - eta * self.args.weight_decay)
        bias_correction1 = 1 - self.beta1
        bias_correction2 = 1 - self.beta2
        # Decay the first and second moment running average coefficient
        exp_avg.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
        exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(1e-8)
        step_size = eta / bias_correction1
        theta.addcdiv_(exp_avg, denom, value=-step_size)        
        
        unrolled_model = self._construct_model_from_theta(theta)
        return unrolled_model

    def _construct_model_from_theta(self, theta):
        model_new = copy.deepcopy(self.model)
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            if v.requires_grad and ("no_mask_embed" not in k):
                v_length = np.prod(v.size())
                params[k] = theta[offset: offset+v_length].view(v.size())
                offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self._get_model_parameters(self.model), vector):
            p.data.add_(v, alpha=R)
        
        outputs = self.model(input, self.multimask_output, self.args.img_size)
        loss = calc_loss(outputs, target, self.ce_loss, self.dice_loss, self.args.dice_param)
        
        grads_p = torch.autograd.grad(loss, self.model.sam.prompt_encoder.no_mask_embed.weight)

        for p, v in zip(self._get_model_parameters(self.model), vector):
            p.data.sub_(v, alpha=2*R)
        
        outputs = self.model(input, self.multimask_output, self.args.img_size)
        loss = calc_loss(outputs, target, self.ce_loss, self.dice_loss, self.args.dice_param)
        
        grads_n = torch.autograd.grad(loss, self.model.sam.prompt_encoder.no_mask_embed.weight)

        for p, v in zip(self._get_model_parameters(self.model), vector):
            p.data.add_(v, alpha=R)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]
    
    def _get_model_parameters(self, model):
        return list(p for n, p in model.named_parameters() if (p.requires_grad and ("no_mask_embed" not in n)))
    
    def _get_model_prompt_parameters(self, model):
        return list(p for n, p in model.named_parameters() if (p.requires_grad and ("no_mask_embed" in n)))