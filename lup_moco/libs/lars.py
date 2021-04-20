#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-21 23:27:53
# @Author  : Dengpan Fu (t-defu@microsoft.com)

import os
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class LARS(object):
    """
    Slight modification of LARC optimizer from
    https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py
    Matches one from SimCLR implementation 
    https://github.com/google-research/simclr/blob/master/lars_optimizer.py
    Args:
        optimizer: Pytorch optimizer to wrap and modify learning rate for.
        trust_coefficient: Trust coefficient for calculating the adaptive lr. 
        See https://arxiv.org/abs/1708.03888
    """

    def __init__(self, 
                 optimizer, 
                 trust_coefficient=0.001, 
                 eps=1e-8, 
                 ):
        self.param_groups = optimizer.param_groups
        self.optim = optimizer
        self.trust_coefficient = trust_coefficient
        self.eps = eps

    def __getstate__(self):
        return self.optim.__getstate__()

    def __setstate__(self, state):
        self.optim.__setstate__(state)

    def __repr__(self):
        return self.optim.__repr__()

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)

    def step(self):
        with torch.no_grad():
            weight_decays = []
            for group in self.optim.param_groups:
                # absorb weight decay control from optimizer
                weight_decay = group['weight_decay'] if 'weight_decay' in group else 0
                weight_decays.append(weight_decay)
                group['weight_decay'] = 0
                for p in group['params']:
                    if p.grad is None:
                        continue

                    if weight_decay != 0:
                        p.grad.data += weight_decay * p.data

                    param_norm = torch.norm(p.data)
                    grad_norm = torch.norm(p.grad.data)
                    adaptive_lr = 1.

                    if param_norm != 0 and grad_norm != 0 and group['layer_adaptation']:
                        adaptive_lr = self.trust_coefficient * param_norm / \
                                        (grad_norm + param_norm * weight_decay + self.eps)

                    p.grad.data += weight_decay * p.data
                    p.grad.data *= adaptive_lr

        self.optim.step()
        # return weight decay control to optimizer
        for i, group in enumerate(self.optim.param_groups):
            group['weight_decay'] = weight_decays[i]

def has_sub_str(source, sub_strs):
    for sub_str in sub_strs:
        if sub_str in source:
            return True
    return False

def get_lars_optimizer(model, lr, momentum=0.9, weight_decay=1e-4, 
    excludes=['bn', 'bias', 'downsample.1']):
    if not excludes:
        param_groups = [{'params': model.parameters(), 
                        'weight_decay': weight_decay, 
                        'layer_adaptation': False}]
    else:
        params_adapt, params_non_adapt = [], []
        for name, param in model.named_parameters():
            if has_sub_str(name, excludes):
                params_non_adapt.append(param)
            else:
                params_adapt.append(param)
        param_groups = [{'params': params_non_adapt, 
                        'weight_decay': 0, 
                        'layer_adaptation': False}, 
                        {'params': params_adapt, 
                        'weight_decay': weight_decay, 
                        'layer_adaptation': True}]
    optimizer = torch.optim.SGD(param_groups, lr=lr, momentum=momentum)
    optimizer = LARS(optimizer)
    return optimizer

