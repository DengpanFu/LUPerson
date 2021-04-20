#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-27 11:10:05
# @Author  : Dengpan Fu (t-defu@microsoft.com)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

@torch.no_grad()
def pad_to_max_size(tensor):
    local_size = torch.tensor([tensor.shape[0]], dtype=torch.int64, device=tensor.device)
    size_list = [torch.zeros([1], dtype=torch.int64, device=tensor.device) 
                 for _ in range(dist.get_world_size())]
    dist.all_gather(size_list, local_size, async_op=False)
    max_size = torch.cat(size_list, dim=0).max().item()
    if local_size < max_size:
        padding = torch.zeros((max_size - local_size, *tensor.shape[1:]), 
                dtype=tensor.dtype, device=tensor.device)
        flag = torch.cat([torch.ones(local_size, dtype=torch.uint8, device=tensor.device), 
                torch.zeros(max_size-local_size, dtype=torch.uint8, device=tensor.device)], dim=0)
        tensor = torch.cat((tensor, padding), dim=0)
    else:
        flag = torch.ones(local_size, dtype=torch.uint8, device=tensor.device)
    return tensor, flag

@torch.no_grad()
def gather_tensors(tensor):
    """ Gather tensors from different workers with different shape
    tensors on different devices must have the same data dims except the bacth_num
    """
    tensor, flag = pad_to_max_size(tensor)
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    flags_gather = [torch.zeros_like(flag) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    dist.all_gather(flags_gather, flag, async_op=False)

    outputs = torch.cat(tensors_gather, dim=0)
    flags = torch.cat(flags_gather, dim=0)
    output = outputs[flags > 0]
    return output


def tensor2numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return x

def numpy2tensor(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if not isinstance(x, torch.Tensor):
        x = torch.Tensor(x)
    return x

def print_scores(mAP, cmc_scores, p_str=''):
    if p_str:
        print('{:s}'.format(p_str), end='')
    print(('[mAP: {:5.2%}], [cmc1: {:5.2%}], [cmc5: {:5.2%}],' 
        ' [cmc10: {:5.2%}]').format(mAP, *cmc_scores[[0, 4, 9]]))

def pairwise_distance(x, y):
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(1).unsqueeze(1).expand(m, n) + \
           torch.pow(y, 2).sum(1).unsqueeze(1).expand(n, m).t()
    # dist.addmm_(1, -2, x, y.t())
    dist.addmm_(x, y.t(), beta=1., alpha=-2.)
    return dist

def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.where(mask)[0]

    cmc[rows_good[0]:] = 1
    d_recall = 1.0/ngood
    precision = (np.arange(len(rows_good), dtype=np.float) + 1) / (rows_good + 1)
    if rows_good[0] == 0:
        old_precision = np.ones(len(rows_good))
        old_precision[1:] = np.arange(1, len(rows_good), dtype=np.float) / rows_good[1:]
    else:
        old_precision = np.arange(len(rows_good), dtype=np.float) / rows_good
    ap = np.sum((precision + old_precision) / 2. * d_recall)

    return ap, cmc

def compute_reid_score(q_feats, q_pids, q_cams, g_feats, g_pids, g_cams):
    dist = pairwise_distance(q_feats, g_feats)
    dist = tensor2numpy(dist)
    q_pids, q_cams = tensor2numpy(q_pids), tensor2numpy(q_cams)
    g_pids, g_cams = tensor2numpy(g_pids), tensor2numpy(g_cams)
    cmc = torch.IntTensor(len(g_pids)).zero_()
    ap = 0.0
    aps = []
    for i in range(len(q_pids)):
        index = dist[i].argsort()
        ql, qc, gl, gc = q_pids[i], q_cams[i], g_pids, g_cams

        good_index = np.where((gl==ql) & (gc!=qc))[0]
        junk_index = np.where(((gl==ql) & (gc==qc)) | (gl==-1))[0]

        ap_tmp, cmc_tmp = compute_mAP(index, good_index, junk_index)
        aps.append(ap_tmp)
        if cmc_tmp[0]==-1:
            continue
        cmc = cmc + cmc_tmp
        ap += ap_tmp

    cmc = cmc.float()
    cmc = cmc / len(q_pids) #average cmc
    ap = ap / len(q_pids)
    return ap, cmc
