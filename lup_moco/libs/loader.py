#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-15 14:40:58
# @Author  : Dengpan Fu (t-defu@microsoft.com)

import torch
from torch.utils.data import Sampler
import torch.distributed as dist

class IterDistributedSampler(Sampler):
    def __init__(self, dataset, total_epochs, batch_size=64, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        assert total_epochs >= 1
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle

        self.effect_bs = self.num_replicas * self.batch_size
        self.total_size = len(dataset) // self.effect_bs * self.effect_bs

        self.num_samples = self.total_size // self.num_replicas

        self.indices = []
        for ep in range(total_epochs):
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(ep)
            if self.shuffle:
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
                indices = indices[:self.total_size]
            else:
                indices = list(range(len(self.dataset)))
            indices = indices[:self.total_size]

             # subsample
            self.indices.extend(indices[self.rank:self.total_size:self.num_replicas])

        self.num_samples = self.num_samples * total_epochs

        # print(f'ebs={self.effect_bs}, nr={self.num_replicas}, bs={self.batch_size}' 
        #      +f' ts={self.total_size}, ns={self.num_samples}, tep={total_epochs}')
        assert len(self.indices) == self.num_samples


    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class InferenceSampler(Sampler):
    """ Sampler used for inference in DDP. 
        This sampler produces different number of samples for different workers.
    """
    def __init__(self, data_num: int, num_replicas: int = None, rank: int = None):
        self.data_num = data_num
        assert(data_num > 0), "dataset is empty!"
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.common_size = (self.data_num + num_replicas - 1) // num_replicas

        self.begin = self.common_size * rank
        self.end   = min(self.common_size * (rank + 1), self.data_num)
        self.indices = range(self.begin, self.end)
        # print(f"rank={rank}, dn={data_num}, common_size={self.common_size}, \
        #         B-E={self.begin}-{self.end}, len_idx={len(self.indices)}")

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)