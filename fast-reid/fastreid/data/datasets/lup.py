#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-12 17:08:58
# @Author  : Dengpan Fu (t-defu@microsoft.com)

import os
import numpy as np
import pickle, lmdb

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class LUP(ImageDataset):
    """ Dataset class for Large Scale Unlabeled data in LMDB format
    """
    dir_name = 'lup'
    def __init__(self, root='datasets', sub_name='lmdb_300_30', **kwargs):
        self.base_root = root
        self.sub_name  = sub_name
        self.data_root = os.path.join(self.base_root, self.dir_name, self.sub_name)
        self.lmdb_path = os.path.join(self.data_root, 'lmdb')
        self.key_path  = os.path.join(self.data_root, 'keys.pkl')

        required_files = [self.data_root, self.lmdb_path, self.key_path]
        self.check_before_run(required_files)

        with open(self.key_path, 'rb') as f:
            data = pickle.load(f)

        self.train = []
        self.pids = set()
        for key, pid in zip(data['keys'], data['pids']):
            self.train.append([key, pid, 0])
            self.pids.add(pid)
        self.pids = sorted(list(self.pids))

        super(LUP, self).__init__(self.train, [], [], **kwargs)


    def get_num_pids(self, data):
        return len(self.pids)

    def get_num_cams(self, data):
        return 1

    def parse_data(self, data):
        return len(self.pids), 1

    def show_test(self):
        raise Exception('LUPerson dataset has no test split currently.')