#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-10-13 14:43:58
# @Author  : Dengpan Fu (t-defu@microsoft.com)

import os
import os.path as osp
import re, tarfile, shutil
import warnings
import numpy as np
from glob import glob

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY

NAME_DICT = {
    'msmt': ('msmt17', ), 
    'msmt17': ('msmt17', ), 
    'duke': ('duke', ), 
    'dukemtmc': ('duke', ), 
    'market': ('market1501', ), 
    'market1501': ('market1501', ), 
    'cuhk': ('cuhk03_np', 'labeled'), 
    'cuhk03': ('cuhk03_np', 'labeled'), 
    'cuhk-lab': ('cuhk03_np', 'labeled'), 
    'cuhk03-lab': ('cuhk03_np','labeled'), 
    'cuhk-det': ('cuhk03_np', 'detected'), 
    'cuhk03-det': ('cuhk03_np', 'detected'), 
}
EXTS = ['*.jpg', '*.png', '*.jpeg', '*.bmp', '*.ppm']

@DATASET_REGISTRY.register()
class CMDM(ImageDataset):
    """ Dataset class for CUHK03-NP(C), Market1501(M), Duke(D), MSMT(M)
    with specical attributes.
    """
    def __init__(self, root='datasets', data_name='duke', split_mode='ori', 
        split_ratio=1.0, repeat_ratio=1.0, tgz_data=None, **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.base_root = root
        self.data_name = data_name
        self.split_mode = split_mode
        self.split_ratio = split_ratio
        self.repeat_ratio = repeat_ratio
        self.tgz_data = tgz_data
        self.dataset_dir = osp.join(self.base_root, *NAME_DICT[self.data_name])

        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self.used_id_files = os.path.join(self.dataset_dir, 'few_ids', 
                                f'used_ids_{self.split_ratio:.02f}.txt')
        self.used_im_files = os.path.join(self.dataset_dir, 'few_ims', 
                                f'used_ims_{self.split_ratio:.02f}.txt')

        self.prepare_data()

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.preprocess(self.train_dir, is_train=True)
        query = self.preprocess(self.query_dir, relabel=False)
        gallery = self.preprocess(self.gallery_dir, relabel=False)

        super(CMDM, self).__init__(train, query, gallery, **kwargs)


    def check_data_folder(self):
        has_train_dir = os.path.isdir(self.train_dir)
        has_query_dir = os.path.isdir(self.query_dir)
        has_gallery_dir = os.path.isdir(self.gallery_dir)
        has_used_im_file = os.path.isfile(self.used_im_files)
        has_used_id_file = os.path.isfile(self.used_id_files)
        valid = (has_train_dir & has_gallery_dir & has_query_dir & 
                 has_used_id_file & has_used_im_file)
        return valid

    def prepare_data(self):
        if self.check_data_folder():
            print(f"Data[{self.data_name}] is prepared at {self.dataset_dir}")
            return True
        else:
            print(f'Preparing [{self.data_name}] ...')
            if self.tgz_data is None or not os.path.exists(self.tgz_data):
                print(f"No tgz[{self.tgz_data}] data provided for data preparing")
                raise IOError(f'No vaild tgz[{self.tgz_data}] data provided')
            if not os.path.exists(self.base_root):
                os.makedirs(self.base_root)
            tgz_dst = os.path.join(self.base_root, os.path.basename(self.tgz_data))
            if not os.path.isfile(tgz_dst):
                print(f"Coping {self.tgz_data} to {tgz_dst}")
                shutil.copy(self.tgz_data, tgz_dst)
            with tarfile.open(tgz_dst) as tar:
                print(f"Extracting {tgz_dst}")
                tar.extractall(self.base_root)
            return True

    def preprocess(self, path, relabel=True, is_train=False):
        if is_train and self.split_mode == 'id':
            return self.process_train_id_mode(path)
        if is_train and self.split_mode == 'im':
            return self.process_train_im_mode(path)

        pattern = re.compile(r'([-\d]+)_c(\d+)')
        all_pids, all_cids = {}, {}
        ret, fpaths = [], []
        for ext in EXTS:
            fpaths.extend(glob(os.path.join(path, ext)))
        fpaths = sorted(fpaths)
        for fpath in fpaths:
            fname = os.path.basename(fpath)
            pid, cid = map(int, pattern.search(fname).groups())
            if pid == -1: continue
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            if cid not in all_cids:
                all_cids[cid] = cid
            pid = all_pids[pid]
            cid -= 1
            ret.append((fpath, pid, cid))
        return ret

    def process_train_id_mode(self, path, relabel=True):
        pattern = re.compile(r'([-\d]+)_c(\d+)')
        fpaths = []
        for ext in EXTS:
            fpaths.extend(glob(os.path.join(path, ext)))
        fpaths = sorted(fpaths)

        pid_container = set()
        for fpath in fpaths:
            pid, _ = map(int, pattern.search(fpath).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)

        ids = []
        if os.path.isfile(self.used_id_files):
            with open(self.used_id_files, 'r') as f:
                lines = f.readlines()
            ids = [int(line.strip()) for line in lines if line.strip()]
            if not len(ids) == int(len(pid_container) * self.split_ratio):
                ids = []
            else:
                print(f"Loading split info with [mode={self.split_mode}]" 
                      f" from {self.used_id_files}")
        if len(ids) < 1:
            num = int(len(pid_container) * self.split_ratio)
            choose_ids = np.random.choice(list(pid_container), num, replace=False)
            ids = sorted(choose_ids)
            with open(self.used_id_files, 'w') as f:
                for iidd in ids:
                    f.write(f'{iidd:d} \n')
            print(f"Saving split info to {self.used_id_files}")

        pid2label = {pid: label for label, pid in enumerate(ids)}
        all_pids, all_cids = [], []

        dataset = []
        for fpath in fpaths:
            pid, camid = map(int, pattern.search(fpath).groups())
            if pid == -1: continue  # junk images are just ignored
            camid -= 1  # index starts from 0
            if not pid in ids: continue
            if relabel: pid = pid2label[pid]
            if not pid in all_pids: all_pids.append(pid)
            if not camid in all_cids: all_cids.append(camid)
            dataset.append((fpath, pid, camid))

        return sorted(dataset)

    def process_train_im_mode(self, path, relabel=True):
        pattern = re.compile(r'([-\d]+)_c(\d+)')
        fpaths = []
        for ext in EXTS:
            fpaths.extend(glob(os.path.join(path, ext)))
        fpaths = sorted(fpaths)

        dataset = []
        if os.path.isfile(self.used_im_files):
            with open(self.used_im_files, 'r') as f:
                lines = f.readlines()
            if len(lines) > 2:
                for line in lines:
                    if line:
                        name, pid, camid = line.strip().split('; ')
                        dataset.append([name, int(pid), int(camid)])
                print(f"Loading split data with [mode={self.split_mode}]" 
                      f" from {self.used_im_files}")
        if len(dataset) < 2:
            dataset = []
            pid_dict = {}
            for fpath in fpaths:
                pid, _ = map(int, pattern.search(fpath).groups())
                if pid == -1: continue  # junk images are just ignored
                if pid in pid_dict:
                    pid_dict[pid].append(os.path.basename(fpath))
                else:
                    pid_dict[pid] = [os.path.basename(fpath)]
            pid2label = {pid: label for label, pid in enumerate(sorted(pid_dict.keys()))}
            for key, value in pid_dict.items():
                num = int(max(np.round(len(value) * self.split_ratio), 1))
                chooses = np.random.choice(value, num, replace=False)
                pid = key
                if relabel: pid = pid2label[pid]
                for choose in chooses:
                    name = str(choose)
                    _, camid = map(int, pattern.search(name).groups())
                    dataset.append([name, pid, camid -1])
            with open(self.used_im_files, 'w') as f:
                for item in dataset:
                    f.write(f"{item[0]:s}; {item[1]:d}; {item[2]:d} \n")
                print(f"Saving split info to {self.used_im_files}")

        all_pids, all_cids = [], []
        for item in dataset:
            item[0] = os.path.join(path, item[0])
            if not item[1] in all_pids: all_pids.append(item[1])
            if not item[2] in all_cids: all_cids.append(item[2])

        nd = []
        if self.repeat_ratio > 1:
            nd = dataset * int(self.repeat_ratio)
            remain = self.repeat_ratio - int(self.repeat_ratio)
            if remain > 0:
                end = int(remain * len(dataset))
                nd.extend(dataset[:end])
            dataset = nd

        return sorted(dataset)
