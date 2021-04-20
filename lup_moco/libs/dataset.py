#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-09-10 16:46:43
# @Author  : Dengpan Fu (t-defu@microsoft.com)

import os
import numpy as np
import lmdb, pickle, cv2
from PIL import Image
from torch.utils.data import Dataset
import re, time
from glob import glob

class PersonDataset(Dataset):
    def __init__(self, data_dir, key_path, transform=None):
        super(PersonDataset, self).__init__()
        self.data_dir = data_dir
        self.key_path = key_path
        self.transform = transform
        if not os.path.exists(self.data_dir):
            raise IOError('dataset dir: {} is non-exist'.format(
                            self.data_dir))
        self.load_dataset_infos()
        self.env = None

    def load_dataset_infos(self):
        if not os.path.exists(self.key_path):
            raise IOError('key info file: {} is non-exist'.format(
                            self.key_path))
        with open(self.key_path, 'rb') as f:
            data = pickle.load(f)
        self.keys = data['keys']
        if 'pids' in data:
            self.labels = np.array(data['pids'], np.int)
        elif 'vids' in data:
            self.labels = np.array(data['vids'], np.int)
        else:
            self.labels = np.zeros(len(self.keys), np.int)
        self.num_cls = len(set(self.labels))

    def __len__(self):
        return len(self.keys)

    def _init_lmdb(self):
        self.env = lmdb.open(self.data_dir, readonly=True, lock=False, 
                        readahead=False, meminit=False)

    def __getitem__(self, index):
        if self.env is None:
            self._init_lmdb()

        key = self.keys[index]
        label = self.labels[index]

        with self.env.begin(write=False) as txn:
            buf = txn.get(key.encode('ascii'))
        img_flat = np.frombuffer(buf, dtype=np.uint8)
        im = cv2.imdecode(img_flat, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            im = Image.fromarray(im)
            im = self.transform(im)
        else:
            im = im / 255.

        return im, label

    def __repr__(self):
        format_string  = self.__class__.__name__ + '(num_imgs='
        format_string += '{:d}, num_cls={:d})'.format(len(self), self.num_cls)
        return format_string

class PersonDatasetFM(PersonDataset):
    def __init__(self, data_dir, key_path, transform=None, strong_transform=None):
        super(PersonDatasetFM, self).__init__(data_dir, key_path, transform)
        self.strong_transform = strong_transform

    def __getitem__(self, index):
        if self.env is None:
            self._init_lmdb()

        key = self.keys[index]
        label = self.labels[index]

        with self.env.begin(write=False) as txn:
            buf = txn.get(key.encode('ascii'))
        img_flat = np.frombuffer(buf, dtype=np.uint8)
        im = cv2.imdecode(img_flat, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            im = Image.fromarray(im)
            im1 = self.transform(im)
            if self.strong_transform is not None:
                im2 = self.strong_transform(im)
            else:
                im2 = im / 255.
        else:
            im1 = im / 255.
            if self.strong_transform is not None:
                im2 = self.strong_transform(im)
            else:
                im2 = im / 255.

        return im1, im2, label

class CMDM(Dataset):
    NAME_DICTs = {'msmt': ('msmt17', ), 
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
    EXTs = ('*.jpg', '*.png', '*.jpeg', '*.bmp', '*.ppm')
    def __init__(self, root='data', data_name='duke', mode='train', transform=None):
        self.base_root   = root
        self.data_name   = data_name
        self.mode        = mode.lower()
        self.transform   = transform
        self.data_dir    = os.path.join(self.base_root, *self.NAME_DICTs[data_name])
        self.train_dir   = os.path.join(self.data_dir, 'bounding_box_train')
        self.query_dir   = os.path.join(self.data_dir, 'query')
        self.gallery_dir = os.path.join(self.data_dir, 'bounding_box_test')
        assert(self.mode in ('train', 'val', 'test')), 'Unknown data mode={}'.format(self.mode)
        self.is_training = self.mode == 'train'
        if self.is_training:
            self.items, self.num_pids, self.num_cams = self.preprocess(self.train_dir)
        else:
            self.query, self.num_q_pids, self.num_q_cams = self.preprocess(
                                            self.query_dir, False, is_query=True)
            self.gallery, self.num_g_pids, self.num_g_cams = self.preprocess(
                                            self.gallery_dir, False, is_query=False)
            self.items = self.query + self.gallery
            self.num_pids = max(self.num_q_pids, self.num_g_pids)
            self.num_cams = max(self.num_q_cams, self.num_g_cams)

    def preprocess(self, path, relabel=True, is_query=None):
        pattern = re.compile(r'([-\d]+)_c(\d+)')
        all_pids, all_cids = {}, {}
        ret, fpaths = [], []
        for ext in self.EXTs:
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
            if is_query is None:
                ret.append([fpath, pid, cid])
            else:
                flag = 1 if is_query else 0
                ret.append([fpath, pid, cid, flag])
        return sorted(ret), int(len(all_pids)), int(len(all_cids))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        if self.is_training:
            fpath, pid, cid = self.items[index]
        else:
            fpath, pid, cid, is_query = self.items[index]

        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        fname = os.path.basename(fpath)

        if self.is_training:
            return img, pid, cid, fname
        else:
            return img, pid, cid, is_query, fname

    def __repr__(self):
        format_string  = self.__class__.__name__ + '('
        format_string += 'data_name={}, mode={}), '.format(self.data_name, self.mode)
        if self.is_training:
            format_string += 'train[img|pid|cam]=[{}|{}|{}])'.format(len(self), 
                                                    self.num_pids, self.num_cams)
        else:
            format_string += 'query[img|qid|qcam]=[{}|{}|{}], '.format(len(self.query), 
                                                    self.num_q_pids, self.num_q_cams)
            format_string += 'gallery[img|gid|gcam]=[{}|{}|{}], '.format(len(self.gallery), 
                                                    self.num_g_pids, self.num_g_cams)
            format_string += 'total[img|pid|cam]=[{}|{}|{}])'.format(len(self), 
                                                    self.num_pids, self.num_cams)
        return format_string

if __name__ == '__main__':
    # train_dir = '/home/store/dengpanfu/LUP/lmdb_200_20/lmdb'
    # info_path = '/home/store/dengpanfu/LUP/keys.pkl'
    # for data in dataset:
    from transform import get_reid_test_transformer
    transformer = get_reid_test_transformer(mean_type='lup_v0')
    root = 'D:\\Datasets\\ReID'
    data = CMDM(root=root, data_name='market', mode='val', transform=transformer)
