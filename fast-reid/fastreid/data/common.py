# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import cv2, lmdb
from PIL import Image
from torch.utils.data import Dataset

from .data_utils import read_image


class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel

        pid_set = set()
        cam_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path, pid, camid = self.img_items[index]
        img = read_image(img_path)
        if self.transform is not None: img = self.transform(img)
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]
        return {
            "images": img,
            "targets": pid,
            "camids": camid,
            "img_paths": img_path,
        }

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)


class LMDBDataset(Dataset):
    """ Person dataset in LMDB format """
    def __init__(self, dataset, transform=None, relabel=False):
        self.dataset   = dataset
        self.transform = transform
        self.relabel   = relabel

        self.img_items = dataset.train
        self.pids      = dataset.pids

        self._num_classes = len(self.pids)
        self._num_cameras = 1

        self.lmdb_path = dataset.lmdb_path
        self.env = None

    def _init_lmdb(self):
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, 
                        readahead=False, meminit=False)

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        if self.env is None:
            self._init_lmdb()
        img_key, pid, camid = self.img_items[index]
        
        with self.env.begin(write=False) as txn:
            buf = txn.get(img_key.encode('ascii'))

        img_flat = np.frombuffer(buf, dtype=np.uint8)
        # im is in BGR order
        im = cv2.imdecode(img_flat, cv2.IMREAD_COLOR)
        im = Image.fromarray(im[:, :, ::-1])

        if self.transform is not None:
            im = self.transform(im)

        return {'images': im, 'targets': pid, 
                'camids': 0,  'img_paths': img_key}

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_cameras(self):
        return self._num_cameras
    