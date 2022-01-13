#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-12-27 01:08:30
# @Author  : Dengpan Fu (fdpan@mail.ustc.edu.cn)

import os
import numpy as np
import cv2
import lmdb
import pickle


base_dir = 'luperson'
lmdb_dir = 'lup_lmdb'
keys = []
vnames = []
env = lmdb.open(lmdb_dir, map_size=1e12)
txn = env.begin(write=True)
cnt = 0
countries = sorted(os.listdir(base_dir))
for i, country in enumerate(countries):
    city_dir = os.path.join(base_dir, country)
    citys = sorted(os.listdir(city_dir))
    for j, city in enumerate(citys):
        key_prefix = '{:02d}_{:02d}'.format(i, j)
        vid_dir = os.path.join(city_dir, city)
        vids = sorted(os.listdir(vid_dir))
        for k, vid in enumerate(vids):
            if vid in vnames: continue
            vnames.append(vid)
            key_vid = '{:04d}'.format(k)
            im_dir = os.path.join(vid_dir, vid)
            names = sorted([x for x in os.listdir(im_dir) if x.endswith('jpg')])
            for m, name in enumerate(names):
                if cnt % 2000 == 0:
                    print('[{:3d}|{:3d}] country={:s}, [{:d}|{:d}] city={:s}, ' \
                        '[{:3d}|{:3d}] vid={:s}, [{:d}|{:d}] name={:s}'.format(i, 
                            len(countries), country, j, len(citys), city, k, len(vids), 
                            vid, m, len(names), name))
                key_main = '{:08d}'.format(m)
                key = key_prefix + '_' + key_vid + '_' + key_main
                im_path = os.path.join(im_dir, name)
                with open(im_path, 'rb') as f:
                    im_str = f.read()
                im = np.fromstring(im_str, np.uint8)
                keys.append(key)
                key_byte = key.encode('ascii')
                txn.put(key_byte, im)
                cnt += 1
        txn.commit()
        txn = env.begin(write=True)
txn.commit()
env.close()

with open('keys.pkl', 'wb') as f:
    pickle.dump(keys, f)

# env = lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=False, meminit=False)
# keys = pickle.load(open('keys.pkl', "rb"))
# for key in keys:
#     with env.begin(write=False) as txn:
#         buf = txn.get(key.encode('ascii'))
#     img_flat = np.frombuffer(buf, dtype=np.uint8)
#     im = cv2.imdecode(img_flat, 1)

