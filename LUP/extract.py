#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-12-26 23:43:52
# @Author  : Dengpan Fu (fdpan@mail.ustc.edu.cn)

import os
import numpy as np
import cv2
import pickle
import argparse
from tqdm import tqdm


def extract_one_video_frames(vid_path, det_dir, save_root):
    vname = os.path.basename(vid_path)
    country, city, _ = vname.split('+')
    save_dir = os.path.join(save_root, country, city)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    det_file = os.path.join(det_dir, vname.replace('.mp4', '.pkl'))
    if not os.path.isfile(det_file):
        print(f'****** [Attention] detection file: {det_file} does not exist, skiped !!')
        return
    with open(det_file, 'rb') as f:
        dets = pickle.load(f)
    cap = cv2.VideoCapture(vid_path)
    nfs = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_ids = sorted(dets.keys())
    for frame_id in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
        ret, image = cap.read()
        if not ret:
            print(f'****** [Attention] video: {vid_path} corrupt at frame: {frame_id - 1} !!')
        for det in dets[frame_id]:
            obj_idx_at_ori_image = det[0]
            bbox = det[1]['bbox'].round().astype(np.int)
            img = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            img_name = f'{frame_id:08d}_{obj_idx_at_ori_image:04d}.jpg'
            img_path = os.path.join(save_dir, img_name)
            if not os.path.exists(img_path):
                cv2.imwrite(img_path, img)


def get_all_videos(vid_dir):
    vids = []
    countries = sorted([x for x in os.listdir(vid_dir) if os.path.isdir(os.path.join(vid_dir, x))])
    for country in countries:
        country_dir = os.path.join(vid_dir, country)
        cities = sorted([x for x in os.listdir(country_dir) if os.path.isdir(os.path.join(country_dir, x))])
        for city in cities:
            city_dir = os.path.join(country_dir, city)
            names = sorted([x for x in os.listdir(city_dir) if os.path.isfile(
                            os.path.join(city_dir, x)) and x.endswith('.mp4')])
            vids += [os.path.join(city_dir, x) for x in names]
    return vids


def parse_args():
    """ Parse input arguments """
    parser = argparse.ArgumentParser(description='download luperson raw videos')
    parser.add_argument('-v',  '--vid_dir',  type=str,  default='videos')
    parser.add_argument('-d',  '--det_dir',  type=str,  default='dets')
    parser.add_argument('-s',  '--save_dir', type=str,  default='luperson')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.vid_dir):
        raise IOError(f'raw video directory: {args.vid_dir} does not exist')
    if not os.path.exists(args.det_dir):
        raise IOError(f'detection file directory: {args.det_dir} does not exist')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    vids = get_all_videos(args.vid_dir)
    for vid in tqdm(vids):
        extract_one_video_frames(vid, args.det_dir, args.save_dir)