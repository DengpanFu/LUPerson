#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-12-26 23:09:49
# @Author  : Dengpan Fu (fdpan@mail.ustc.edu.cn)

import os
import numpy as np
import argparse
from tqdm import tqdm


def download_one_video(vid, save_dir, save_name=None):
    url = "https://www.youtube.com/watch?v=" + vid
    if save_name is None:
        save_name = vid
    fpath = os.path.join(save_dir, save_name + '".%(ext)s"')
    if os.path.exists(fpath.replace('".%(ext)s"', '.mp4')):
        return
    cmd = 'youtube-dl -o {:s} -f "bestvideo[ext=mp4][height<=?720][filesize<=500M]/best[height<=?720][filesize<=500M]" {:s}'.format(fpath, url)
    os.system(cmd)


def parse_args():
    """ Parse input arguments """
    parser = argparse.ArgumentParser(description='download luperson raw videos')
    parser.add_argument('-f',  '--vid_name_file',  type=str,  default='vname.txt')
    parser.add_argument('-s',  '--save_dir',       type=str,  default='videos')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if not os.path.isfile(args.vid_name_file):
        raise IOError(f'video name records file: {args.vid_name_file} does not exist')
    with open(args.vid_name_file, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    for item in tqdm(lines):
        country, city, vid = item.split('+')
        save_dir = os.path.join(save_root, country, city)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        download_one_video(vid, save_dir, save_name=item)
