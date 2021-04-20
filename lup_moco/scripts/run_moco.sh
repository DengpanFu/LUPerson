#!/bin/bash

DIR="/home/dengpanfu/data"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python lup_moco.py \
    --data_path "${DIR}/LUP/lmdbs/lmdb" \
    --info_path "${DIR}/LUP/lmdbs/keys.pkl" \
    --eval_path "${DIR}/reid" \
    --eval_name "market" \
    -a resnet50 \
    --lr 0.3 \
    --optimizer "SGD" \
    -j 32 \
    --batch-size 2560 \
    --dist-url 'tcp://localhost:13701' \
    --multiprocessing-distributed 1 \
    --world-size 1 \
    --rank 0 \
    --T 0.1 \
    --aug_type 'ori-cj+sre' \
    --cos 1 \
    --snap_dir 'snapshots/lup/moco' \
    --log_dir 'logs/lup/moco' \
    --mix 1 \
    --auto_resume 1 \
    --save_freq 20 \
    --print-freq 10 \
    --epochs 200 \
    --mean_type "lup" \
    --eval_freq -1
    