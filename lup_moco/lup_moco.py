#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-27 15:38:58
# @Author  : Dengpan Fu (t-defu@microsoft.com)

import argparse
import builtins
import math
import os, sys
import random
import shutil
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from libs.dataset import PersonDataset, CMDM
from libs.transform import get_lup_transformer, get_reid_test_transformer
from libs.moco_builder import MoCo
from libs.loader import IterDistributedSampler, InferenceSampler
from libs.lars import get_lars_optimizer
from libs.utils import gather_tensors, compute_reid_score, print_scores

try:
    from torch.cuda.amp import autocast, GradScaler
    MIX_P = True
except Exception as exc:
    MIX_P = False

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# dataset settings
parser.add_argument('--data_path', metavar='DIR', help='path to dataset')
parser.add_argument('--info_path', metavar='DIR', help='path to dataset info')
parser.add_argument('--eval_path', metavar='DIR', help='path to evaluation data')
parser.add_argument('--eval_name', type=str, default='market', help='evaluation data name')

# training settings
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=model_names, 
                    help='model arch: ' + '|'.join(model_names) + ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save_freq', default=10, type=int, metavar='N', 
                    help='save checkpoint frequency (default: 10)')
parser.add_argument('--eval_freq', default=10, type=int, metavar='N', 
                    help='evaluation frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--auto_resume', type=int, default=1)
parser.add_argument('--mix', type=int, default=1)
parser.add_argument('--workdir', default="", help="The working directory.")
parser.add_argument('--snap_dir', default='snapshots', type=str)
parser.add_argument('--log_dir', default='logs', type=str)
parser.add_argument('--eval_only', default=0, type=int, help='if only do evaluate')

# optimizer settings
parser.add_argument('--optimizer', default='LARS', type=str,
                    help='one of LARS|SGD')
parser.add_argument('--warmup_epochs', default=0, type=int, metavar='N',
                    help='number of epochs to warmup learning rate (default: 10)')
parser.add_argument('--lr', '--learning-rate', default=0.3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default='60,80', type=str,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

# distributed settings
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', type=int, default=1,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# SimCLR settings
parser.add_argument('--moco_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--T', default=0.1, type=float,
                    help='softmax temperature (default: 0.1)')
parser.add_argument('--moco_k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco_m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--mlp', type=int, default=1, help='use mlp head')

parser.add_argument('--cos', default=1, type=int, 
                    help='use cosine lr schedule')
parser.add_argument('--aug_type', default='ori', type=str)
parser.add_argument('--mean_type', default='lup', type=str)


def main():
    args = parser.parse_args()

    args.auto_resume = True if args.auto_resume >= 1 else False
    args.mix = True if args.mix >= 1 else False
    args.eval_only = True if args.eval_only >= 1 else False
    args.cos = True if args.cos >= 1 else False
    args.mlp = True if args.mlp >= 1 else False
    args.schedule = [int(x) for x in args.schedule.split(',')]

    if args.workdir:
        args.snap_dir  = os.path.join(args.workdir, args.snap_dir)
        args.log_dir   = os.path.join(args.workdir, args.log_dir)
        args.data_path = os.path.join(args.workdir, args.data_path)
        args.info_path = os.path.join(args.workdir, args.info_path)
        args.eval_path = os.path.join(args.workdir, args.eval_path)

    cudnn.benchmark = True
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.mix = MIX_P and args.mix

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        # builtins.print = print_pass
    else:
        if not os.path.exists(args.snap_dir):
            os.makedirs(args.snap_dir)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        txt_log_name = "train_log_" + time.strftime("%Y-%m-%d_%H-%M-%S", 
                        time.gmtime()) + '.txt'
        sys.stdout = Logger(os.path.join(args.log_dir, txt_log_name))

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.rank == 0:
        print(f"Training configs:")
        print(f"{args} \n")

    # Data loading code
    train_transformer = get_lup_transformer(args.aug_type, args.mean_type)
    val_transformer   = get_reid_test_transformer(args.mean_type)

    if args.rank == 0:
        print(f"Training Transformers: \n {str(train_transformer)}")
        print(f"Testing  Transformers: \n {str(val_transformer)} \n")

    train_dataset = PersonDataset(args.data_path, args.info_path, train_transformer)
    val_dataset   = CMDM(args.eval_path, args.eval_name, 'test', val_transformer)
    if args.rank == 0:
        print(f"Training set: {train_dataset}")
        print(f"Testing  set: {val_dataset}")

    # create model
    if args.rank == 0:
        print("Creating model '{}'".format(args.arch))
    model = MoCo(models.__dict__[args.arch], dim=args.moco_dim, 
                 K=args.moco_k, m=args.moco_m, T=args.T, mlp=args.mlp)
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            if args.optimizer == 'LARS': 
                args.lr_mult = args.batch_size / 256
            else:
                args.lr_mult = 1.
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], 
                                                              broadcast_buffers=False)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.optimizer == 'LARS':
        if args.rank == 0: print("Use LARS as optimizer.")
        optimizer = get_lars_optimizer(model, 
                                args.lr_mult * args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    else:
        if args.rank == 0: print("Use SGD as optimizer.")
        optimizer = torch.optim.SGD(model.parameters(), 
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.auto_resume:
        # 'ckpt_{:04d}.pth'
        # if have ckpt_latest.pth, load it. Or find the record with max epoch
        latest_checkpoint = os.path.join(args.snap_dir, 'ckpt_latest.pth')
        if os.path.isfile(latest_checkpoint):
            args.resume = latest_checkpoint
        else:
            snaps = os.listdir(args.snap_dir)
            snaps = [x for x in snaps if x.endswith('.pth')]
            if len(snaps) > 0:
                max_pre_epoch = 0
                for snap in snaps:
                    tmp_epoch = int(snap[:-4].split('_')[1])
                    if tmp_epoch > max_pre_epoch:
                        max_pre_epoch = tmp_epoch
                latest_checkpoint = 'ckpt_{:04d}.pth'.format(max_pre_epoch)
                args.resume = os.path.join(args.snap_dir, latest_checkpoint)
            else:
                print(f'=>rank[{args.rank}] no previous snapshots eixst for auto-resume.')

    if args.resume:
        if os.path.isfile(args.resume):
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=>rank[{}] loaded checkpoint '{}' (epoch {})"
                  .format(args.rank, args.resume, checkpoint['epoch']))
        else:
            print("=>rank[{}] no checkpoint found at '{}'".format(args.rank, args.resume))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = InferenceSampler(len(val_dataset))
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                shuffle=(train_sampler is None), num_workers=args.workers, 
                                pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, pin_memory=True, 
                        num_workers=args.workers, sampler=val_sampler, drop_last=False)

    if args.eval_only:
        mAP, cmc = evaluate(val_loader, model, args)
        if args.rank == 0: print_scores(mAP, cmc)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank == 0):
            print("PROGRESS: {:02d}%".format(int(epoch / args.epochs * 100)))

        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        epoch_start = time.time()
        loss, acc1 = train(train_loader, model, criterion, optimizer, epoch, args)
        is_save = args.multiprocessing_distributed and args.rank == 0 and args.save_freq > 0
        is_save |= not args.multiprocessing_distributed

        time_str = time.strftime("%Y/%m/%d-%H:%M:%S", time.gmtime())
        if args.rank == 0:
            print(f"{time_str}: Epoch[{epoch}]; Loss={loss:.3f}; Acc={acc1:.3f}; " + 
                  f"Time={(time.time() - epoch_start):.3f}; " + 
                  f"LR={optimizer.param_groups[0]['lr']:.2e}")

        if args.eval_freq > 0 and (epoch % args.eval_freq == 0 or epoch == args.epochs - 1):
            eval_time = time.time()
            if args.rank == 0: print(f"Evaluating Epoch[{epoch}] ...")
            mAP, cmc = evaluate(val_loader, model, args)
            if args.rank == 0: 
                print_scores(mAP, cmc, p_str=f"Epoch[{epoch}]: ")
                print(f'Evaluation using {time.time() - eval_time:.2f}s.')

        if is_save:
            latest = os.path.join(args.snap_dir, 'ckpt_latest.pth')
            filename = os.path.join(args.snap_dir, 'ckpt_{:04d}.pth'.format(epoch))
            save_cur = epoch % args.save_freq == 0 or epoch == args.epochs - 1
            print(f'Saving checkpoint to {filename if save_cur else latest} ', end='')
            save_time = time.time()
            save_checkpoint({'epoch': epoch + 1, 'arch': args.arch, 
                'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, 
                save_cur=save_cur, cur_name=filename, lastest_name=latest)
            print(f'Using {time.time() - save_time:.2f}s.')


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('T', ':.3f')
    data_time = AverageMeter('DT', ':.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':.2f')
    top5 = AverageMeter('Acc@5', ':.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    if args.print_freq > 1:
        print_freq = (len(train_loader) + args.print_freq - 1) // args.print_freq
    else:
        print_freq = - args.print_freq // 1
    print_freq = max(print_freq, 1)

    # switch to train mode
    model.train()

    end = time.time()

    scaler = GradScaler() if args.mix else None
    
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        if args.mix:
            with autocast():
                output, target = model(im_q=images[0], im_k=images[1])
                loss = criterion(output, target)
        else:
            output, target = model(im_q=images[0], im_k=images[1])
            loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.mix:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.rank == 0 and i % print_freq == 0:
            progress.display(i)

    return losses.avg, top1.avg


def evaluate(val_loader, model, args):
    model.eval()
    feats, pids, cams, flags = [], [], [], []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            img, pid, cam, flag = data[0:4]
            feat = model(img.cuda(args.gpu, non_blocking=True))
            feats.append(feat)
            pids.append(pid.cuda(args.gpu, non_blocking=True))
            cams.append(cam.cuda(args.gpu, non_blocking=True))
            flags.append(flag.cuda(args.gpu, non_blocking=True))
    feats = torch.cat(feats, dim=0)
    pids  = torch.cat(pids, dim=0)
    cams  = torch.cat(cams, dim=0)
    flags = torch.cat(flags, dim=0)
    dist.barrier()
    feats = gather_tensors(feats)
    pids  = gather_tensors(pids)
    cams  = gather_tensors(cams)
    flags = gather_tensors(flags)
    if args.rank == 0:
        q_idx = flags > 0
        g_idx = flags < 1
        mAP, cmc = compute_reid_score(feats[q_idx], pids[q_idx], cams[q_idx], 
                                      feats[g_idx], pids[g_idx], cams[g_idx])
        return mAP, cmc
    else:
        return 0, 0


def save_checkpoint(state, save_cur, cur_name='ckpt.pth', lastest_name='latest.pth'):
    torch.save(state, lastest_name)
    if save_cur:
        shutil.copyfile(lastest_name, cur_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}[{val' + self.fmt + '}|{avg' + self.fmt + '}]'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, iters_per_epoch, meters, iters=None, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(iters_per_epoch)
        if not iters is None:
            self.iters_fmtstr = self._get_batch_fmtstr(iters)
        else:
            self.iters_fmtstr = None
        self.iters_per_epoch = iters_per_epoch
        self.iters = iters
        self.meters = meters
        self.prefix = prefix

    def display(self, i):
        entries = self.prefix + self.batch_fmtstr.format(i % self.iters_per_epoch)
        if not self.iters_fmtstr is None:
            entries += self.iters_fmtstr.format(i)
        entries = [entries]
        entries += [str(meter) for meter in self.meters]
        print('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            log_dir = os.path.dirname(fpath)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr * args.lr_mult
    if epoch < args.warmup_epochs:
        # warm up
        lr = args.lr + (args.lr * args.lr_mult - args.lr) / args.warmup_epochs * epoch
    elif args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
