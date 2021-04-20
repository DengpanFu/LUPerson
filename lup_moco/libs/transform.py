#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-01-15 14:42:59
# @Author  : Dengpan Fu (t-defu@microsoft.com)

import os
import numpy as np
from PIL import ImageFilter, Image
import random, math
import torch
from torchvision import transforms as T

STAT_DICT = {'lupws_300_30': {'mean': [0.3463, 0.3077, 0.3120], 'std': [0.2648, 0.2512, 0.2492]}, 
             'lupws_200_20': {'mean': [0.3452, 0.3070, 0.3114], 'std': [0.2633, 0.2500, 0.2480]}, 
             'imagenet':     {'mean': [0.3430, 0.3080, 0.3220], 'std': [0.2520, 0.2400, 0.2430]}, 
             'lup':          {'mean': [0.3525, 0.3106, 0.3140], 'std': [0.2660, 0.2522, 0.2505]}}

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'sigma={0})'.format(self.sigma)
        return format_string

class RectScale(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)

class RandomSizedRectCrop(object):
    def __init__(self, height, width, al=0.64, ah=1.0, 
        rl=2.0, rh=3.0, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.al, self.ah = al, ah
        self.rl, self.rh = rl, rh
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(self.al, self.ah) * area
            aspect_ratio = random.uniform(self.rl, self.rh)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.width, self.height), self.interpolation)

        # Fallback
        scale = RectScale(self.height, self.width,
                          interpolation=self.interpolation)
        return scale(img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'size=[{:d}, {:d}], '.format(self.height, self.width)
        format_string += 'area_ratio=[{:.3f}, {:.3f}], '.format(self.al, self.ah)
        format_string += 'aspect_ratio=[{:.3f}, {:.3f}]'.format(self.rl, self.rh)
        return format_string

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its 
        pixels. 'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation 
         will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image. {0.4 or 0.2}
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. {[0.4914, 0.4822, 0.4465] or [0.485, 0.456, 0.406]}
    """
    _means = [[0., 0., 0.], [0.485, 0.456, 0.406], [0.4914, 0.4822, 0.4465]]
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean_value=0):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.mean_value = mean_value
        if isinstance(self.mean_value, int):
            assert(self.mean_value in (0, 1, 2))
            self.mean = self._means[self.mean_value]
        elif isinstance(self.mean_value, float):
            assert(0. < self.mean_value < 1.)
            self.mean = [self.mean_value] * 3
        elif isinstance(self.mean_value, (list, tuple, np.ndarray, torch.Tensor)):
            assert(len(self.mean_value) == 3)
            self.mean = mean_value
        else:
            raise ValueError('Wrong mean value={} in RandomErase.'.format(self.mean_value))

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        img_size = img.size
        for attempt in range(100):
            area = img.size(1) * img.size(2)

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size(2) and h < img.size(1):
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'random_prob={0}, '.format(self.probability)
        format_string += 'earsed_area_ratio={0}, '.format([self.sl, self.sh])
        format_string += 'earsed_aspect_ratio=[{:.3f}, {:.3f}], '.format(self.r1, 1/self.r1)
        format_string += 'earsing_value={0})'.format(self.mean)
        return format_string

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

    def __repr__(self):
        format_string  = self.__class__.__name__ + 'with base transform: \n'
        format_string += str(self.base_transform)
        return format_string


def get_reid_train_transformer(mean_type='lup', height=256, width=128, 
    resize_pad_crop=True, with_re=True):
    if not mean_type.lower() in STAT_DICT:
        raise TypeError('Unknown mean value status type: {}'.format(mean_type))
    mean = STAT_DICT[mean_type.lower()]['mean']
    std  = STAT_DICT[mean_type.lower()]['std']
    normalizer = T.Normalize(mean=mean, std=std)
    if resize_pad_crop:
        augmentation = [T.Resize((height, width), interpolation=3), 
                        T.Pad(10), 
                        T.RandomCrop((height, width))]
    else:
        augmentation = [RandomSizedRectCrop(height, width, interpolation=3)]
    augmentation.append(T.RandomHorizontalFlip(p=0.5))
    augmentation.append(T.ToTensor())
    augmentation.append(normalizer)
    if with_re:
        augmentation.append(RandomErasing(probability=0.5))
    transformer = T.Compose(augmentation)
    return transformer


def get_reid_test_transformer(mean_type='lup', height=256, width=128):
    if not mean_type.lower() in STAT_DICT:
        raise TypeError('Unknown mean value status type: {}'.format(mean_type))
    mean = STAT_DICT[mean_type.lower()]['mean']
    std  = STAT_DICT[mean_type.lower()]['std']
    normalizer = T.Normalize(mean=mean, std=std)
    test_transformer = T.Compose([T.Resize((height, width), interpolation=3), 
                                  T.ToTensor(), 
                                  normalizer])
    return test_transformer


def get_lup_transformer(aug_type='moco', mean_type='lup', height=256, width=128, two_crop=True):
    """ LUP dataset is only used for training. """
    if not mean_type.lower() in STAT_DICT:
        raise TypeError('Unknown mean value status type: {}'.format(mean_type))
    mean = STAT_DICT[mean_type.lower()]['mean']
    std  = STAT_DICT[mean_type.lower()]['std']
    normalizer = T.Normalize(mean=mean, std=std)

    if aug_type == 'reid':
        augmentation = [RandomSizedRectCrop(height, width, interpolation=3)]
    else:
        augmentation = [T.RandomResizedCrop((height, width), scale=(0.2, 1.))]
    
    aug_dict = {'ori':           {'cj': 0.8, 'cj-s': 0.4, 'gs': 0.2, 'gb': 0.5,                       }, 
                'ori-cj':        {                        'gs': 0.2, 'gb': 0.5,                       }, 
                'ori-gs':        {'cj': 0.8, 'cj-s': 0.4,            'gb': 0.5,                       }, 
                'ori-gb':        {'cj': 0.8, 'cj-s': 0.4, 'gs': 0.2,                                  }, 
                'ori+re':        {'cj': 0.8, 'cj-s': 0.4, 'gs': 0.2, 'gb': 0.5, 're': 0.5, 're-s': 0.4},
                'ori-cj+re':     {                        'gs': 0.2, 'gb': 0.5, 're': 0.5, 're-s': 0.4}, 
                'ori-gs+re':     {'cj': 0.8, 'cj-s': 0.4,            'gb': 0.5, 're': 0.5, 're-s': 0.4}, 
                'ori-gb+re':     {'cj': 0.8, 'cj-s': 0.4, 'gs': 0.2,            're': 0.5, 're-s': 0.4}, 
                'ori-cj-gb+re':  {                        'gs': 0.2,            're': 0.5, 're-s': 0.4}, 
                'ori+wcj+re':    {'cj': 0.8, 'cj-s': 0.2, 'gs': 0.2, 'gb': 0.5, 're': 0.5, 're-s': 0.4}, 
                'ori+wcj+sre':   {'cj': 0.8, 'cj-s': 0.2, 'gs': 0.2, 'gb': 0.5, 're': 0.5, 're-s': 0.6}, 
                'ori+wwcj+re':   {'cj': 0.8, 'cj-s': 0.1, 'gs': 0.2, 'gb': 0.5, 're': 0.5, 're-s': 0.4}, 

                'ori+sre':       {'cj': 0.8, 'cj-s': 0.4, 'gs': 0.2, 'gb': 0.5, 're': 0.5, 're-s': 0.6}, 
                'ori-cj-gb+sre': {                        'gs': 0.2,            're': 0.5, 're-s': 0.6}, 
                'ori-cj+sre':    {                        'gs': 0.2, 'gb': 0.5, 're': 0.5, 're-s': 0.6}, 
                'ori-cj+wre':    {                        'gs': 0.2, 'gb': 0.5, 're': 0.5, 're-s': 0.2}, 
                'ori-cj+ssre':   {                        'gs': 0.2, 'gb': 0.5, 're': 0.5, 're-s': 0.8}, 

                'moco':          {'cj': 1.0, 'cj-s': 0.4, 'gs': 0.2,                                  }, 
                'reid':          {                                              're': 0.5, 're-s': 0.4}, 
                }
    
    if not aug_type in aug_dict:
        raise TypeError('Unknown augmentation type: {}'.format(aug_type))
    aug_kwargs = aug_dict[aug_type]
    color_jitter_p = aug_kwargs.get('cj', 0.)
    color_jitter_s = aug_kwargs.get('cj-s', 0.4)
    if 0. < color_jitter_p < 1.:
        augmentation.append(T.RandomApply([T.ColorJitter(
                color_jitter_s, color_jitter_s, color_jitter_s, 0.1)], color_jitter_p))
    elif color_jitter_p == 1:
        augmentation.append(T.ColorJitter(color_jitter_s, color_jitter_s, color_jitter_s, color_jitter_s))

    gray_scale_p = aug_kwargs.get('gs', 0.)
    if 0. < gray_scale_p <= 1.:
        augmentation.append(T.RandomGrayscale(p=gray_scale_p))

    gaussian_blur_p = aug_kwargs.get('gb', 0.)
    if 0. < gaussian_blur_p < 1.:
        augmentation.append(T.RandomApply([GaussianBlur([.1, 2.])], p=gaussian_blur_p))

    augmentation.append(T.RandomHorizontalFlip())
    augmentation.append(T.ToTensor())
    augmentation.append(normalizer)

    random_erase_p = aug_kwargs.get('re', 0.)
    random_erase_s = aug_kwargs.get('re-s', 0.4)
    if 0. < random_erase_p < 1.:
        augmentation.append(RandomErasing(probability=random_erase_p, sh=random_erase_s))

    transformer = T.Compose(augmentation)
    if two_crop:
        transformer = TwoCropsTransform(transformer)
    return transformer

if __name__ == '__main__':
    trans = get_reid_train_transformer()
    print(trans)