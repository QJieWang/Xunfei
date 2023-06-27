import argparse
import csv
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import sys
import time
import math


import torch.optim.lr_scheduler as lr_scheduler


class WarmupReduceLROnPlateau:
    def __init__(self, optimizer, warmup_epochs, warmup_factor, plateau_params):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **plateau_params)
        self.current_lr = None
        self.current_epoch = 0

    def step(self, epoch, loss):
        self.current_epoch = epoch

        if self.current_epoch < self.warmup_epochs:
            self.warmup_lr()
        else:
            self.plateau_scheduler.step(loss)

    def warmup_lr(self):
        if self.current_lr is None:
            self.current_lr = self.optimizer.param_groups[0]['lr']

        warmup_lr = self.current_lr * (self.current_epoch / self.warmup_epochs) * self.warmup_factor
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr


def mIoU(predictions, targets, info=False):  # Mean per class accuracy
    unique_labels = np.unique(targets)
    num_unique_labels = len(unique_labels)
    ious = []
    for index in range(num_unique_labels):
        pred_i = predictions == index
        label_i = targets == index
        intersection = np.logical_and(label_i, pred_i)
        union = np.logical_or(label_i, pred_i)
        iou_score = np.sum(intersection.numpy())/np.sum(union.numpy())
        ious.append(iou_score)
    if info:
        print("per-class mIOU: ", ious)
    return np.mean(ious)


def get_predictions(output):
    bs, c, h, w = output.size()
    values, indices = output.cpu().max(1)
    indices = indices.view(bs, h, w)  # bs x h x w
    return indices


def mkdir(path: str):
    """Create directory.
     Create directory if it is not exist, else do nothing.
     Parameters
     ----------
     path: str
        Path of your directory.
     Examples
     --------
     mkdir("data/raw/train/")
     """
    try:
        if path is None:
            pass
        else:
            os.stat(path)
    except Exception:
        os.makedirs(path)
