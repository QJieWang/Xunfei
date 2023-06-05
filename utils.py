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
