import re
import csv
import time
import wandb
import numpy as np
import torch
import os
import argparse
from transform import Apple_train_transform, Apple_test_transform
from dataset import AppleDataset
import torchvision.models as models
from model import My_regnet, My_efficientnet
from torch import nn, optim
import random
from utils import WarmupReduceLROnPlateau
parser = argparse.ArgumentParser(description='PyTorch Apple Testing')
parser.add_argument('--use_cuda', default=True, type=bool, help='use cuda or not')  # 使用GPU
parser.add_argument('--gpu', default='5', type=str, help='gpu id')  # GPU id
parser.add_argument('--data', default="/home/medicaldata/WTJData/xunfei/苹果病害图像识别挑战赛公开数据/", metavar='DIR', help='path to dataset')  # 数据集路径
parser.add_argument('--model', default='regnet', type=str, help='model')  # 模型
parser.add_argument('--batch_size', default=64, type=int, help='batch size')  # 批大小
parser.add_argument('--checkpoint', default="/home/image003/xunfei/apple-regnet-mixup-0.9949.pth", type=str, help='checkpoint')  # 多线程
args = parser.parse_args()
if "regnet" in args.model:
    # 加载模型
    model = My_regnet(num_classes=9)
else:
    model = My_efficientnet(num_classes=9)
if args.use_cuda:
    assert torch.cuda.is_available(), 'No GPU Found!'
    device = 'cuda:' + args.gpu
else:
    device = 'cpu'
model = model.to(device)
# args.checkpoint = "/home/image003/xunfei/apple-regnet-mixup-0.9949.pth"
checkpoint = torch.load(args.checkpoint)

accuracy = re.search(r'\d+\.\d+', args.checkpoint).group()

model.load_state_dict(checkpoint['model_state_dict'])
Data_test = AppleDataset(args.data, "test", transform=Apple_test_transform)
testloader = torch.utils.data.DataLoader(Data_test,
                                         batch_size=args.batch_size,
                                         shuffle=False, num_workers=8)

label_map = {0: "d1", 1: "d2",  2: "d3",  3: "d4",  4: "d5",  5: "d6",  6: "d7",  7: "d8",  8: "d9"}
model.eval()
with torch.no_grad(), open(f'./results/apple-results-{accuracy}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['uuid', 'label'])  # Write the header row

    for i, (inputs, labels, img_name) in enumerate(testloader):
        inputs = inputs.to(device)
        output = model(inputs)
        output = output.argmax(1)

        for j in range(len(output)):
            writer.writerow([img_name[j], label_map[output[j].item()]])  # Write the UUID and label to the CSV file
        print(output)
