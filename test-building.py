import zipfile
from PIL import Image
from pkbar import Kbar
import pandas as pd
import re
import csv
import time
import wandb
import numpy as np
import torch
import os
import argparse
from transform import Building_train_transform, Building_test_transform
from dataset import BuildingDataset
from dataset import AppleDataset
import torchvision.models as models
from model import My_regnet, My_efficientnet
from torch import nn, optim
import random
from utils import WarmupReduceLROnPlateau, mkdir
import segmentation_models_pytorch as smp
from glob import glob
parser = argparse.ArgumentParser(description='PyTorch Apple Testing')
parser.add_argument('--use_cuda', default=True, type=bool, help='use cuda or not')  # 使用GPU
parser.add_argument('--gpu', default='5', type=str, help='gpu id')  # GPU id
parser.add_argument('--data', default="/home/medicaldata/WTJData/xunfei/高分辨率遥感影像建筑物变化检测挑战赛公开数据-初赛/初赛测试集/*", metavar='DIR', help='path to dataset')  # 数据集路径
parser.add_argument('--batch_size', default=32, type=int, help='batch size')  # 批大小
parser.add_argument('--checkpoint', default="/home/image003/xunfei/resnet50/resnet50_fold4_0.7839911580085754.pth", type=str, help='checkpoint')  # 多线程
parser.add_argument('--model', default='resnet50', type=str, help='model')  # 模型
args = parser.parse_args()
model = smp.Unet(
    encoder_name=args.model,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,                      # model output channels (number of classes in your dataset)
)
if args.use_cuda:
    assert torch.cuda.is_available(), 'No GPU Found!'
    device = 'cuda:' + args.gpu
else:
    device = torch.device('cpu')

# args.checkpoint = "/home/image003/xunfei/apple-regnet-mixup-0.9949.pth"
checkpoint = torch.load(args.checkpoint)
mkdir("submit")
mkdir("submit2")
model.load_state_dict(checkpoint['model'])
model = model.to(device)
temp = glob("/home/medicaldata/WTJData/xunfei/高分辨率遥感影像建筑物变化检测挑战赛公开数据-初赛/初赛测试集/Image1/*")
temp2 = glob("/home/medicaldata/WTJData/xunfei/高分辨率遥感影像建筑物变化检测挑战赛公开数据-初赛/初赛测试集/Image2/*")
df_test = pd.DataFrame(data={"img1_path": np.array(temp), "img2_path": np.array(temp2)})
Data_test = BuildingDataset(df_test, transform=Building_test_transform, split="test")
testloader = torch.utils.data.DataLoader(Data_test,
                                         batch_size=args.batch_size,
                                         shuffle=False, num_workers=8)
model.eval()
with torch.no_grad():
    kbar = Kbar(len(testloader))

    for i, batch_data in enumerate(testloader):
        img_1, img_2, _, file_name = batch_data
        img_1, img_2 = img_1.to(device), img_2.to(device)
        output = model(abs(img_1 - img_2))
        output = output.cpu().numpy()  # 将输出移回到CPU并转换为Numpy数组
        for index in range(len(file_name)):
            temp = np.array(np.argmax(output[index], 0), dtype=np.uint8)
            temp_128 = temp*128
            temp = Image.fromarray(temp)
            temp_128 = Image.fromarray(temp_128)
            temp.save(f"./submit/{file_name[index].replace('tif','png')}")
            temp_128.save(f"./submit2/{file_name[index].replace('tif','png')}")
        kbar.add(1)


def zip_directory(directory, zip_name):
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, directory)
                zip_path = os.path.join('submit', relative_path)
                zipf.write(file_path, zip_path)


directory = './submit'
zip_name = 'submit.zip'

zip_directory(directory, zip_name)

print('文件夹已成功压缩为submit.zip')
