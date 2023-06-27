import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import wandb
import time
from colorama import Fore, Style
import torch.nn as nn
import segmentation_models_pytorch as smp
from dataset import BuildingDataset
from glob import glob
import numpy as np
import pandas as pd
from transform import Building_train_transform, Building_test_transform
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from torch.utils.data import DataLoader
import torch
import argparse
from pkbar import Kbar
from utils import get_predictions, mIoU, mkdir
from pkbar import Kbar  # 进度条
parser = argparse.ArgumentParser(description='PyTorch Building')
# parser.add_argument('--use_cuda', default=True, type=bool, help='use cuda or not')  # 使用GPU
parser.add_argument('--gpu', default='5', type=str, help='gpu id')  # GPU id
parser.add_argument('--model', default='resnet50', type=str, help='model')  # 模型
parser.add_argument('--batch_size', default=20, type=int, help='batch size')  # 批大小
parser.add_argument('--epochs', default=50, type=int, help='epochs')  # 批大小
parser.add_argument('--folds', default=5, type=int, help='num of folds')  # 批大小
parser.add_argument('--seed', default=961717, type=int, help='num of seed')  # 批大小
parser.add_argument('--scheduler', default='ReduceLROnPlateau', type=str, help='scheduler')  # 批大小
# parser.add_argument('--checkpoint', default="/home/image003/xunfei/apple-regnet-mixup-0.9949.pth", type=str, help='checkpoint')  # 多线程
args = parser.parse_args()
# device = torch.device(f"cuda:{args.gpu}")
device = torch.device(f"cuda:{args.gpu}")
# 创建总数据集
temp = glob("/home/medicaldata/WTJData/xunfei/高分辨率遥感影像建筑物变化检测挑战赛公开数据-初赛/初赛训练集/Image1/*")
temp2 = glob("/home/medicaldata/WTJData/xunfei/高分辨率遥感影像建筑物变化检测挑战赛公开数据-初赛/初赛训练集/Image2/*")
label = glob("/home/medicaldata/WTJData/xunfei/高分辨率遥感影像建筑物变化检测挑战赛公开数据-初赛/初赛训练集/label1/*")
df_train = pd.DataFrame(data={"img1_path": np.array(temp), "img2_path": np.array(temp2), "label_path": label})
kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
df_train['fold'] = -1


temp = glob("/home/medicaldata/WTJData/xunfei/高分辨率遥感影像建筑物变化检测挑战赛公开数据-初赛/初赛测试集/Image1/*")
temp2 = glob("/home/medicaldata/WTJData/xunfei/高分辨率遥感影像建筑物变化检测挑战赛公开数据-初赛/初赛测试集/Image2/*")
df_test = pd.DataFrame(data={"img1_path": np.array(temp), "img2_path": np.array(temp2)})
# wandb
now = time.strftime('%Y-%m-%d-%HH', time.localtime())
# wandb.init(
#     project="Building",  # 项目
#     name=f"experiment_{args.model}_{now}",  # 模型

# )


def train_loop(train_df, val_df, fold):
    train_data = BuildingDataset(train_df, transform=Building_train_transform, split="train")
    val_data = BuildingDataset(val_df, transform=Building_train_transform, split="train")
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=8)
    model.to(device)
    best_score = 0.

    for epoch in range(args.epochs):
        # 开始训练
        loss, iou = train(train_dataloader)
        val_loss, val_iou = validate(val_dataloader, epoch)

        if args.scheduler == "ReduceLROnPlateau":
            scheduler.step(val_iou)
            # wandb.log({"lr": scheduler.optimizer.param_groups[0]["lr"]})
        elif args.scheduler == "CosineAnnealingWarmRestarts":
            pass
        score = val_iou
        if score > best_score:
            best_score = score
            torch.save({'model': model.state_dict()},
                       f'./{args.model}/{args.model}_fold{fold}_{score}.pth')
    torch.save({'model': model.state_dict()},
               f'./{args.model}/{args.model}_fold{fold}_last.pth')

    return None


def train(train_loader):
    # TODO:重写
    metrics_names = ['loss', 'train_miou']  # 需要统计的指标
    kbar = Kbar(len(train_loader), num_epochs=args.epochs, width=20, stateful_metrics=metrics_names)
    model.train()
    train_loss = 0.0
    train_iou, temp_iou = 0, 0
    for i, batchdata in enumerate(train_loader):
        img_1, img_2, target, file_name = batchdata
        img_1, img_2, target = img_1.to(device), img_2.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(torch.abs(img_1-img_2))
        loss = criterion(output, target.long())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        temp_iou = getmiou(output, target)
        train_iou += temp_iou
        kbar.update(i, values=[("loss", loss.item()), ("train_miou", temp_iou)])
    kbar.add(1)
    return train_loss/len(train_loader), train_iou/len(train_loader)


def getmiou(pre, target):
    bs, c, h, w = pre.shape
    values, indices = torch.max(pre, 1)
    indices = indices.view(bs, h, w)
    iou = (indices*target).sum()/target.sum()
    return iou


def validate(val_loader, epoch):
    # TODO:重写
    model.eval()

    val_loss = 0.0
    val_iou = 0
    with torch.no_grad():
        for i, batchdata in enumerate(val_loader):
            img_1, img_2, target, file_name = batchdata
            img_1, img_2, target = img_1.to(device), img_2.to(device), target.to(device)

            # compute output
            output = model(torch.abs(img_1-img_2))

            loss = criterion(output, target.long())
            val_loss += loss.item()
            # 二分类通过乘法实现miou
            val_iou += getmiou(output, target)
    output = output.data.cpu().numpy()
    target = target.data.cpu().numpy()

    plt.figure()
    plt.subplot(121)

    plt.imshow(target[0] * 128)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(122)
    predict = np.argmax(output[0], 0)
    plt.imshow(predict * 128)
    plt.xticks([])
    plt.yticks([])

    plt.savefig(f"output/{epoch}.png")
    return val_loss / len(val_loader), val_iou / len(val_loader)


def test():
    pass


model = smp.Unet(
    encoder_name=args.model,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,                      # model output channels (number of classes in your dataset)
)
mkdir(args.model)
model.to(device)
# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.Adam(model.parameters(), 0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-7)
for fold, (train_idx, val_idx) in enumerate(kf.split(df_train["img1_path"])):
    df_train.loc[train_idx, 'fold'] = fold

    print(f"========== fold: {fold + 1} training ==========")
    df_fold_train = df_train[df_train["fold"] == fold].reset_index(drop=True)
    df_fold_val = df_train[df_train["fold"] != fold].reset_index(drop=True)
    train_loop(df_fold_train, df_fold_val, fold)
