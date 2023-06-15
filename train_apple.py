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
# 解析命令行参数
parser = argparse.ArgumentParser(description='PyTorch Apple Training')
parser.add_argument('--use_cuda', default=True, type=bool, help='use cuda or not')  # 使用GPU
parser.add_argument('--gpu', default='4', type=str, help='gpu id')  # GPU id
parser.add_argument('--data', default="/home/medicaldata/WTJData/xunfei/苹果病害图像识别挑战赛公开数据/", metavar='DIR', help='path to dataset')  # 数据集路径
parser.add_argument('--model', default='regnet', type=str, help='model')  # 模型
parser.add_argument("--num_workers", default=8, type=int, help='number of workers')  # 多线程加载数据
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')  # 学习率
# args.warmup_epochs, args.warmup_factor,
parser.add_argument("--warmup_epochs", default=5, type=int, help='warmup epochs')  # warmup epochs")
parser.add_argument("--warmup_factor", default=2, type=float, help='warmup factor')  # warmup factor
parser.add_argument('--resume', '-r', default=False, action='store_true', help='resume from checkpoint')  # 是否从断点处恢复训练
parser.add_argument('--name', default='apple', type=str, help='name of experiment')  # 实验名称
parser.add_argument('--seed', default=9617, type=int, help='random seed')  # 随机种子
parser.add_argument('--batch_size', default=25, type=int, help='batch size')  # 批大小
parser.add_argument('--epoch', default=400, type=int, help='total epochs to run')  # 总共训练多少个epoch
parser.add_argument('--mixup', default=False, type=float, help='mixup alpha')  # mixup参数
parser.add_argument('--alpha', default=0.2, type=float,
                    help='mixup/cutmix alpha (default: 1)')
parser.add_argument('--cutmix', default=True, type=float, help='cutmix alpha')  # cutmix参数
args = parser.parse_args()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if "regnet" in args.model:
    # 加载模型
    model = My_regnet(num_classes=9)
else:
    model = My_efficientnet(num_classes=9)
optimizer = torch.optim.AdamW(model.parameters(),  lr=args.lr)
now = time.strftime('%Y-%m-%d-%HH', time.localtime())
wandb.init(
    project="Apple",  # 项目
    name=f"experiment_{args.model}_{now}",  # 模型
    config=args,  # 配置
)

if args.seed != 0:
    torch.manual_seed(args.seed)
if args.use_cuda:
    assert torch.cuda.is_available(), 'No GPU Found!'
    device = 'cuda:' + args.gpu
else:
    device = 'cpu'
transform_model = ""
if args.mixup > 0 and args.cutmix > 0:
    raise ValueError('Cannot use both mixup and cutmix at the same time.')
if args.mixup > 0:
    print('Using mixup with alpha = {}'.format(args.alpha))
    transform_model = "mixup"
if args.cutmix > 0:
    print('Using cutmix with alpha = {}'.format(args.alpha))
    transform_model = "cutmix"
wandb.init(
    project="Apple",  # 项目
    name=f"experiment_{args.model}_{now}_{transform_model}",  # 模型
    config=args,  # 配置
)

print(f"args.resume={args.resume}")
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('/home/image003/xunfei/checkpoint/apple-regnet-mixup-0.9903045734991676_9617.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    best_acc = checkpoint['acc']
    best_acc = 0
    start_epoch = checkpoint['epoch'] + 1
    start_epoch = 0  # TODO: 等删除
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)
    print('==> Loaded checkpoint at epoch: %d, with loss: %.3f, acc: %.3f')
    model.to(device)
if not os.path.isdir('results'):
    os.mkdir('results')
model = model.to(device)
criterion = nn.CrossEntropyLoss()

plateau_params = {'mode': 'max', 'patience': 4, 'factor': 0.1,  'min_lr': 0.0001}
scheduler = WarmupReduceLROnPlateau(optimizer, args.warmup_epochs, args.warmup_factor, plateau_params)


def train(trainloader, model, criterion, optimizer, scheduler):
    model.train()
    train_loss = 0
    top1_correct = 0
    top5_correct = 0
    for batch_idx, (input, target, img_name) in enumerate(trainloader):
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        # output = model(input)
        if args.mixup > 0.:
            input, targets_a, targets_b, lam = mixup_data(input, target,
                                                          args.alpha)
            output = model(input)
            loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
        elif args.cutmix > 0.:
            input, targets_a, targets_b, lam = cutmix_data(input, target,
                                                           args.alpha)
            output = model(input)
            loss = cutmix_criterion(criterion, output, targets_a, targets_b, lam)
        else:
            output = model(input)
            loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = output.max(1)
        top1_correct += (predicted == target).sum().item()

        _, top5_predicted = output.topk(5, 1)
        top5_correct += sum([1 for i in range(len(target)) if target[i] in top5_predicted[i]])

        print('Train loss', loss.item())

        train_loss += loss.item()
    print('Train loss', train_loss / len(trainloader))
    print('Top1 accuracy', top1_correct / len(trainloader))
    print('Top5 accuracy', top5_correct / len(trainloader))
    # 使用wandb记录实验过程中的信息
    wandb.log({"train_loss": train_loss / len(trainloader)})
    wandb.log({"train_top1": top1_correct})
    wandb.log({"train_top5": top5_correct})
    wandb.log({"lr": scheduler.optimizer.param_groups[0]["lr"]})


def validate(val_loader, model, criterion):
    model.eval()
    val_acc = 0.0
    val_loss = 0
    top1_correct, top5_correct = 0, 0
    with torch.no_grad():
        end = time.time()

        for i, (input, target, img_name) in enumerate(val_loader):
            input = input.to(device)
            target = target.to(device)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            val_acc += (output.argmax(1) == target).sum().item()
            _, predicted = output.max(1)
            top1_correct += (predicted == target).sum().item()

            _, top5_predicted = output.topk(5, 1)
            top5_correct += sum([1 for i in range(len(target)) if target[i] in top5_predicted[i]])
            val_loss += loss.item()
    wandb.log({"val_loss": val_loss / len(val_loader)})
    wandb.log({"val_top1": top1_correct})
    wandb.log({"val_top5": top5_correct})

    return val_acc / len(val_loader.dataset)


def mixup_data(inputs, targets, alpha):
    batch_size = inputs.size(0)
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(batch_size)
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
        targets_a, targets_b = targets, targets[index]
        return mixed_inputs, targets_a, targets_b, lam
    else:
        return inputs, targets, targets, 1


def cutmix_data(inputs, targets, alpha):
    batch_size = inputs.size(0)
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)
        index = torch.randperm(batch_size)
        mixed_inputs = inputs.clone()
        mixed_inputs[:, :, :, :] = inputs[index, :, :, :]
        mixed_inputs[:, :, :, :] = lam * inputs + (1 - lam) * mixed_inputs
        targets_a, targets_b = targets, targets[index]
        return mixed_inputs, targets_a, targets_b, lam
    else:
        return inputs, targets, targets, 1


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, f'./checkpoint/apple-{args.model}-{transform_model}-{acc}' + '_'
               + str(args.seed)+'.pth')


if __name__ == '__main__':
    # TODO:这里明天重写
    # self.label_map = {"d1": 0, "d2": 1, "d3": 2, "d4": 3, "d5": 4, "d6": 5, "d7": 6, "d8": 7, "d9": 8}
    # d1 黑斑病, d2 褐斑病, d3 青枯叶斑病, d4 灰斑病,d5 健康,d6 花叶病毒病,d7 白粉病,d8 锈病,d9 疮痂病
    from torch.utils.data import DataLoader, WeightedRandomSampler
    class_num = {'d1': 292, 'd2': 288, 'd3': 2227, 'd4': 238, 'd5': 362, 'd6': 260, 'd7': 829, 'd8': 1928, 'd9': 3787}
    class_counts = [292, 288, 2227, 238, 362, 260, 829, 1928, 3787]
    class_weights = [sum(class_counts) / c for c in class_counts]
    class_weights = torch.FloatTensor(class_weights).to(device)
    sampler = WeightedRandomSampler(class_weights, sum(class_counts), replacement=True)
    Data_train = AppleDataset(args.data, "train", transform=Apple_train_transform)
    Data_test = AppleDataset(args.data, "test", transform=Apple_test_transform)
    # trainloader = torch.utils.data.DataLoader(Data_train,
    #                                           batch_size=args.batch_size,
    #                                           shuffle=True, num_workers=8, sampler=sampler)
    trainloader = torch.utils.data.DataLoader(Data_train,
                                              batch_size=args.batch_size,
                                              num_workers=8, shuffle=True)
    testloader = torch.utils.data.DataLoader(Data_test,
                                             batch_size=args.batch_size,
                                             shuffle=False, num_workers=8)
    for epoch in range(start_epoch, start_epoch + args.epoch):
        train(trainloader, model, criterion, optimizer, scheduler)

        acc = validate(trainloader, model, criterion)
        scheduler.step(epoch, acc)
        print('Epoch:', epoch, 'Val Acc:', acc)
        if acc > best_acc:
            best_acc = acc
            checkpoint(acc, epoch)
            if best_acc > 0.999:
                best_acc = 0.99-0.02
