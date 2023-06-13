"""
    测试通过
    图像增强常用的库有：Albumentations, imgaug, Augmentor, torchvision
    也可以自己使用opencv和skimage自己写函数进行图像增强
    需要注意的是除了对图像进行增强，还需要对标签进行相应的变换
    使用albumentations.pytorch的ToTensorV2而不是torch的ToTensor好像有什么问题，但是记不清了。
"""
from albumentations.pytorch import ToTensorV2
import numpy as np
import albumentations as A  # https://albumentations.ai/docs/getting_started/image_augmentation/
from albumentations import DualTransform, ImageOnlyTransform
from torchvision import transforms as T
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torch
# cutout、Mixup、CutMix
# https://blog.csdn.net/weixin_45928096/article/details/122406271
# https://www.zhihu.com/question/319291048/answer/1223507388
# 疑问？将mixup写在数据增强阶段，还是在模型训练阶段？
# https://www.zhihu.com/question/308572298/answer/585140274写在了训练阶段
# https://zhuanlan.zhihu.com/p/536480131写在了训练阶段
# https://zhuanlan.zhihu.com/p/59805913 写在了训练阶段
# https://blog.csdn.net/weixin_45928096/article/details/122406271写在了数据增强阶段
# 综上，mixup应该是一种训练方法，但也是一种数据增强方法，写在训练阶段更好
# 总结，在官方实现中https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py#L119
# 在batch中，随机选择一张图，batch中的所有图都和它做mixup，而不是每张图都随机选择一张图做mixup，完结撒花
# 但是这样有一个问题，就是同一类的mixup其实无效
Apple_train_transform = A.Compose([
    A.RandomRotate90(),  # 随机旋转90度(图像和标签同时旋转)
    A.Resize(256, 256),  # 缩放到256*256
    A.RandomCrop(224, 224),  # 随机裁剪到224*224
    A.HorizontalFlip(p=0.5),  # 水平翻转
    A.RandomGridShuffle(),  # 随机网格洗牌
    A.GaussianBlur(),  # 高斯模糊
    A.VerticalFlip(p=0.5),  # 垂直翻转
    A.RandomContrast(p=0.5),  # 随机对比度对比度范围Default: (-0.2, 0.2)
    A.RandomBrightnessContrast(p=0.5),   # 随机亮度和对比度，两个范围都是(-0.2, 0.2)
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # ？不知道为什么要缩放到这个范围，可能是使用了预训练权重吧
    A.pytorch.ToTensorV2()  # 转换为tensor
])

Apple_test_transform = A.Compose([
    A.Resize(256, 256),  # 缩放到256*256
    A.RandomCrop(224, 224),  # 随机裁剪到224*224
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # ？不知道为什么要缩放到这个范围，可能是使用了预训练权重吧
    A.pytorch.ToTensorV2()  # 转换为tensor
])

Building_train_transform = A.Compose([
    A.RandomRotate90(),  # 随机旋转90度(图像和标签同时旋转)
    A.Resize(256, 256),  # 缩放到256*256
    A.RandomCrop(224, 224),  # 随机裁剪到224*224
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),  # 水平翻转
    A.RandomContrast(p=0.5),  # 随机对比度对比度范围Default: (-0.2, 0.2)
    A.RandomBrightnessContrast(p=0.5),   # 随机亮度和对比度，两个范围都是(-0.2, 0.2)
    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # ？不知道为什么要缩放到这个范围，可能是使用了预训练权重吧
    A.pytorch.ToTensorV2()  # 转换为tensor
], additional_targets={'image0': 'image', }
)
Building_test_transform = A.Compose([
    A.Resize(256, 256),  # 缩放到256*256
    A.RandomCrop(224, 224),  # 随机裁剪到224*224
    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # ？不知道为什么要缩放到这个范围，可能是使用了预训练权重吧
    A.pytorch.ToTensorV2()  # 转换为tensor
], additional_targets={'image0': 'image', })


def show(img_list, file_name="test.png"):
    if not isinstance(img_list, list):
        img_list = [img_list]
    fig, axs = plt.subplots(ncols=len(img_list), squeeze=False)
    for i, img in enumerate(img_list):
        if type(img) == torch.Tensor:
            img = img.detach()
            img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.savefig(f"{file_name}")


if __name__ == '__main__':
    from dataset import AppleDataset, BuildingDataset
    path = ["/home/medicaldata/WTJData/xunfei/苹果病害图像识别挑战赛公开数据/", "/home/medicaldata/WTJData/xunfei/高分辨率遥感影像建筑物变化检测挑战赛公开数据-初赛"]
    # # 测试苹果数据增强
    # Data_train = AppleDataset(path[0], "train")
    # Data_test = AppleDataset(path[0], "test")
    # # 保存训练集结果
    # Trans_train = Apple_train_transform
    # img, label, index = Data_train[0]
    # temp = Trans_train(image=img, label=label)
    # img_new = temp["image"]
    # show([img, img_new], file_name="train_apple.png")
    # # 保存测试集结果
    # Trans_test = Apple_test_transform
    # img, label, index = Data_test[0]
    # temp = Trans_test(image=img)
    # img_new = temp["image"]
    # show([img, img_new], file_name="test_apple.png")
    # Trans_test = Apple_test_transform
    # 测试建筑物数据增强
    Data_train = BuildingDataset(path[1], "初赛训练集")
    Data_test = BuildingDataset(path[1], "初赛测试集")

    Trans_test = Building_test_transform
    # 保存训练集结果
    Trans_train = Building_train_transform
    img_1, img_2, mask, index = Data_train[0]
    temp = Trans_train(image=img_1, image0=img_2, mask=mask)
    img_new_1, img_new_2, mask_new = temp["image"], temp["image0"], temp["mask"]
    show([img_1, img_2, img_new_1, img_new_2, mask, mask_new], file_name="train_Building.png")
    # 保存测试集结果
    Trans_test = Building_test_transform
    img_1, img_2, mask, index = Data_test[0]
    temp = Trans_test(image=img_1, image0=img_2)
    img_new_1, img_new_2 = temp["image"], temp["image0"]
    show([img_1, img_2, img_new_1, img_new_2], file_name="test_Building.png")
