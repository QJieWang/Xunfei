"""
AppleDataset测试通过
BuildingDataset测试通过


Returns:
    _type_: _description_
"""
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from PIL import Image
import os
import cv2


class AppleDataset(Dataset):
    """
    苹果病害数据集,训练集和测试集的文件结构不同，所以需要根据split关键词进行判断

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, path, split, transform=None) -> None:
        super().__init__()
        self.split = split
        self.path = path
        self.transform = transform
        self.img_list, self.label = self.get_img_list(self.path)
        self.label_map = {"d1": 0, "d2": 1, "d3": 2, "d4": 3, "d5": 4, "d6": 5, "d7": 6, "d8": 7, "d9": 8}
        # d1 黑斑病, d2 褐斑病, d3 青枯叶斑病, d4 灰斑病,d5 健康,d6 花叶病毒病,d7 白粉病,d8 锈病,d9 疮痂病

    def get_img_list(self, path):
        """
        返回图像的路径和标签

        Args:
            path (_type_): _description_

        Returns:
            _type_: _description_
        """
        img_list = []
        img_label = []
        if self.split == "train":
            for disease in os.listdir(os.path.join(path, self.split)):
                if disease.endswith("txt"):
                    continue
                img_list += os.listdir(os.path.join(path, self.split, disease))
                img_label += [disease]*len(os.listdir(os.path.join(path, self.split, disease)))
        elif self.split == "test":
            img_list = os.listdir(os.path.join(path, self.split))
        return img_list, img_label

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        label = []
        # lable = None
        if self.split == "train":
            label = self.label[index]
            path = os.path.join(self.path, self.split, label, self.img_list[index])
        else:
            path = os.path.join(self.path, self.split, self.img_list[index])
        img = Image.open(path)
        img = np.array(img)
        if self.transform:
            img = self.transform(image=img)["image"]
        if self.label:
            label = self.label_map[self.label[index]]
        return img, label, self.img_list[index]


class BuildingDataset(Dataset):
    """
    这里的Transform写的不是特别好，但是确实第一次遇到三张图片的情况，所以将Transform分别写

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, paht, split, transform=[None]*3) -> None:
        super().__init__()
        self.path = paht
        self.transform_1, self.transform_2, self.transform_3 = transform
        self.split = split
        self.img_list, self.label = self.get_img_list(os.path.join(self.path, split))
        # 初赛测试集  初赛训练集

    def get_img_list(self, path):
        img_list = []
        img_label = []

        temp_1 = os.listdir(os.path.join(path, "Image1"))
        temp_2 = os.listdir(os.path.join(path, "Image2"))
        if temp_1 != temp_2:
            assert "Image1和Image2的文件名不一致"
        else:
            img_list = temp_1
        if self.split == "初赛训练集":
            img_label += os.listdir(os.path.join(path,  "label1"))
        return img_list, img_label

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        label = None
        img_1 = Image.open(os.path.join(self.path, self.split, "Image1", self.img_list[index]))
        img_2 = Image.open(os.path.join(self.path, self.split, "Image2", self.img_list[index]))
        if self.split == "初赛训练集":
            label = Image.open(os.path.join(self.path, self.split, "label1", self.label[index]))
        if self.transform_1:
            img_1 = self.transform_1(img_1)
        if self.transform_2:
            img_2 = self.transform_2(img_2)
        if self.transform_3:
            label = self.transform_3(label)
        return np.array(img_1), np.array(img_2), np.array(label), self.img_list[index]


if __name__ == '__main__':
    path = ["/home/medicaldata/WTJData/xunfei/苹果病害图像识别挑战赛公开数据/", "/home/medicaldata/WTJData/xunfei/高分辨率遥感影像建筑物变化检测挑战赛公开数据-初赛"]
    Data_train = AppleDataset(path[0], "train")
    Data_test = AppleDataset(path[0], "test")
    print(Data_train[0])
    print(Data_test[0])
    Data_train = BuildingDataset(path[1], "初赛训练集")
    Data_test = BuildingDataset(path[1], "初赛测试集")
    print(Data_train[0])
    print(Data_test[0])
