# 构建数据集
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from common import one_hot_it


class ChangeDataset(Dataset):
    # 这里的transforms、num_classes和ignore_index需要，避免PaddleSeg在Eval时报错
    def __init__(self, dataset_path, mode = 'train', num_classes=2, ignore_index=255):

        list_path = os.path.join(dataset_path, (mode + '_list.txt'))
            
        self.img_folder = dataset_path#os.path.join(dataset_path, mode)
        self.data_list = self.__get_list(list_path)
        self.data_num = len(self.data_list)
        self.num_classes = num_classes  # 分类数
        self.ignore_index = ignore_index  # 忽视的像素值
        self.label_info = pd.read_csv(os.path.join(dataset_path, 'label_info.csv'))
        self.label_color = np.array([[1,0],[0,1]])
        self.label_color = np.array([[1, 0], [0, 1]])

        assert self.data_list != None, "no data list could load"

        self.sst1_images = []
        self.sst2_images = []
        self.gt_images = []
        for _file in self.data_list:
            self.sst1_images.append(os.path.join(self.img_folder, "A", _file))
            self.sst2_images.append(os.path.join(self.img_folder, "B", _file))
            self.gt_images.append(os.path.join(self.img_folder, "label", _file))

    def __getitem__(self, index):

        A_path = self.sst1_images[index]
        B_path = self.sst2_images[index]
        lab_path = self.gt_images[index]

        A_img = np.array(cv2.imread(A_path), dtype=np.float32)
        B_img = np.array(cv2.imread(B_path), dtype=np.float32)
        image = np.concatenate((A_img, B_img), axis=-1, dtype=np.float32)  # 将两个时段的数据concat在通道层
        w, h, _ = image.shape

        image = np.transpose(image, [2,0,1])
        image = torch.Tensor(image)#.astype('float32')

        label = np.array(cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE))  # cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE)
        label = np.array((label != 0), dtype=np.int8)
        label = self.label_color[label]
        label = np.transpose(label, axes=[2,0,1])
        label = torch.tensor(label, dtype=torch.int64)#.astype('int64')
        data =  image,  label
        return data

    def __len__(self):
        return self.data_num

    # 这个用于把list.txt读取并转为list
    def __get_list(self, list_path):
        # data_list = os.listdir(list_path)
        # return data_list
        with open(list_path, 'r') as f:
            data_list = f.read().split('\n')[:-1]
        return data_list


if __name__ == "__main__":
    dataset_path = '/mnt/data/Datasets/CLCD'
    # 完成三个数据的创建