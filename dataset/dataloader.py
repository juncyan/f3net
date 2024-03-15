import torch
import torchvision.transforms as tfs
import os
import cv2
from PIL import Image
import numpy as np
from torch.utils import data
import pandas as pd

#label_info = {"0":np.array([0,0,0]), "1":np.array([255,255,255])}
def one_hot_it(label, label_info):
    semantic_map = []
    for info in label_info:
        color = label_info[info].values
        # print("label:\n", label.shape,label)
        # print("color:\n", color)
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    return np.stack(semantic_map, axis=-1)


class DataReader(data.Dataset):
    def __init__(self,path_root="./dataset/",mode="train"):
        super(DataReader,self).__init__()
        self.path_root = path_root
        self.sst1_images_dir = None
        #self.label_info = {"0": np.array([0, 0, 0]), "1": np.array([255, 255, 255])}
        self.label_info = pd.read_csv(os.path.join(path_root, 'label_info.csv'))
        self.label_color = np.array([[1,0],[0,1]])
        with open(os.path.join(path_root, ("{}_list.txt".format(mode))), 'r') as f:
            self.sst1_images_dir = f.read().split('\n')[:-1]

        assert self.sst1_images_dir != None, "no data list could load"

        self.sst1_images=[os.path.join(self.path_root,"A",img) for img in self.sst1_images_dir]
        self.sst2_images = [os.path.join(self.path_root, "B", img) for img in self.sst1_images_dir]
        self.gt_images=[os.path.join(self.path_root,"label",img) for img in self.sst1_images_dir]

    def __getitem__(self, item):

        sst1 = Image.open(self.sst1_images[item])
        sst2 = Image.open(self.sst2_images[item])
        gt = Image.open(self.gt_images[item])

        sst1 = tfs.ToTensor()(sst1)
        sst2=tfs.ToTensor()(sst2)
        gt = np.array(gt)
        if (len(gt.shape) == 3):
            gt = one_hot_it(gt, self.label_info)
        #gt = np.argmax(gt,axis=2)
        else:
            gt = np.array((gt != 0),dtype=np.int8)
            gt = self.label_color[gt]
        gt = np.transpose(gt, [2, 0, 1])
        gt = torch.from_numpy(gt).type(torch.float32)
        return sst1, sst2, gt

    def __len__(self):
        return len(self.sst1_images)



class DataReaderwithEdge(data.Dataset):
    def __init__(self, dataset_path, mode):
        super(DataReaderwithEdge, self).__init__()
        self.data_list = self._get_list(os.path.join(dataset_path, (mode + '_list.txt')))
        self.data_num = len(self.data_list)

        self.label_info = pd.read_csv(os.path.join(dataset_path, 'label_info.csv'))
        self.label_color = np.array([[0,1],[1,0]])
        self.sst1_images = []
        self.sst1_edge = []
        self.sst2_images = []
        self.sst2_edge = []
        self.gt_images = []
        for _file in self.data_list:
            self.sst1_images.append(os.path.join(dataset_path, "A", _file))
            self.sst2_images.append(os.path.join(dataset_path, "B", _file))
            self.sst1_edge.append(os.path.join(dataset_path, "AEdge", _file))
            self.sst2_edge.append(os.path.join(dataset_path, "BEdge", _file))
            self.gt_images.append(os.path.join(dataset_path, "label", _file))

    def __getitem__(self, index):
        # print(self.data_list[index])
        A_path = self.sst1_images[index]
        B_path = self.sst2_images[index]
        AEdge_path = self.sst1_edge[index]
        BEdge_path = self.sst2_edge[index]
        lab_path = self.gt_images[index]

        A_img = self._normalize(np.array(Image.open(A_path)))
        B_img = self._normalize(np.array(Image.open(B_path)))

        edge1 = cv2.imread(AEdge_path, cv2.IMREAD_UNCHANGED)
        edge2 = cv2.imread(BEdge_path, cv2.IMREAD_UNCHANGED)
        sst1 = np.concatenate((A_img, edge1[..., np.newaxis]), axis=-1)
        sst2 = np.concatenate((B_img, edge2[..., np.newaxis]), axis=-1)

        sst1 = torch.from_numpy(sst1.transpose(2, 0, 1)).type(torch.float32)
        sst2 = torch.from_numpy(sst2.transpose(2, 0, 1)).type(torch.float32)

        label = np.array(Image.open(lab_path))  # cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE)
        if (len(label.shape) == 3):
            label = one_hot_it(label, self.label_info)
        else:
            label = np.array((label != 0), dtype=np.int8)
            label = self.label_color[label]
        gt = np.argmax(label, axis=-1)
        # label = tfs.ToTensor()(label[np.newaxis, :]).astype('int64')
        # gt = np.transpose(label, [2, 0, 1])
        gt = torch.from_numpy(gt).type(torch.int64)
        return sst1, sst2, gt


    def __len__(self):
        return self.data_num

    # 这个用于把list.txt读取并转为list
    def _get_list(self, list_path):
        with open(list_path, 'r') as f:
            data_list = f.read().split('\n')[:-1]
        return data_list

    @staticmethod
    def _normalize(img, mean=[0.485, 0.456, 0.406], std=[1, 1, 1]):
        im = img.astype(np.float32, copy=False)
        mean = np.array(mean)[np.newaxis, np.newaxis, :]
        std = np.array(std)[np.newaxis, np.newaxis, :]
        im = im / 255.0
        im -= mean
        im /= std
        return im
