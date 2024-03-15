import torch
import torchvision.transforms as tfs
import os
import cv2
from PIL import Image
import random
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
        with open(os.path.join(path_root, ("{}_list.txt".format(mode))), 'r') as f:
            self.sst1_images_dir = f.read().split('\n')[:-1]

        assert self.sst1_images_dir != None, "no data list could load"
        self.sst1_images=[os.path.join(self.path_root,"A",img) for img in self.sst1_images_dir]
        self.gt_images=[os.path.join(self.path_root,"label",img) for img in self.sst1_images_dir]

    def __getitem__(self, item):

        sst1 = Image.open(self.sst1_images[item])
        gt = Image.open(self.gt_images[item])

        sst1 = tfs.ToTensor()(sst1)
        gt = np.array(gt)
        if (len(gt.shape) == 3):
            gt = one_hot_it(gt, self.label_info)
        #gt = np.argmax(gt,axis=2)
        else:
            gt = np.array((gt != 0),dtype=np.int8)
            gt = self.label_color[gt]
        gt = np.transpose(gt, [2, 0, 1])
        gt = torch.from_numpy(gt).type(torch.float32)
        return sst1, gt

    def __len__(self):
        return len(self.sst1_images)

    def handle_image(self,img):
        img=np.array(img)
        img=Image.fromarray(img).convert("RGB")
        return self.to_tensor(img).type(torch.float32)

    def handle_label(self,label):
        label=np.array(label)
        label=one_hot_it(label,self.label_info).astype(np.uint8)
        label=np.transpose(label,[2,0,1])
        label=torch.from_numpy(label).type(torch.float32)
        return label


    def randomRotation(image, label, mode=Image.BICUBIC):

        """
         对图像进行随机任意角度(0~360度)旋转
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL的图像image
        :return: 旋转转之后的图像
        """
        random_angle = np.random.randint(1, 360)
        return image.rotate(random_angle, mode), label.rotate(random_angle, Image.NEAREST)

    def randomCrop(image, label):
        """
        对图像随意剪切,考虑到图像大小范围(68,68),使用一个一个大于(36*36)的窗口进行截图
        :param image: PIL的图像image
        :return: 剪切之后的图像
        """
        image_width = image.size[0]
        image_height = image.size[1]
        crop_win_size = np.random.randint(40, 68)
        random_region = (
            (image_width - crop_win_size) >> 1, (image_height - crop_win_size) >> 1, (image_width + crop_win_size) >> 1,
            (image_height + crop_win_size) >> 1)
        return image.crop(random_region), label

    @staticmethod
    def randomGaussian(image, label, mean=0.2, sigma=0.3):
        """
         对图像进行高斯噪声处理
        :param image:
        :return:
        """

        def gaussianNoisy(im, mean=0.2, sigma=0.3):
            """
            对图像做高斯噪音处理
            :param im: 单通道图像
            :param mean: 偏移量
            :param sigma: 标准差
            :return:
            """
            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im
        # 将图像转化成数组
        img = np.asarray(image)
        img.flags.writeable = True  # 将数组改为读写模式
        width, height = img.shape[:2]
        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
        img[:, :, 0] = img_r.reshape([width, height])
        img[:, :, 1] = img_g.reshape([width, height])
        img[:, :, 2] = img_b.reshape([width, height])
        return Image.fromarray(np.uint8(img)), label



class DataReaderwithEdge(data.Dataset):
    def __init__(self, dataset_path, mode):
        super(DataReaderwithEdge, self).__init__()
        self.data_list = self._get_list(os.path.join(dataset_path, (mode + '_list.txt')))
        self.data_num = len(self.data_list)

        self.label_info = pd.read_csv(os.path.join(dataset_path, 'label_info.csv'))

        self.sst1_images = []
        self.sst1_edge = []
        self.gt_images = []
        for _file in self.data_list:
            self.sst1_images.append(os.path.join(dataset_path, "A", _file))
            self.sst1_edge.append(os.path.join(dataset_path, "AEdge", _file))
            self.gt_images.append(os.path.join(dataset_path, "label", _file))

    def __getitem__(self, index):
        # print(self.data_list[index])
        A_path = self.sst1_images[index]
        AEdge_path = self.sst1_edge[index]
        lab_path = self.gt_images[index]

        A_img = self._normalize(np.array(Image.open(A_path)))

        edge1 = cv2.imread(AEdge_path, cv2.IMREAD_UNCHANGED)
        sst1 = np.concatenate((A_img, edge1[..., np.newaxis]), axis=-1)

        sst1 = torch.from_numpy(sst1.transpose(2, 0, 1)).type(torch.float32)

        label = np.array(Image.open(lab_path))  # cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE)
        if (len(label.shape) == 3):
            label = one_hot_it(label, self.label_info)

        gt = np.argmax(label, axis=-1)
        # label = tfs.ToTensor()(label[np.newaxis, :]).astype('int64')
        # gt = np.transpose(label, [2, 0, 1])
        gt = torch.from_numpy(gt).type(torch.int64)
        return sst1, gt


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
    
    def handle_image(self,img):
        img=np.array(img)
        img=Image.fromarray(img).convert("RGB")
        return self.to_tensor(img).type(torch.float32)

    def handle_label(self,label):
        label=np.array(label)
        label=one_hot_it(label,self.label_info).astype(np.uint8)
        label=np.transpose(label,[2,0,1])
        label=torch.from_numpy(label).type(torch.float32)
        return label


    def randomRotation(image, label, mode=Image.BICUBIC):

        """
         对图像进行随机任意角度(0~360度)旋转
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL的图像image
        :return: 旋转转之后的图像
        """
        random_angle = np.random.randint(1, 360)
        return image.rotate(random_angle, mode), label.rotate(random_angle, Image.NEAREST)

    def randomCrop(image, label):
        """
        对图像随意剪切,考虑到图像大小范围(68,68),使用一个一个大于(36*36)的窗口进行截图
        :param image: PIL的图像image
        :return: 剪切之后的图像
        """
        image_width = image.size[0]
        image_height = image.size[1]
        crop_win_size = np.random.randint(40, 68)
        random_region = (
            (image_width - crop_win_size) >> 1, (image_height - crop_win_size) >> 1, (image_width + crop_win_size) >> 1,
            (image_height + crop_win_size) >> 1)
        return image.crop(random_region), label

    @staticmethod
    def randomGaussian(image, label, mean=0.2, sigma=0.3):
        """
         对图像进行高斯噪声处理
        :param image:
        :return:
        """

        def gaussianNoisy(im, mean=0.2, sigma=0.3):
            """
            对图像做高斯噪音处理
            :param im: 单通道图像
            :param mean: 偏移量
            :param sigma: 标准差
            :return:
            """
            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im
        # 将图像转化成数组
        img = np.asarray(image)
        img.flags.writeable = True  # 将数组改为读写模式
        width, height = img.shape[:2]
        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
        img[:, :, 0] = img_r.reshape([width, height])
        img[:, :, 1] = img_g.reshape([width, height])
        img[:, :, 2] = img_b.reshape([width, height])
        return Image.fromarray(np.uint8(img)), label
