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


class CDReader(data.Dataset):
    def __init__(self,path_root="./dataset/",mode="train", en_edge=False):
        super(CDReader,self).__init__()
        self.en_edge = en_edge

        self.data_list = self._get_list(os.path.join(path_root, mode))
        self.data_num = len(self.data_list)

        self.label_info = pd.read_csv(os.path.join(path_root, 'label_info.csv'))
        self.label_color = np.array([[1,0],[0,1]])

        self.path_root = os.path.join(path_root, mode)

        self.sst1_images = []
        self.sst1_edge = []
        self.sst2_images = []
        self.sst2_edge = []
        self.gt_images = []
        
        if self.en_edge:
            for _file in self.data_list:
                self.sst1_images.append(os.path.join(self.path_root, "A", _file))
                self.sst2_images.append(os.path.join(self.path_root, "B", _file))
                self.sst1_edge.append(os.path.join(self.path_root, "AEdge", _file))
                self.sst2_edge.append(os.path.join(self.path_root, "BEdge", _file))
                self.gt_images.append(os.path.join(self.path_root, "label", _file))
        else:
            for _file in self.data_list:
                self.sst1_images.append(os.path.join(self.path_root, "A", _file))
                self.sst2_images.append(os.path.join(self.path_root, "B", _file))
                self.gt_images.append(os.path.join(self.path_root, "label", _file))

    def __getitem__(self, index):

        A_path = self.sst1_images[index]
        B_path = self.sst2_images[index]
        lab_path = self.gt_images[index]

        A_img = self._normalize(np.array(Image.open(A_path)))
        B_img = self._normalize(np.array(Image.open(B_path)))

        if self.en_edge:
            AEdge_path = self.sst1_edge[index]
            BEdge_path = self.sst2_edge[index]
            edge1 = cv2.imread(AEdge_path, cv2.IMREAD_UNCHANGED)
            edge2 = cv2.imread(BEdge_path, cv2.IMREAD_UNCHANGED)
            A_img = np.concatenate((A_img, edge1[..., np.newaxis]), axis=-1)  # 将两个时段的数据concat在通道层
            B_img = np.concatenate((B_img, edge2[..., np.newaxis]), axis=-1)  # 将两个时段的数据concat在通道层
        # w, h, _ = A_img.shape

        sst1 = tfs.ToTensor()(A_img)
        sst2 = tfs.ToTensor()(B_img)

        gt = np.array(Image.open(lab_path))
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
        return self.data_num

    def _get_list(self, list_path):
        
        # data_list = None
        
        # with open(list_path, 'r') as f:
        #     data_list = f.read().split('\n')[:-1]
        # return data_list
        data_list = os.listdir(os.path.join(list_path,'A'))
        return data_list

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
    
    @staticmethod
    def _normalize(img, mean=[0.485, 0.456, 0.406], std=[1, 1, 1]):
        # return img
        im = img.astype(np.float32, copy=False)
        mean = np.array(mean)[np.newaxis, np.newaxis, :]
        std = np.array(std)[np.newaxis, np.newaxis, :]
        im = im / 255.0
        im -= mean
        im /= std
        return im



class TestReader(CDReader):
    def __init__(self, path_root, mode, en_edge=False):
        super(TestReader, self).__init__(path_root, mode, en_edge)

        self.data_name = os.path.split(path_root)[-1]

        self.file_name = []
        if self.en_edge:
            for _file in self.data_list:
                self.sst1_images.append(os.path.join(f"{self.path_root}/A", _file))
                self.sst2_images.append(os.path.join(f"{self.path_root}/B", _file))
                self.sst1_edge.append(os.path.join(f"{self.path_root}/AEdge", _file))
                self.sst2_edge.append(os.path.join(f"{self.path_root}/BEdge", _file))
                self.gt_images.append(os.path.join(f"{self.path_root}/label", _file))
                self.file_name.append(_file)
        else:
            for _file in self.data_list:
                self.sst1_images.append(os.path.join(f"{self.path_root}/A", _file))
                self.sst2_images.append(os.path.join(f"{self.path_root}/B", _file))
                self.gt_images.append(os.path.join(f"{self.path_root}/label", _file))
                self.file_name.append(_file)


    def __getitem__(self, index):
        # print(self.data_list[index])
        A_path = self.sst1_images[index]
        B_path = self.sst2_images[index]
        lab_path = self.gt_images[index]

        A_img = self._normalize(np.array(Image.open(A_path)))
        B_img = self._normalize(np.array(Image.open(B_path)))

        if self.en_edge:
            AEdge_path = self.sst1_edge[index]
            BEdge_path = self.sst2_edge[index]
            edge1 = cv2.imread(AEdge_path, cv2.IMREAD_UNCHANGED)
            edge2 = cv2.imread(BEdge_path, cv2.IMREAD_UNCHANGED)
            A_img = np.concatenate((A_img, edge1[..., np.newaxis]), axis=-1)  # 将两个时段的数据concat在通道层
            B_img = np.concatenate((B_img, edge2[..., np.newaxis]), axis=-1)  # 将两个时段的数据concat在通道层
        # w, h, _ = A_img.shape

        sst1 = tfs.ToTensor()(A_img)
        sst2 = tfs.ToTensor()(B_img)

        gt = np.array(Image.open(lab_path))
        if (len(gt.shape) == 3):
            gt = one_hot_it(gt, self.label_info)
        #gt = np.argmax(gt,axis=2)
        else:
            gt = np.array((gt != 0),dtype=np.int8)
            gt = self.label_color[gt]
            
        gt = np.transpose(gt, [2, 0, 1])
        gt = torch.from_numpy(gt).type(torch.float32)

        return sst1, sst2, gt, self.file_name[index]

    def __len__(self):
        return self.data_num


def detect_building_edge(data_path, save_pic_path):
    canny_low = 180
    canny_high = 210
    hough_threshold = 64
    hough_minLineLength = 16
    hough_maxLineGap = 3
    hough_rho = 1
    hough_theta = np.pi / 180
    image_names=os.listdir(data_path)
    for image_name in image_names:
        img=cv2.imread(os.path.join(data_path, image_name))
        shape=img.shape[:2]
        img_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        edges=cv2.Canny(img_gray,canny_low,canny_high)
        lines=cv2.HoughLinesP(edges,hough_rho,hough_theta,hough_threshold,hough_minLineLength,hough_maxLineGap)
        line_pic = np.zeros(shape, np.uint8)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_pic, (x1, y1), (x2, y2), 1, thickness=1)
        cv2.imwrite(os.path.join(save_pic_path, image_name),line_pic)


if __name__ == "__main__":
    dataset_path = '/mnt/data/Datasets/CLCD'
    x = np.random.random([4,4,3])
    mean = np.std(x, axis=(0,1))
    mean = np.array(mean)[np.newaxis, np.newaxis, :]
    print(x)
    print(mean)

    