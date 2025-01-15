import torch
import torchvision.transforms as tfs
import os
import cv2
from PIL import Image
import random
import numpy as np
from torch.utils import data
from skimage import io
import pandas as pd


# label_info = {"0":np.array([0,0,0]), "1":np.array([255,255,255])}
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
    def __init__(self, path_root="./dataset/", mode="train"):
        super(CDReader, self).__init__()
        self.test = mode == "test"
        self.path_root = os.path.join(path_root, mode)

        self.data_list = self._get_list(self.path_root)
        self.data_num = len(self.data_list)

        self.label_info = pd.read_csv(os.path.join(path_root, 'label_info.csv'))
        self.label_color = np.array([[1, 0], [0, 1]])

        self.file_name = []
        self.sst1_images = []
        self.sst2_images = []
        self.gt_images = []

        for _file in self.data_list:
            self.sst1_images.append(os.path.join(self.path_root, "A", _file))
            self.sst2_images.append(os.path.join(self.path_root, "B", _file))
            self.gt_images.append(os.path.join(self.path_root, "label", _file))
            self.file_name.append(_file)

    def __getitem__(self, index):

        A_path = self.sst1_images[index]
        B_path = self.sst2_images[index]
        cd_path = self.gt_images[index]

        A_img = np.array(io.imread(A_path), np.float32)
        B_img = np.array(io.imread(B_path), np.float32)

        gt = np.array(io.imread(cd_path))

        if (len(gt.shape) == 3):
            gt = one_hot_it(gt, self.label_info)
        else:
            gt = np.array((gt != 0), dtype=np.int8)
            gt = self.label_color[gt]

        gt = np.transpose(gt, [2, 0, 1])
        gt = torch.from_numpy(gt).type(torch.float32)

        sst1 = torch.from_numpy(A_img)
        sst2 = torch.from_numpy(B_img)
        if self.test == False:
            return sst1, sst2, gt
        return sst1, sst2, gt, self.file_name[index]

    def __len__(self):
        return self.data_num

    def _get_list(self, list_path):
        data_list = os.listdir(os.path.join(list_path, 'A'))
        return data_list


def detect_building_edge(data_path, save_pic_path):
    canny_low = 180
    canny_high = 210
    hough_threshold = 64
    hough_minLineLength = 16
    hough_maxLineGap = 3
    hough_rho = 1
    hough_theta = np.pi / 180
    image_names = os.listdir(data_path)
    for image_name in image_names:
        img = cv2.imread(os.path.join(data_path, image_name))
        shape = img.shape[:2]
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_gray, canny_low, canny_high)
        lines = cv2.HoughLinesP(edges, hough_rho, hough_theta, hough_threshold, hough_minLineLength, hough_maxLineGap)
        line_pic = np.zeros(shape, np.uint8)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_pic, (x1, y1), (x2, y2), 1, thickness=1)
        cv2.imwrite(os.path.join(save_pic_path, image_name), line_pic)


if __name__ == "__main__":
    dataset_path = '/mnt/data/Datasets/CLCD'
    x = np.random.random([4, 4, 3])
    mean = np.std(x, axis=(0, 1))
    mean = np.array(mean)[np.newaxis, np.newaxis, :]
    print(x)
    print(mean)

