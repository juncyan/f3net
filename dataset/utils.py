import cv2
import torch
import os
from torchvision import transforms
import matplotlib.pyplot as plt
import datetime
import csv
import random
import pandas as pd
import numpy as np


#create and get label information
def writer_csv(csv_dir,operator="w",headers=None,lists=None):
    with open(csv_dir,operator,newline="") as csv_file:
        f_csv=csv.writer(csv_file)
        if headers!=None:
            f_csv.writerow(headers)
        if lists!=None:
            f_csv.writerows(lists)

def save_numpy_as_csv(scv_dir,d_numpy,fmt="%.4f"):
    assert len(d_numpy.shape) <= 2
    if len(d_numpy.shape)==1:
        d_numpy = np.expand_dims(d_numpy, 0)
    with open(scv_dir,"a") as f:
        np.savetxt(f, d_numpy, fmt=fmt,delimiter=',')

def reader_csv(csv_dir):
    ann = pd.read_csv(csv_dir)
    label = {}
    for iter, row in ann.iterrows():
        label_name = row['name']
        r = row['r']
        g = row['g']
        b = row['b']
        label[label_name] = [int(r), int(g), int(b)]
    return label

def read_excel(path):
    dara_xls = pd.ExcelFile(path)
    data = {}
    for sheet in dara_xls.sheet_names:
        df = dara_xls.parse(sheet_name=sheet,header=None)
        #print(type(df.values))
        data[sheet] = df.values
    return data

def read_csv(csv_dir):
    data = pd.read_csv(csv_dir).values
    return data

def scale_image(input,factor):
    #效果不理想，边缘会有损失，不建议使用 2020/5/17 hjq
    #input.shape=[m,n],output.shape=[m//factor,n//factor]
    #将原tensor压缩factor

    h=input.shape[0]//factor
    w=input.shape[1]//factor

    return cv2.resize(input,(w,h),interpolation=cv2.INTER_NEAREST)

#show or save figure
def sub_2_imshow(tensor,mode,save_folder=None,name=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)

    if save_folder ==None:
        save_folder = args.test_image_dir + datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d") + "/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if name == None:
        name = datetime.datetime.strftime(datetime.datetime.now(), "%H_%M_%S")

    if mode == "plt":
        plt.imshow(image)
        plt.savefig(save_folder + name + ".png")
        plt.pause(1)
    elif mode == "cv2":
        cv2.imwrite(save_folder+name+".png", cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2BGR))
        plt.pause(1)

def sub_imshow(tensor,mode,save_folder=None,name=None):
    assert type(tensor)==torch.Tensor
    if len(tensor.shape)==4:
        for i in range(tensor.shape[0]):
            sub_2_imshow(tensor[i],mode,save_folder=save_folder,name=name)
    else:
        sub_2_imshow(tensor,mode,save_folder=save_folder,name=name)

#mode=["plt.save","cv2.write"] = [0,1]

def imshow(img_data,mode="cv2",save_folder=None,name=None):      #将torch.Tensor数据显示为图片
    assert type(img_data) in [torch.Tensor, tuple, list]
    if type(img_data) == torch.Tensor:
        sub_imshow(img_data,mode,save_folder=save_folder,name=name)
    else:
        for img in img_data:
            sub_imshow(img,mode,save_folder=save_folder,name=name)


if __name__ == "__main__":
    print("data_reader.utils run")
    x= ['1','3','5','7','9','0','2','4','6','8']
    y = random.sample(x,3)
    print(y)
