import cv2
import csv
import os
import glob
import pandas as pd
import numpy as np


__all__ = ["writer_csv", 'save_numpy_as_csv', 'reader_csv', 'read_excel', 'read_csv', 'scale_image',
           'one_hot_it']

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

def generate_matrics_csv(base_path):
    data_names = os.listdir(base_path)
    
    for data_name in data_names:
        data = dict()
        keys = set()

        dmpath = os.path.join(base_path, data_name)
        if not os.path.isdir(dmpath):
            continue
        data_models = os.listdir(dmpath)
        
        for idx, data_model in enumerate(data_models):
            metrics_path = glob.glob(os.path.join(f"{dmpath}/{data_model}", '*.csv'))[0]
            
            metrics = pd.read_csv(metrics_path) 
            mkeys =  metrics.keys()
            for k in mkeys:
                keys.add(k)
            try:
                max_idx = metrics["miou"].idxmax()
            except:
                max_idx = metrics["mIoU"].idxmax()
            
            best_m = metrics.iloc[max_idx]
            model_name = data_model.split("202")[0]
            d = {f"{model_name}{idx}":dict(best_m)}
            data.update(d)
        
        keys = list(keys)
        keys.sort()
        indexs = []
        dc = {}
        for k in keys:
            dc.update({k:[]})

        for k in data.keys():
            d = data[k]
            indexs.append(k)
            for sk in keys:
                if sk in d.keys():
                    v = d[sk]
                else:
                    v = 0.
                dc[sk].append(v)
        
        dc = pd.DataFrame(dc, index=indexs)
        dc.to_csv(f"{base_path}/{data_name}_metrics.csv")

def cls_count(label):
    cls_nums = []
    color_label = np.array([[0, 0, 0], [255, 255, 255], [0, 128, 0], [0, 0, 128]])
    for info in color_label:
        color = info
        # print("label:\n", label.shape,label)
        # print("color:\n", color)
        equality = np.equal(label, color)
        matrix = np.sum(equality, axis=-1)
        nums = np.sum(matrix == 3)
        cls_nums.append(nums)
    return cls_nums

if __name__ == "__main__":
    print("data_reader.utils run")
    x= ['1','3','5','7','9','0','2','4','6','8']
    x = reader_csv("../colors.csv")
    print(x)
    # with open("../colors.csv","w",newline="") as csv_file:
    #     filewrite = csv.writer(csv_file)
    #     filewrite.writerow([" ","0","1"])
    #     filewrite.writerow(["R","0","1.1"])
    #     filewrite.writerow(["G","0.5","1.5"])
    #     filewrite.writerow(["B","0.7","1.7"])
