import cv2
import numpy as np
import os

dataname = "sysu"
lab_dir = r"C:\Users\p2316861\Desktop\datasets\{}\label".format(dataname)
pre_dir = r"C:\Users\p2316861\Desktop\Results\{}".format(dataname)
dst_dir = r"C:\Users\p2316861\Desktop\figures\{}".format(dataname)
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

label_info = np.array([[0,0,0],[255,255,255]])
color = np.array([[0,0,0],[255,255,255],[0,128,0],[0,0,128]])
def one_hot_it(label, label_info):
    semantic_map = []
    for info in label_info:
        color = info

        # print("label:\n", label.shape,label)
        # print("color:\n", color)
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    return np.stack(semantic_map, axis=-1,dtype=np.int32)

if __name__ == "__main__":
    print("test")
    labells = os.listdir(lab_dir)
    models = os.listdir(pre_dir)
    for model in models:
        dst_path = os.path.join(dst_dir, model)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        pred_path = os.path.join(pre_dir, model)
        prels = os.listdir(pred_path)
        for f in prels:
            if f in labells:
                lab = cv2.imread(os.path.join(lab_dir, f))
                pre = cv2.imread(os.path.join(pred_path, f))

                lab = one_hot_it(lab, label_info)
                pre = one_hot_it(pre, label_info)

                lab = np.argmax(lab, axis=-1)
                pre = np.argmax(pre, axis=-1)

                flag = lab - pre
                pre[flag == -1] = 2
                pre[flag == 1] = 3
                img = color[pre]
                cv2.imwrite(os.path.join(dst_path, f), img)
