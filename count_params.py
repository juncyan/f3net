# 调用官方库及第三方库
import torch
from torch.utils.data import DataLoader
import numpy as np
import datetime
import platform
import random
import os
import shutil
import glob

# dataset_name = "GVLM_CD_d"
# dataset_name = "LEVIR_c"
# dataset_name = "CLCD"
dataset_name = "SYSCD_d"

dataset_path = '/mnt/data/Datasets/{}'.format(dataset_name)

with open(f"{dataset_path}/test_list.txt", 'r') as f:
    data_list = f.read().split('\n')[:-1]

dst_dir = f"/mnt/data/Results/{dataset_name}"

dsta = f"{dst_dir}/A"
dstb = f"{dst_dir}/B"
dstl = f"{dst_dir}/label"

os.makedirs(dsta)
os.makedirs(dstb)
os.makedirs(dstl)

for d in data_list:
    shutil.copy(f"{dataset_path}/A/{d}", f"{dsta}/{d}")
    shutil.copy(f"{dataset_path}/B/{d}", f"{dstb}/{d}")
    shutil.copy(f"{dataset_path}/label/{d}", f"{dstl}/{d}")
            
            