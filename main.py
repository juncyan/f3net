# 调用官方库及第三方库
import torch
from torch.utils.data import DataLoader
import numpy as np
import datetime
import platform
import random
import os

# 基础功能
from dataset.CDReader import CDReader, TestReader
from work.train import train

# 模型导入
from cd_models.mscanet.model import MSCANet
from cd_models.dminet import DMINet
from cd_models.dsamnet import DSAMNet


from common import Args

# dataset_name = "GVLM_CD"
# dataset_name = "LEVIR_CD"
# dataset_name = "CLCD"
dataset_name = "SYSU_CD"
dataset_path = '/mnt/data/Datasets/{}'.format(dataset_name)
num_classes = 2
batch_size = 4
num_epochs = 100 


def seed_torch(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    # 代码运行预处理
    seed_torch(32765)
    torch.cuda.empty_cache()
    torch.cuda.init()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # mode:["train","eval","test"] or [1,2,3]

    model = DSAMNet()


    model_name = model.__str__().split("(")[0]
    args = Args('output/{}'.format(dataset_name.lower()), model_name)
    args.data_name = dataset_name
    args.num_classes = num_classes
    args.batch_size = batch_size
    args.iters = num_epochs
    args.pred_idx = 0

    # pred_data = Reader_Only_Image()
    eval_data = CDReader(path_root = dataset_path, mode="val", en_edge=False)
    train_data = CDReader(path_root = dataset_path, mode="train", en_edge=False)
    
    # dataloader_pred = DataLoader(pred_data, batch_size, num_workers=1)
    dataloader_eval = DataLoader(dataset=eval_data, batch_size=args.batch_size, num_workers=16,
                                 shuffle=False, drop_last=True)
    dataloader_train = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=16,
                                  shuffle=True, drop_last=True)
    
    test_data = TestReader(path_root = dataset_path, mode="test", en_edge=False)
    dataloader_test = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=0,
                                  shuffle=True, drop_last=True)
    
   
    
