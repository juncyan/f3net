import torch
from torch.utils.data import DataLoader
import numpy as np
import datetime
import platform
import random
import os

# 基础功能
from dataset.CDReader import CDReader
from work.train import train

# 模型导入
from cd_models.mscanet.model import MSCANet
from cd_models.dminet import DMINet
from cd_models.dsamnet import DSAMNet
from f3net.f3net import F3Net


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

    model = F3Net()


    model_name = model.__str__().split("(")[0]
    args = Args('output/{}'.format(dataset_name.lower()), model_name)
    args.data_name = dataset_name
    args.num_classes = num_classes
    args.batch_size = batch_size
    args.iters = num_epochs
    args.pred_idx = 0

    eval_data = CDReader(path_root = dataset_path, mode="val", en_edge=False)
    train_data = CDReader(path_root = dataset_path, mode="train", en_edge=False)
    test_data = CDReader(path_root=dataset_path, mode="test", en_edge=False)

    eval_data = DataLoader(dataset=eval_data, batch_size=args.batch_size, num_workers=16,
                                 shuffle=False, drop_last=True)
    train_data = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=16,
                                  shuffle=True, drop_last=True)

    test_data = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=0,
                                  shuffle=True, drop_last=True)

    train(model, train_data, eval_data, test_data)
    
   
    
