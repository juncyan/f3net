# 调用官方库及第三方库
import torch
from torch.utils.data import DataLoader
import numpy as np
import datetime
import platform
import random
import os

# 基础功能
from dataset.dataloader import DataReader
from dataset.CDReader import CDReader, TestReader
from work.train import train
from work.predict import predict

# 模型导入
from core.models.unet import UNet
from core.models.deeplabv3_plus import DeepLabV3Plus
from core.models.pspnet import PSPNet
from core.models.denseaspp import DenseASPP
from core.models.hrnet import HRNet
from core.models.dfanet import DFANet
from core.models.fcn import FCN32s
from cd_models.mscanet.model import MSCANet  #CropLandCD
from cd_models.aernet import AERNet
from cd_models.a2net import LightweightRSCDNet
from cd_models.ussfcnet.ussfcnet import USSFCNet
from cd_models.dtcdscn import DTCDSCNet
from cd_models.changeformer import ChangeFormerV6
from cd_models.dminet import DMINet
from cd_models.siamunet_diff import SiamUnet_diff
from cd_models.dsamnet import DSAMNet
from cd_models.bit_cd import BIT_CD
from cd_models.SNUNet import SNUNet
from cd_models.ResUnet import ResUnet
from cd_models.icifnet import ICIFNet
from cd_models.dsifn import DSIFN
from cd_models.bisrnet import BiSRNet
from common import Args



# class parameter:
#     lr = params["lr"]
#     momentum = params["momentum"]
#     weight_decay = params["weight_decay"]
#     num_epochs = num_epochs
#     batch_size = batch_size

# dataset_name = "GVLM_CD_d"
# dataset_name = "LEVIR_c"
dataset_name = "CLCD"
# dataset_name = "SYSCD_d"
dataset_path = '/mnt/data/Datasets/{}'.format(dataset_name)



if __name__ == "__main__":
    # 代码运行预处理
    torch.cuda.empty_cache()
    torch.cuda.init()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # mode:["train","eval","test"] or [1,2,3]

    # 模型选择
    # model = DeepLabV3Plus(6, 2, backbone="xception", pretrained_base=False,dilated=True)
    # model = DenseASPP(num_classes, pretrained_base=False)
    # model = LEDNet(args.num_classes,"resnet50")
    # model = BiSeNet(args.num_classes)
    # model = DFANet(num_classes, norm_layer=torch.nn.BatchNorm2d)
    # model = UNet(6, num_classes)
    # model = PSPNet(num_classes, pretrained_base=False)
    # model = HRNet(num_classes)
    # model = FCN32s(num_classes, aux=True, pretrained_base=False)
    # model = CDNet(img_size=512)
    # model = BiSRNet()
    # model = USSFCNet(in_ch=3).cuda()
    # model = DTCDSCNet()
    # model = SUNnet(4,out_size=[512,512]).cuda()
    # model = DMINet()
    # model = UChange()
    # model = DSAMNet(2).cuda()
    # model = BIT_CD().cuda()
    # model = ICIFNet(2).cuda()
    # model = SUNnet()
    model = DSIFN()

    test_data = TestReader(dataset_path, mode="test",en_edge=False)
    model = model.cuda()
    weight_path = r"/home/jq/Code/torch/output/clcd/DSIFN_2023_11_13_11/iter_200.pth"
    predict(model, test_data, weight_path,test_data.data_name,2)
    # x = torch.rand([1,3,256,256]).cuda()
    # y = model(x,x)
    # print(y.shape)    
