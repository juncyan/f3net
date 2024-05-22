import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import *
from models.resnet import resnet34, resnet50
from .module import *

class F3Net(nn.Module):
    def __init__(self,in_channels=3, num_classes=2):
        super().__init__()
        kernels = 7
        self.in_channels = in_channels
        self.encode1 = backbone(in_channels)#(3, 34)
        self.encode2 = backbone(in_channels)

        self.lkff1 = FFA(64, kernels)
        self.lkff2 = FFA(128, kernels)
        self.lkff3 = FFA(256, kernels)
        self.lkff4 = FFA(512, kernels)

        self.ppm = PMM(512)
        
        self.up1 = CFDF(1024, 256, kernels)
        self.up2 = CFDF(512, 128, kernels)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        
        self.classier = nn.Sequential(ConvBN(64, num_classes,3), nn.Sigmoid())

    def forward(self, x1, x2=None):
        if x2 == None and x1.shape[1] == 2* self.in_channels:
            x1 ,x2 = x1[:, :self.in_channels, :, :], x1[:, self.in_channels:, :, :]
        self.feature1 = self.encode1(x1)
        self.feature2 = self.encode2(x2)
        
        self.augf1 = self.lkff1(self.feature1[0], self.feature2[0])
        self.augf2 = self.lkff2(self.feature1[1], self.feature2[1])
        self.augf3 = self.lkff3(self.feature1[2], self.feature2[2])
        self.augf4 = self.lkff4(self.feature1[3], self.feature2[3])
        
        self.flast = self.ppm(self.augf4)

        y = self.up1(self.flast, self.augf4)
        y = self.up2(y, self.augf3)
        y = self.up3(y, self.augf2)
        y = self.up4(y, self.augf1)
        # y = nn.functional.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        return self.classier(y)


class BTF(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.zip = ConvBNReLU(2 * in_channels, in_channels, 3)
    
    def forward(self, x1, x2):
        x = torch.concat([x1, x2], 1)
        y = self.zip(x)
        return y

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.bilinear = bilinear
        if bilinear:
            #self.up = nn.functional.interpolate(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.Conv2dTranspose(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        if(self.bilinear):
            x1 =nn.functional.interpolate(x1,scale_factor=2, mode='bilinear', align_corners=True)
        else:
            x1 = self.up(x1)
        # print("x2.size():", x2.shape)
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.concat([x2, x1], dim=1)
        return self.conv(x)

class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            #nn.MaxPool2d(2),
            nn.Conv2d(in_channels,in_channels,3,2,1),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class backbone(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        # factor = 2 #if bilinear else 1
        # self.down4 = Down(512, 512)
    
    def forward(self, x):
        res = []
        y = self.inc(x)
        res.append(y)
        y = self.down1(y)
        res.append(y)
        y = self.down2(y)
        res.append(y)
        y = self.down3(y)
        # res.append(y)
        # y = self.down4(y)
        res.append(y)
        return res



if __name__ == "__main__":
    print("test DACDNet")
    x = torch.rand([1,6,512,512]).cuda()
    
