import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import *
import numpy as np


class LargeKernelBlock(nn.Module):
    def __init__(self, dims, inter, kernels=7):
        super().__init__()
        self.proj1 = DepthwiseConvBN(dims, dims, kernels)
        self.sp1 = SeparableConvBNReLU(dims, inter, 3)
        # self.proj2 = DepthwiseConvBN(inter, inter, kernels)
        self.sp2 = SeparableConvBNReLU(inter, dims, 3)
        self.activate = nn.ReLU()
    
    def forward(self, x):
        y = self.proj1(x)
        y = nn.GELU()(y)
        y = self.sp1(y)
        # y = self.proj2(y)
        y = self.sp2(y)
        y = x + y
        return self.activate(y)


class FSA(nn.Module):
    # fEATURE SIFTING AND AGGREGATION 
    def __init__(self, in_channel, kernel=7):
        super().__init__()
        self.cbr1 = ConvBNReLU(2 * in_channel, in_channel, 3, 1)
        self.lkc = DepthwiseConvBN(in_channel, in_channel, kernel)
        self.cbr2 = ConvBNReLU(in_channel, in_channel, 3, 1)

        self.att1 = nn.Sequential(ConvBN(in_channel, 1, 3), nn.Sigmoid())
        self.att2 = nn.Sequential(ConvBN(in_channel, 1, 3), nn.Tanh())

        self.att = ConvBNReLU(2,1,3)

        self.outcbn = ConvBNReLU(in_channel, in_channel, 3)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], 1)
        y = self.cbr1(x)
        y = self.lkc(y)
        y = nn.GELU()(y)
        y = self.cbr2(y)

        y1 = self.att1(y)
        y2 = self.att2(y)

        egg = torch.concat([y1, y2], 1)
        egg = self.att(egg)

        y = egg * y
        y = self.outcbn(y)
        return y

class LKAFF(nn.Module):
    # Large kernels feature fusion
    def __init__(self, in_channels, kernels=7):
        super().__init__()
        dims = in_channels * 2
        self.zip_channels = ConvBNReLU(dims, in_channels, 1)
        self.lfc = DepthwiseConvBN(in_channels, in_channels, kernels)
    
        self.sa = nn.Sequential(ConvBN(2, 1, 3), nn.Sigmoid())

        self.outcbn = ConvBNReLU(in_channels, in_channels, 3)

    def forward(self, x1, x2):
        x = torch.concat([x1, x2], 1)
        x = self.zip_channels(x)
        y = self.lfc(x)
        y = nn.GELU()(y)

        max_feature = torch.max(y, dim=1, keepdim=True).values
        mean_feature = torch.mean(y, dim=1, keepdim=True)
        
        att_feature = torch.concat([max_feature, mean_feature], dim=1)
        # y1 = max_feature * y
        # y2 = mean_feature * y
        # att_feature = torch.concat([y1, y2], dim=1)
        y = self.sa(att_feature)
        y = y * x
        y = self.outcbn(y)
        return y


class SFF(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.channels = in_channels
        self.se = nn.Sequential(nn.Linear(2*in_channels, in_channels//4, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_channels//4, 2*in_channels, bias=False),
                                nn.Sigmoid())
        self.cbn = ConvBNReLU(2 * in_channels, in_channels, 3)
        self.gamma = torch.nn.parameter.Parameter(data=torch.Tensor([0.0]), requires_grad=True)

    def forward(self, x1, x2):
        x_shape = x1.shape
        x = torch.cat([x1, x2], dim=1)
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(x_shape[0],2*x_shape[1])
        se = self.se(avg_pool).view(x_shape[0],2*x_shape[1],1,1)
        x = x * se.expand_as(x)

        x = self.cbn(x)

        query = torch.reshape(x, (x_shape[0], self.channels, -1))
        # key: n, h * w, c
        key = torch.reshape(x, (x_shape[0], self.channels, -1))
        # torch.transpose(x,)
        key = torch.transpose(key, 2, 1)

        # sim: n, c, c
        sim = torch.bmm(query, key)
        # The danet author claims that this can avoid gradient divergence
        sim = torch.max(sim, dim=-1, keepdim=True).values.tile(
            (1, 1, self.channels)) - sim
        sim = F.softmax(sim, dim=-1)

        # feat: from (n, c, h * w) to (n, c, h, w)
        value = torch.reshape(x, (x_shape[0], x_shape[1], -1))
        feat = torch.bmm(sim, value)
        feat = torch.reshape(feat, x_shape)

        y = self.gamma * feat + x
        return y

class PMM(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out

class PyramidPoolingModule(nn.Module):
    def __init__(self, pyramids=[1,2,3,6]):
        super(PyramidPoolingModule, self).__init__()
        self.pyramids = pyramids

    def forward(self, input):
        feat = input
        height, width = input.shape[2:]
        for bin_size in self.pyramids:
            x = F.adaptive_avg_pool2d(input, output_size=bin_size)
            x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=True)
            feat  = feat + x
        return feat
    
class SFFUp(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.inter_dim = in_channels // 2
        self.zip_ch = ConvBNReLU(in_channels, self.inter_dim, 3)

        self.native = ConvBNReLU(self.inter_dim, self.inter_dim//2, 3, 1)

        self.ppmn = CAM(self.inter_dim//2)

        self.aux = nn.Sequential(
            PSConv(self.inter_dim//2, self.inter_dim//2, 3, 1, padding=1, groups=int(self.inter_dim / 8), dilation=1,bias=False),
            nn.BatchNorm2d(self.inter_dim//2),
            nn.ReLU(inplace=True))

        self.outcbr = ConvBNReLU(self.inter_dim, out_channels, 3, 1)

    def forward(self, x1, x2):
        if x1.shape != x2.shape:
            x1 = F.interpolate(x1, x2.shape[-2:], mode='bilinear')

        x = torch.concat([x2, x1], dim=1)
        y = self.zip_ch(x)
        y1 = self.native(y)
        y2 = self.aux(y1)
        y1 = self.ppmn(y1)
        y = torch.cat([y1, y2], 1)
        
        return self.outcbr(y)

class LSFFUp(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.inter_dim = in_channels // 2
        self.zip_ch = ConvBNReLU(in_channels, self.inter_dim, 3)

        self.native = nn.Sequential(DepthwiseConvBN(self.inter_dim, self.inter_dim, 7, 3), nn.GELU(), ConvBNReLU(self.inter_dim, self.inter_dim//2, 1))

        self.ppmn = CAM(self.inter_dim//2)

        self.aux = nn.Sequential(
            PSConv(self.inter_dim//2, self.inter_dim//2, 3, 1, padding=1, groups=int(self.inter_dim / 8), dilation=1,bias=False),
            nn.BatchNorm2d(self.inter_dim//2),
            nn.ReLU(inplace=True))

        self.outcbr = ConvBNReLU(self.inter_dim, out_channels, 3, 1)

    def forward(self, x1, x2):
        if x1.shape != x2.shape:
            x1 = F.interpolate(x1, x2.shape[-2:], mode='bilinear')

        x = torch.concat([x2, x1], dim=1)
        y = self.zip_ch(x)
        y1 = self.native(y)
        y2 = self.aux(y1)
        y1 = self.ppmn(y1)
        y = torch.cat([y1, y2], 1)
        
        return self.outcbr(y)

class SpikingNeuron(nn.Module):
    def __init__(self, threshold=0.5, decay=0.25):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.membrance_potential = 0.
    
    def forward(self, x):
        self.membrance_potential += x
        spike = (self.membrance_potential >= self.threshold).float()
        self.membrance_potential = self.membrance_potential * (1 - spike)* self.decay
        return spike


if __name__ == "__main__":
    print("spp")
    x = torch.rand([1, 16, 256, 256]).cuda()
    