import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import *
import numpy as np


class FFA(nn.Module):
    # fEATURE FUSING AND AGGREGATION 
    def __init__(self, in_channel, kernels=7):
        super().__init__()
        self.cbr1 = ConvBNReLU(2 * in_channel, in_channel, 3, 1)
        self.lkc = DepthwiseConvBN(in_channel, in_channel, kernels)
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

class CISConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=3, groups=1, dilation_set=4,
                 bias=False):
        super(CISConv, self).__init__()
        self.prim = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=dilation, dilation=dilation,
                              groups=groups * dilation_set, bias=bias)
        self.prim_shift = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=2 * dilation, dilation=2 * dilation,
                                    groups=groups * dilation_set, bias=bias)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=bias)

        self.groups = groups

    def forward(self, x):
        x_split = (z.chunk(2, dim=1) for z in x.chunk(self.groups, dim=1))
        x_merge = torch.cat(tuple(torch.cat((x2, x1), dim=1) for (x1, x2) in x_split), dim=1)
        x_shift = self.prim_shift(x_merge)
        return self.prim(x) + self.conv(x) + x_shift
    

class CFDF(nn.Module):
    def __init__(self, in_channels, out_channels, kernels=7):
        super().__init__()
        self.inter_dim = in_channels // 2
        self.zip_ch = ConvBNReLU(in_channels, self.inter_dim, 3)

        self.native = nn.Sequential(DepthwiseConvBN(self.inter_dim, self.inter_dim, kernels, 3), nn.GELU(), ConvBNReLU(self.inter_dim, self.inter_dim//2, 1))

        self.aux = nn.Sequential(
            CISConv(self.inter_dim//2, self.inter_dim//2, 3, 1, padding=1, groups=int(self.inter_dim / 8), dilation=1,bias=False),
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
        y = torch.cat([y1, y2], 1)
        
        return self.outcbr(y)




if __name__ == "__main__":
    print("spp")
    x = torch.rand([1, 16, 256, 256]).cuda()
    