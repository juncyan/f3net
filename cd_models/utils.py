import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn import Uniform


__all__ = ["DepthWiseConv2D", "SeparableConvBNReLU","ConvBn","ConvBnReLU", "DropPath"]


class DepthWiseConv2D(nn.Module):
    def __init__(self, in_channels, kernel, stride, bias=True):
        super(DepthWiseConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel, stride=stride, padding=kernel//2, groups=in_channels, bias=bias)

    def forward(self, x):
        y = self.conv(x)
        return y


class SeparableConvBNReLU(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride, pointwise_bias=True):
        super().__init__()
        self.depthwise_conv = nn.Sequential(DepthWiseConv2D(in_channels, kernel_size, stride=stride),
                                           nn.BatchNorm2d(in_channels))

        self.piontwise_conv = ConvBnReLU(in_channels,out_channels,kernel_size=1,stride=1,bias=pointwise_bias)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.piontwise_conv(x)
        return x


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        return y

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,stride, padding, dilation, groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.relu(y)
        return y

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob <= 0. or not self.training:
            return x
        keep_prob = torch.to_tensor(1-self.drop_prob)
        shape = (x.shape[0],) + (1, ) *(x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape)
        random_tensor = torch.floor(random_tensor)
        y = torch.divide(x, keep_prob) * random_tensor
        return y

    @property
    def drop_path(self):
        if self.drop_prob == 0 or not self.training:
            return 'Identity'
        return 'Drop_Path'

# class SELayer(nn.Module):
#     def __init__(self, ch, reduction_ratio=16):
#         super(SELayer, self).__init__()
#         self.pool = nn.AdaptiveAvgPool2D(1)
#         stdv = 1.0 / torch.sqrt(ch)
#         c_ = ch // reduction_ratio
#         weight_attr_1 = torch.ParamAttr(initializer=Uniform(-stdv, stdv))
#         self.squeeze = nn.Linear(ch, c_, weight_attr=weight_attr_1, bias=None)

#         stdv = 1.0 / torch.sqrt(c_)
#         weight_attr_2 = torch.ParamAttr(initializer=Uniform(-stdv, stdv))
#         self.extract = nn.Linear(c_,ch, weight_attr=weight_attr_2, bias=None)

#     def forward(self, x):
#         out = self.pool(x)
#         out = torch.squeeze(out, axis=[2, 3])
#         out = self.squeeze(out)
#         out = F.relu(out)
#         out = self.extract(out)
#         out = F.sigmoid(out)
#         out = torch.unsqueeze(out, axis=[2, 3])
#         y = out * x
#         return y


if __name__ == "__main__":
    print('utils')
    x = torch.rand([5,3,16,16], dtype=torch.float32).cuda()
    y = torch.to_tensor(0.2)
    print(x[0,0,0,0], x[0,0,0,0].item())

    # m = DepthWiseConv2D(3,1).to("gpu:0")
    # y = m(x)
    # print(x == y)