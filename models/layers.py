import torch
import torch.nn as nn
import torch.nn.functional as F


class PSConv(nn.Module):
    # refers: https://github.com/d-li14/PSConv
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=3, groups=1, dilation_set=4,
                 bias=False):
        super(PSConv, self).__init__()
        self.prim = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=dilation, dilation=dilation,
                              groups=groups * dilation_set, bias=bias)
        self.prim_shift = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=2 * dilation, dilation=2 * dilation,
                                    groups=groups * dilation_set, bias=bias)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=bias)

        def backward_hook(grad):
            out = grad.clone()
            out[self.mask] = 0
            return out

        self.mask = torch.zeros(self.conv.weight.shape).bool().cuda()
        _in_channels = in_ch // (groups * dilation_set)
        _out_channels = out_ch // (groups * dilation_set)
        for i in range(dilation_set):
            for j in range(groups):
                self.mask[(i + j * groups) * _out_channels: (i + j * groups + 1) * _out_channels,
                i * _in_channels: (i + 1) * _in_channels, :, :] = 1
                self.mask[((i + dilation_set // 2) % dilation_set + j * groups) *
                          _out_channels: ((i + dilation_set // 2) % dilation_set + j * groups + 1) * _out_channels,
                i * _in_channels: (i + 1) * _in_channels, :, :] = 1
        self.conv.weight.data[self.mask] = 0
        self.conv.weight.register_hook(backward_hook)
        self.groups = groups

    def forward(self, x):
        x_split = (z.chunk(2, dim=1) for z in x.chunk(self.groups, dim=1))
        x_merge = torch.cat(tuple(torch.cat((x2, x1), dim=1) for (x1, x2) in x_split), dim=1)
        x_shift = self.prim_shift(x_merge)
        return self.prim(x) + self.conv(x) + x_shift
    
class ConvBNReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)

        
        self._batch_norm = nn.BatchNorm2d(out_channels)
        self._relu = nn.ReLU()

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._relu(x)
        return x


class ConvBN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()
        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)
        
        self._batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        return x


class ConvReLUPool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1)
        self._relu = nn.ReLU()
        self._max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self._relu(x)
        x = self._max_pool(x)
        return x


class SeparableConvBNReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 pointwise_bias=None,
                 **kwargs):
        super().__init__()
        self.depthwise_conv = ConvBN(
            in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            **kwargs)
        
        self.piontwise_conv = ConvBNReLU(
            in_channels,
            out_channels,
            kernel_size=1,
            groups=1,
            bias=pointwise_bias)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.piontwise_conv(x)
        return x


class DepthwiseConvBN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()
        self.depthwise_conv = ConvBN(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            **kwargs)

    def forward(self, x):
        x = self.depthwise_conv(x)
        return x


class AuxLayer(nn.Module):
    """
    The auxiliary layer implementation for auxiliary loss.

    Args:
        in_channels (int): The number of input channels.
        inter_channels (int): The intermediate channels.
        out_channels (int): The number of output channels, and usually it is num_classes.
        dropout_prob (float, optional): The drop rate. Default: 0.1.
    """

    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 dropout_prob=0.1,
                 **kwargs):
        super().__init__()

        self.conv_bn_relu = ConvBNReLU(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=3,
            padding=1,
            **kwargs)

        self.dropout = nn.Dropout(p=dropout_prob)

        self.conv = nn.Conv2d(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=1)

    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x


class PAM(nn.Module):
    """
    Position attention module.
    Args:
        in_channels (int): The number of input channels.
    """

    def __init__(self, in_channels):
        super().__init__()
        mid_channels = in_channels // 8
        self.mid_channels = mid_channels
        self.in_channels = in_channels

        self.query_conv = nn.Conv2d(in_channels, self.mid_channels, 1, 1)
        self.key_conv = nn.Conv2d(in_channels, self.mid_channels, 1, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1, 1)

        # self.gamma = self.create_parameter(
        #     shape=[1],
        #     dtype='float32',
        #     default_initializer=nn.initializer.Constant(0))
        self.gamma = torch.nn.parameter.Parameter(data=torch.Tensor([0.0]), requires_grad=True).float()

    def forward(self, x):
        x_shape = x.shape

        # query: n, h * w, c1
        query = self.query_conv(x)
        query = torch.reshape(query, (x_shape[0], self.mid_channels, -1))
        query = torch.transpose(query, 2, 1)

        # key: n, c1, h * w
        key = self.key_conv(x)
        key = torch.reshape(key, (x_shape[0], self.mid_channels, -1))

        # sim: n, h * w, h * w
        sim = torch.bmm(query, key)
        sim = F.softmax(sim, dim=-1)

        value = self.value_conv(x)
        value = torch.reshape(value, (x_shape[0], self.in_channels, -1))
        sim = torch.transpose(sim, 2, 1)

        # feat: from (n, c2, h * w) -> (n, c2, h, w)
        feat = torch.bmm(value, sim)
        feat = torch.reshape(feat,
                              (x_shape[0], self.in_channels, x_shape[2], x_shape[3]))

        out = self.gamma * feat + x
        return out


class CAM(nn.Module):
    """
    Channel attention module.
    Args:
        in_channels (int): The number of input channels.
    """

    def __init__(self, channels):
        super().__init__()

        self.channels = channels
        self.gamma = torch.nn.parameter.Parameter(data=torch.Tensor([0.0]), requires_grad=True)
        # self.create_parameter(
        #     shape=[1],
        #     dtype='float32',
        #     default_initializer=nn.initializer.Constant(0))

    def forward(self, x):
        x_shape = x.size()
        # query: n, c, h * w
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

        out = self.gamma * feat + x
        return out
