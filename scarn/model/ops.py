import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def init_weights(modules):
    pass


def yCbCr2rgb(input_im):
    im_flat = input_im.contiguous().view(-1, 3).float()
    mat = torch.tensor([[1.164, 1.164, 1.164],
                        [0, -0.392, 2.017],
                        [1.596, -0.813, 0]])
    bias = torch.tensor([-16.0 / 255.0, -128.0 / 255.0, -128.0 / 255.0])
    temp = (im_flat + bias).mm(mat)
    out = temp.view(3, list(input_im.size())[1], list(input_im.size())[2])
    return out


def rgb2yCbCr(input_im):
    im_flat = input_im.contiguous().view(-1, 3).float()
    mat = torch.tensor([[0.257, -0.148, 0.439],
                        [0.564, -0.291, -0.368],
                        [0.098, 0.439, -0.071]])
    bias = torch.tensor([16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0])
    temp = im_flat.mm(mat) + bias
    out = temp.view(3, input_im.shape[1], input_im.shape[2])
    return out
   

# 这个Module的作用就是每个通道的颜色减去某个数，目的应该是将各个通道的值映射到-1到1之间，提高训练速度
class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()

        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign

        # 输入通道数为3，输出通道数为3，卷积核大小是1*1，stride是进行一次卷积后特征图滑动1格，padding是最边缘补0数
        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        # 用3*3的对角矩阵初始化卷积核
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        # bias记录的是rgb的经验均值
        self.shifter.bias.data = torch.Tensor([r, g, b])

        # Freeze the mean shift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x


# 这个Block的作用就是加一个conv+relu
class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )

        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        return out


# 残差网络，conv+relu+conv学习到残差，然后加到原始输入中，最后再一个relu。它可以看成是一种高级的卷积操作
class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


# carn作者提出了一种残差-E模块，用来提高效率。它可以看成是对上面那个module的替代
class EResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, out_channels,
                 group=1):
        super(EResidualBlock, self).__init__()

        # groups大于1就使用分组卷积，比如如果是2，输入通道和输出通道都是原来的一半，个由一个卷积去负责。
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )

        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


# 将多个通道放大
class UpsampleBlock(nn.Module):
    def __init__(self, 
                 n_channels, scale, multi_scale, 
                 group=1):
        super(UpsampleBlock, self).__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, group=group)
        else:
            self.up = _UpsampleBlock(n_channels, scale=scale, group=group)

        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)


class _UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, group=1):
        super(_UpsampleBlock, self).__init__()

        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                # 通过卷积操作将原来的通道数量变成原来的4倍
                modules += [nn.Conv2d(n_channels, 4*n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
                # 再把通道数量还原回原来的数量，但是长宽就变成原来的2倍
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            modules += [nn.Conv2d(n_channels, 9*n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]

        self.body = nn.Sequential(*modules)
        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        return out
