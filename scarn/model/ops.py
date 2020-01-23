import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class MeanShift(nn.Module):
    def __init__(self, sub, mean_gray=0.437):
        super(MeanShift, self).__init__()

        sign = -1 if sub else 1
        self.mean_gray = mean_gray * sign

    def forward(self, x):
        x = x + self.mean_gray
        return x


def init_weights(modules):
    pass


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
