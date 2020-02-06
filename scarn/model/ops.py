import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(modules):
    pass


def yCbCr2rgb(input_im):
    y = input_im[:, 0:1, :, :]
    cb = input_im[:, 1:2, :, :]
    cr = input_im[:, 2:3, :, :]
    r = 1.164 * (y - 16.0 / 255.0) + 1.596 * (cr - 128.0 / 255.0)
    g = 1.164 * (y - 16.0 / 255.0) - 0.392 * (cb - 128.0 / 255.0) - 0.813 * (cr - 128.0 / 255.0)
    b = 1.164 * (y - 16.0 / 255.0) + 2.017 * (cb - 128.0 / 255.0)
    out = torch.cat((r, g, b), 1)
    return out


def rgb2yCbCr(input_im):
    r = input_im[:, 0:1, :, :]
    g = input_im[:, 1:2, :, :]
    b = input_im[:, 2:3, :, :]
    y = r * 0.257 + g * 0.564 + b * 0.098 + 16.0 / 255.0
    cb = -0.148 * r - 0.291 * g + 0.439 * b + 128.0 / 255.0
    cr = 0.439 * r - 0.368 * g - 0.071 * b + 128.0 / 255.0
    out = torch.cat((y, cb, cr), 1)
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

        if group > 0:
            # groups大于1就使用分组卷积，比如如果是2，输入通道和输出通道都是原来的一半，个由一个卷积去负责。
            self.body = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 1, 1, 0),
            )
        else:
            self.body = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, groups=1),
                nn.ReLU(inplace=True),

                nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 1, 1, 0, groups=1),
                nn.ReLU(inplace=True),

                nn.Conv2d(out_channels, out_channels, 1, 1, 0),
            )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class BilinearUpsampleBlock(nn.Module):
    def __init__(self, scale, multi_scale):
        super(BilinearUpsampleBlock, self).__init__()

        if multi_scale:
            self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up3 = nn.Upsample(scale_factor=3, mode='bilinear', align_corners=True)
            self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            self.up5 = nn.Upsample(scale_factor=5, mode='bilinear', align_corners=True)
            self.up6 = nn.Upsample(scale_factor=6, mode='bilinear', align_corners=True)
        else:
            self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)

        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
            elif scale == 5:
                return self.up5(x)
            elif scale == 6:
                return self.up6(x)
        else:
            return self.up(x)


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
            self.up5 = _UpsampleBlock(n_channels, scale=5, group=group)
            self.up6 = _UpsampleBlock(n_channels, scale=6, group=group)
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
            elif scale == 5:
                return self.up5(x)
            elif scale == 6:
                return self.up6(x)
        else:
            return self.up(x)


class _UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, group=1):
        super(_UpsampleBlock, self).__init__()

        modules = self._get_group_modules(n_channels, scale,
                                          group) if group > 0 else self._get_depthwise_separable_modules(n_channels,
                                                                                                         scale)

        self.body = nn.Sequential(*modules)
        init_weights(self.modules)

    def _get_depthwise_separable_modules(self, n_channels, scale):
        modules = []
        if scale == 2 or scale == 4:
            for _ in range(int(math.log(scale, 2))):
                modules += [nn.Conv2d(n_channels, n_channels, 3, 1, 1, groups=n_channels), nn.ReLU(inplace=True)]
                modules += [nn.Conv2d(n_channels, 9 * n_channels, 1, 1, 0, groups=1), nn.ReLU(inplace=True)]
                modules += [nn.PixelShuffle(3)]
                modules += [nn.Upsample(scale_factor=2.0 / 3.0, mode='bilinear', align_corners=True),
                            nn.ReLU(inplace=True)]
        elif scale == 3:
            modules += [nn.Conv2d(n_channels, n_channels, 3, 1, 1, groups=n_channels), nn.ReLU(inplace=True)]
            modules += [nn.Conv2d(n_channels, 9 * n_channels, 1, 1, 0, groups=1), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]
        elif scale == 5:
            modules += [nn.Conv2d(n_channels, n_channels, 3, 1, 1, groups=n_channels), nn.ReLU(inplace=True)]
            modules += [nn.Conv2d(n_channels, 9 * n_channels, 1, 1, 0, groups=1), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]
            modules += [nn.Upsample(scale_factor=2.0 / 3.0, mode='bilinear', align_corners=True), nn.ReLU(inplace=True)]
            modules += [nn.Conv2d(n_channels, n_channels, 3, 1, 1, groups=n_channels), nn.ReLU(inplace=True)]
            modules += [nn.Conv2d(n_channels, 9 * n_channels, 1, 1, 0, groups=1), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]
            modules += [nn.Upsample(scale_factor=5.0 / 6.0, mode='bilinear', align_corners=True), nn.ReLU(inplace=True)]
        elif scale == 6:
            modules += [nn.Conv2d(n_channels, n_channels, 3, 1, 1, groups=n_channels), nn.ReLU(inplace=True)]
            modules += [nn.Conv2d(n_channels, 9 * n_channels, 1, 1, 0, groups=1), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]
            modules += [nn.Upsample(scale_factor=2.0 / 3.0, mode='bilinear', align_corners=True), nn.ReLU(inplace=True)]
            modules += [nn.Conv2d(n_channels, n_channels, 3, 1, 1, groups=n_channels), nn.ReLU(inplace=True)]
            modules += [nn.Conv2d(n_channels, 9 * n_channels, 1, 1, 0, groups=1), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]
        return modules

    def _get_group_modules(self, n_channels, scale, group=1):
        modules = []
        if scale == 2 or scale == 4:
            for _ in range(int(math.log(scale, 2))):
                modules += [nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
                modules += [nn.PixelShuffle(3)]
                modules += [nn.Upsample(scale_factor=2.0 / 3.0, mode='bilinear', align_corners=True),
                            nn.ReLU(inplace=True)]
        elif scale == 3:
            modules += [nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]
        elif scale == 5:
            modules += [nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]
            modules += [nn.Upsample(scale_factor=2.0 / 3.0, mode='bilinear', align_corners=True), nn.ReLU(inplace=True)]
            modules += [nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]
            modules += [nn.Upsample(scale_factor=5.0 / 6.0, mode='bilinear', align_corners=True), nn.ReLU(inplace=True)]
        elif scale == 6:
            modules += [nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]
            modules += [nn.Upsample(scale_factor=2.0 / 3.0, mode='bilinear', align_corners=True), nn.ReLU(inplace=True)]
            modules += [nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]
        return modules

    def forward(self, x):
        out = self.body(x)
        return out
