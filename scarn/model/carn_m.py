from .ops import *


class Block(nn.Module):
    def __init__(self, channels=64, group=1):
        super(Block, self).__init__()

        self.b1 = EResidualBlock(channels, channels, group=group)
        self.c1 = BasicBlock(channels * 2, channels, 1, 1, 0)
        self.c2 = BasicBlock(channels * 3, channels, 1, 1, 0)
        self.c3 = BasicBlock(channels * 4, channels, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b1(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b1(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3


class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()

        scale = kwargs.get("scale")
        multi_scale = kwargs.get("multi_scale")
        group = kwargs.get("group", 1)
        channels = 64

        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)

        # 和tensorflow不同，卷积输入格式是[B,C,H,W]，对比TF是[B,H,W,C]
        self.entry = nn.Conv2d(1, channels, 3, 1, 1)

        self.b1 = Block(channels, group=group)
        self.b2 = Block(channels, group=group)
        self.b3 = Block(channels, group=group)
        self.c1 = BasicBlock(channels * 2, channels, 1, 1, 0)
        self.c2 = BasicBlock(channels * 3, channels, 1, 1, 0)
        self.c3 = BasicBlock(channels * 4, channels, 1, 1, 0)

        self.upsample = UpsampleBlock(channels, scale=scale, multi_scale=multi_scale, group=group)
        self.exit = nn.Conv2d(channels, 3, 3, 1, 1)

    def forward(self, x, scale):
        # 输入的像素减去通道经验均值
        x = self.sub_mean(x)
        # 用3*3的卷积核将原来的3通道变成64通道
        x = self.entry(x)
        c0 = o0 = x

        # 将64个通道经过各种交叉的卷积
        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        # 放大长宽
        out = self.upsample(o3, scale=scale)

        # 将64个通道还原回3个
        out = self.exit(out)
        # 颜色加上均值
        out = self.add_mean(out)

        return out
