from .ops import *


class Block(nn.Module):
    def __init__(self, channels=64, group=1):
        super(Block, self).__init__()

        self.b1 = ResidualBlock(channels, channels)
        self.b2 = ResidualBlock(channels, channels)
        self.b3 = ResidualBlock(channels, channels)
        self.c1 = BasicBlock(channels * 2, channels, 1, 1, 0)
        self.c2 = BasicBlock(channels * 3, channels, 1, 1, 0)
        self.c3 = BasicBlock(channels * 4, channels, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
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

        self.entry = nn.Conv2d(3, channels, 3, 1, 1)

        self.b1 = Block(channels, channels)
        self.b2 = Block(channels, channels)
        self.b3 = Block(channels, channels)
        self.c1 = BasicBlock(channels * 2, channels, 1, 1, 0)
        self.c2 = BasicBlock(channels * 3, channels, 1, 1, 0)
        self.c3 = BasicBlock(channels * 4, channels, 1, 1, 0)

        self.upsample = UpsampleBlock(channels, scale=scale,
                                          multi_scale=multi_scale,
                                          group=group)
        self.exit = nn.Conv2d(channels, 3, 3, 1, 1)

    def forward(self, x, scale):
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        out = self.upsample(o3, scale=scale)

        out = self.exit(out)
        out = self.add_mean(out)

        return out
