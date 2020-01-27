import os
import glob
import h5py
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


# 随机从一张图中扣出一块
def random_crop(hr, lr, size, scale):
    h, w = lr.shape[:-1]
    x = random.randint(0, w-size)
    y = random.randint(0, h-size)

    hsize = size*scale
    hx, hy = x*scale, y*scale

    crop_lr = lr[y:y+size, x:x+size].copy()
    crop_hr = hr[hy:hy+hsize, hx:hx+hsize].copy()

    return crop_hr, crop_lr


def random_flip_and_rotate(im1, im2):
    # 上下左右随机翻转
    if random.random() < 0.5:
        im1 = np.flipud(im1)
        im2 = np.flipud(im2)

    if random.random() < 0.5:
        im1 = np.fliplr(im1)
        im2 = np.fliplr(im2)

    # 随机旋转
    angle = random.choice([0, 1, 2, 3])
    im1 = np.rot90(im1, angle)
    im2 = np.rot90(im2, angle)

    # have to copy before be called by transform function
    return im1.copy(), im2.copy()


class TrainDataset(data.Dataset):
    def __init__(self, path, size, scale):
        super(TrainDataset, self).__init__()

        self.size = size
        h5f = h5py.File(path, "r")
        
        self.hr = [v[:][..., np.newaxis] for v in h5f["HR"].values()]
        # perform multi-scale training
        if scale == 0:
            self.scale = [2, 4]
            self.lr = [[v[:][..., np.newaxis] for v in h5f["X{}".format(i)].values()] for i in self.scale]
        else:
            self.scale = [scale]
            self.lr = [[v[:][..., np.newaxis] for v in h5f["X{}".format(scale)].values()]]
        
        h5f.close()

        # 用来将图片转换成tensor
        self.transform = transforms.Compose([
            transforms.ToTensor()  # range [0, 255] -> [0.0,1.0]
        ])

    def __getitem__(self, index):
        size = self.size

        # 每次选出所有图，然后切分出一块
        item = [(self.hr[index], self.lr[i][index]) for i, _ in enumerate(self.lr)]
        item = [random_crop(hr, lr, size, self.scale[i]) for i, (hr, lr) in enumerate(item)]
        item = [random_flip_and_rotate(hr, lr) for hr, lr in item]
        
        return [(self.transform(hr), self.transform(lr)) for hr, lr in item]

    def __len__(self):
        return len(self.hr)
        

class TestDataset(data.Dataset):
    def __init__(self, dirname, scale):
        super(TestDataset, self).__init__()

        self.name = dirname.split("/")[-1]
        self.scale = scale
        
        if "DIV" in self.name:
            self.hr = glob.glob(os.path.join("{}_HR".format(dirname), "*.png"))
            self.lr = glob.glob(os.path.join("{}_LR_bicubic".format(dirname), 
                                             "X{}/*.png".format(scale)))
        else:
            all_files = glob.glob(os.path.join(dirname, "x{}/*.png".format(scale)))
            self.hr = [name for name in all_files if "HR" in name]
            self.lr = [name for name in all_files if "LR" in name]

        self.hr.sort()
        self.lr.sort()

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        hr = Image.open(self.hr[index]).convert('L')
        lr = Image.open(self.lr[index]).convert('L') 

        # 读出不经过缩放的所有像素
        hr = hr.convert("1")
        lr = lr.convert("1")
        filename = self.hr[index].split("/")[-1]

        return self.transform(hr), self.transform(lr), filename

    def __len__(self):
        return len(self.hr)
