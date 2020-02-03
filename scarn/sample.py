import os
import json
import time
import importlib
import argparse
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.autograd import Variable
from dataset import TestDataset
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="carn_m")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoint/carn_m/carn_m.pth")
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--sample_dir", type=str, default="./sample")
    # parser.add_argument("--test_data_dir", type=str, default="dataset/DIV2K/DIV2K_valid")
    parser.add_argument("--test_data_dir", type=str, default="./sample/test")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--scale", type=int, default=0)
    parser.add_argument("--shave", type=int, default=20)

    return parser.parse_args()


def normalize(x, device):
    x = x.to(device)
    return ((x[:, 0, :, :] - x[:, 1, :, :]) / (x[:, 2, :, :] + 0.06)).unsqueeze(1)


def save_image(tensor, filename):
    tensor = tensor.cpu()
    print(tensor.shape)
    # ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    # im = Image.fromarray(ndarr)
    ndarr = tensor[0].mul(255).clamp(0, 255).byte().numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)


def sample(net, device, dataset, cfg, scale):
    for step, (hr, lr, name) in enumerate(dataset):
        if "DIV2K" in dataset.name:
            t1 = time.time()
            h, w = lr.size()[1:]
            h_half, w_half = int(h/2), int(w/2)
            h_chop, w_chop = h_half + cfg.shave, w_half + cfg.shave

            lr_patch = torch.Tensor(4, 1, h_chop, w_chop)
            lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
            lr_patch[1].copy_(lr[:, 0:h_chop, w-w_chop:w])
            lr_patch[2].copy_(lr[:, h-h_chop:h, 0:w_chop])
            lr_patch[3].copy_(lr[:, h-h_chop:h, w-w_chop:w])
            lr_patch = normalize(lr_patch, device)
            
            sr = net(lr_patch, scale).detach()
            
            h, h_half, h_chop = h*scale, h_half*scale, h_chop*scale
            w, w_half, w_chop = w*scale, w_half*scale, w_chop*scale

            result = torch.Tensor(1, h, w).to(device)
            result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
            result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop-w+w_half:w_chop])
            result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop-h+h_half:h_chop, 0:w_half])
            result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop-h+h_half:h_chop, w_chop-w+w_half:w_chop])

            # 将归一化后的像素还原
            m = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
            sr = result * (m(lr[2]) + 0.06) + m(lr[1])
            t2 = time.time()
        else:
            t1 = time.time()
            if lr is not None:
                lr = lr.unsqueeze(0)
                lr = normalize(lr, device)

                result = net(lr, scale).detach().squeeze(0)
                # 将归一化后的像素还原
                m = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
                sr = result * (m(lr[2]) + 0.06) + m(lr[1])
                lr = lr.squeeze(0)
            t2 = time.time()
        
        model_name = cfg.ckpt_path.split(".")[0].split("/")[-1]
        sr_dir = os.path.join(cfg.sample_dir,
                              model_name, 
                              cfg.test_data_dir.split("/")[-1],
                              "x{}".format(scale),
                              "SR")
        hr_dir = os.path.join(cfg.sample_dir,
                              model_name, 
                              cfg.test_data_dir.split("/")[-1],
                              "x{}".format(scale),
                              "HR")
        
        os.makedirs(sr_dir, exist_ok=True)
        os.makedirs(hr_dir, exist_ok=True)

        sr_im_path = os.path.join(sr_dir, "{}".format(name.replace("HR", "SR")))
        hr_im_path = os.path.join(hr_dir, "{}".format(name))

        if sr is not None:
            save_image(sr, sr_im_path)
        if hr is not None:
            save_image(hr, hr_im_path)
        print("Saved {} ({}x{} -> {}x{}, {:.3f}s)"
            .format(sr_im_path, lr.shape[1], lr.shape[2], sr.shape[1], sr.shape[2], t2-t1))


def main(cfg):
    module = importlib.import_module("model.{}".format(cfg.model))
    if cfg.scale > 0:
        net = module.Net(scale=cfg.scale, group=cfg.group, channels=cfg.channels)
    else:
        net = module.Net(multi_scale=True, group=cfg.group, channels=cfg.channels)
    print(json.dumps(vars(cfg), indent=4, sort_keys=True))

    state_dict = torch.load(cfg.ckpt_path, map_location=None if torch.cuda.is_available() else torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        # name = k[7:] # remove "module."
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    
    device = torch.device("cpu")
    net = net.to(device)
    if cfg.scale > 0:
        dataset = TestDataset(cfg.test_data_dir, cfg.scale)
        sample(net, device, dataset, cfg, cfg.scale)
    else:
        for scale in [2, 4]:
            dataset = TestDataset(cfg.test_data_dir, scale)
            sample(net, device, dataset, cfg, scale)
 

if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
