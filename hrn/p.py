import os
import json
import time
import importlib
import argparse
import numpy as np
from collections import OrderedDict
import torch
from torch.autograd import Variable
from dataset import TestDataset
from PIL import Image

from torch.utils.data import DataLoader
import skimage.measure as measure


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoint")
    parser.add_argument("--group", type=int, default=1)
    parser.add_argument("--test_data_dir", type=str, default="dataset/Urban100")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--shave", type=int, default=20)

    return parser.parse_args()


def psnr(im1, im2):
    def im2double(im):
        min_val, max_val = 0, 255
        out = (im.astype(np.float64)-min_val) / (max_val-min_val)
        return out
        
    im1 = im2double(im1)
    im2 = im2double(im2)
    psnr = measure.compare_psnr(im1, im2)
    return psnr

def sample(net, device, dataset, cfg):
    scale = cfg.scale
    mean_psnr = 0
    for step, (hr, lr, name) in enumerate(dataset):
        if "" in dataset.name:
            t1 = time.time()
            h, w = lr.size()[1:]
            h_half, w_half = int(h/2), int(w/2)
            h_chop, w_chop = h_half + cfg.shave, w_half + cfg.shave

            lr_patch = torch.FloatTensor(4, 3, h_chop, w_chop)
            lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
            lr_patch[1].copy_(lr[:, 0:h_chop, w-w_chop:w])
            lr_patch[2].copy_(lr[:, h-h_chop:h, 0:w_chop])
            lr_patch[3].copy_(lr[:, h-h_chop:h, w-w_chop:w])
            lr_patch = lr_patch.to(device)
            
            sr = net(lr_patch, cfg.scale).detach()
            
            h, h_half, h_chop = h*scale, h_half*scale, h_chop*scale
            w, w_half, w_chop = w*scale, w_half*scale, w_chop*scale

            result = torch.FloatTensor(3, h, w).to(device)
            result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
            result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop-w+w_half:w_chop])
            result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop-h+h_half:h_chop, 0:w_half])
            result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop-h+h_half:h_chop, w_chop-w+w_half:w_chop])
            sr = result
            t2 = time.time()
            
            hr = hr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            sr = sr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            
            bnd = 4
            im1 = hr[bnd:-bnd, bnd:-bnd]
            im2 = sr[bnd:-bnd, bnd:-bnd]
            mean_psnr += psnr(im1, im2) / len(dataset)
            
    return mean_psnr
        
        
def main(cfg):
    module = importlib.import_module("model.{}".format(cfg.model))
    net = module.Net(multi_scale=True, group=cfg.group)
    #net = module.Net(scale=cfg.scale, group=cfg.group)
    print(json.dumps(vars(cfg), indent=4, sort_keys=True))
    ckpt_path = cfg.ckpt_dir
    pathDir = os.listdir(ckpt_path)
    pathDir.sort()
    l = len(pathDir)
    for path in pathDir:
        if "" in path:
            state_dict = torch.load(os.path.join(ckpt_path, path))
            new_state_dict = OrderedDict()
    
            for k, v in state_dict.items():
                name = k
            # name = k[7:] # remove "module."
                new_state_dict[name] = v

            net.load_state_dict(new_state_dict)
    
            device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
            net = net.to(device)
    
            dataset = TestDataset(cfg.test_data_dir, cfg.scale)
            psnr=sample(net, device, dataset, cfg)
            print("path_name:",path," scale:",cfg.scale, "  psnr:",psnr )
        
	

with torch.no_grad():
    if __name__ == "__main__":
        cfg = parse_args()
        main(cfg)

