import os
import random
import numpy as np
import scipy.misc as misc
import skimage.measure as measure
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from collections import OrderedDict
from dataset import TrainDataset, TestDataset

os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"

best_psnr = 0

class Solver():
    def __init__(self, model, cfg):
        if cfg.scale > 0:
            self.refiner = model(scale=cfg.scale, 
                                 group=cfg.group)
        else:
            self.refiner = model(multi_scale=True, 
                                 group=cfg.group)
        
        if cfg.loss_fn in ["MSE"]: 
            self.loss_fn = nn.MSELoss()
        elif cfg.loss_fn in ["L1"]: 
            self.loss_fn = nn.L1Loss()
        elif cfg.loss_fn in ["SmoothL1"]:
            self.loss_fn = nn.SmoothL1Loss()

        self.optim = optim.Adam(
            filter(lambda p: p.requires_grad, self.refiner.parameters()), 
            cfg.lr)
        
        self.train_data = TrainDataset(cfg.train_data_path, 
                                       scale=cfg.scale, 
                                       size=cfg.patch_size)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=cfg.batch_size,
                                       num_workers=1,
                                       shuffle=True, drop_last=True)
        
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.refiner = self.refiner.to(self.device)
        #checkpoint = torch.load('checkpoint/dcarn_best.pth')
        #self.refiner.load_state_dict(checkpoint)
        self.loss_fn = self.loss_fn
        #self.tv_loss = TVLoss()

        self.cfg = cfg
        self.step = 0
        
        if cfg.verbose:
            num_params = 0
            for param in self.refiner.parameters():
                num_params += param.nelement()
            print("# of params:", num_params)

        os.makedirs(cfg.ckpt_dir, exist_ok=True)

    def fit(self):
        global best_psnr
        cfg = self.cfg
        #refiner = nn.DataParallel(self.refiner, 
                                  #device_ids=[0,1])
        refiner = self.refiner
        
        
        #device_ids = [1, 2]
        #state_dict = torch.load(os.path.join("/team_stor1/jili/CARN-pytorch-master/checkpoint/hcan8x3/hcan8x3_best.pth"))
        #new_state_dict = OrderedDict()
        #for k, v in state_dict.items():
        #    name = k
        #    new_state_dict[name] = v
        #refiner = refiner.to(self.device)
        #refiner.load_state_dict(new_state_dict)
        torch.backends.cudnn.benchmark = True
        refiner = nn.DataParallel(refiner, device_ids=[0,1])
        learning_rate = cfg.lr
        loss_value = []
        while True:
            for inputs in self.train_loader:
                self.refiner.train()

                if cfg.scale > 0:
                    scale = cfg.scale
                    hr, lr = inputs[-1][0], inputs[-1][1]
                else:
                    # only use one of multi-scale data
                    # i know this is stupid but just temporary
                    scale = random.randint(2, 4)
                    hr, lr = inputs[scale-2][0], inputs[scale-2][1]
                
                hr = hr.to(self.device)
                lr = lr.to(self.device)
                
                sr = refiner(lr, scale)
                loss = self.loss_fn(sr, hr) 
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.refiner.parameters(), cfg.clip)
                self.optim.step()

                learning_rate = self.decay_learning_rate()
                for param_group in self.optim.param_groups:
                    param_group["lr"] = learning_rate
                
                self.step += 1
                
                if cfg.verbose and self.step % cfg.print_interval == 0:  
                    psnr = self.evaluate("dataset/DIV2K_valid.h5", scale=cfg.scale, size=cfg.patch_size, num_step=self.step)
                    best_psnr = psnr if psnr > best_psnr else best_psnr
                    print("step: %d scale: %d psnr: %f best_psnr: %f"%(self.step,scale,psnr,best_psnr))
                    #print(self.step)                                  
                    self.save(cfg.ckpt_dir, cfg.ckpt_name,self.step)
                        
            if self.step > cfg.max_steps: break

    
    def load(self, path):
        self.refiner.load_state_dict(torch.load(path))
        splited = path.split(".")[0].split("_")[-1]
        try:
            self.step = int(path.split(".")[0].split("_")[-1])
        except ValueError:
            self.step = 0
        print("Load pretrained {} model".format(path))

    def save(self, ckpt_dir, ckpt_name,step):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, str(step)))
        torch.save(self.refiner.state_dict(), save_path)

    def decay_learning_rate(self):
        lr = self.cfg.lr * (0.5 ** (self.step // self.cfg.decay))
        return lr
    def evaluate(self, test_data_dir,size , scale=2, num_step=0):
        cfg = self.cfg
        mean_psnr = 0
        self.refiner.eval()
        
        #test_data   = TestDataset(test_data_dir, scale=scale)
        test_data = TrainDataset(test_data_dir, 
                                 scale=scale, 
                                 size=size)
        test_loader = DataLoader(test_data,
                                 batch_size=1,
                                 num_workers=1,
                                 shuffle=False)

        for step, inputs in enumerate(test_loader):
            if cfg.scale > 0:
                scale = cfg.scale
                hr, lr = inputs[-1][0], inputs[-1][1]
            else:
                scale = random.randint(2, 4)
                hr, lr = inputs[scale-2][0], inputs[scale-2][1] 
            lr = lr.to(self.device)
            sr = self.refiner(lr,scale).detach()
            
            #h, w = lr.size()[2:]
            #h_half, w_half = int(h/2), int(w/2)
            #h_chop, w_chop = h_half + cfg.shave, w_half + cfg.shave

            # split large image to 4 patch to avoid OOM error
            #lr_patch = torch.FloatTensor(4, 3, h_chop, w_chop)
            #lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
            #lr_patch[1].copy_(lr[:, 0:h_chop, w-w_chop:w])
            #lr_patch[2].copy_(lr[:, h-h_chop:h, 0:w_chop])
            #lr_patch[3].copy_(lr[:, h-h_chop:h, w-w_chop:w])
            #lr_patch = lr_patch.to(self.device)
            
            # run refine process in here!
            #sr = self.refiner(lr_patch, scale).data
            
            #h, h_half, h_chop = h*scale, h_half*scale, h_chop*scale
            #w, w_half, w_chop = w*scale, w_half*scale, w_chop*scale
            
            # merge splited patch images
            #result = torch.FloatTensor(3, h, w).to(self.device)
            #result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
            #result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop-w+w_half:w_chop])
            #result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop-h+h_half:h_chop, 0:w_half])
            #result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop-h+h_half:h_chop, w_chop-w+w_half:w_chop])
            #sr = result
            hr = hr.squeeze(0)
            sr = sr.squeeze(0)
            hr = hr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    
            sr = sr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            
            # evaluate PSNR
            # this evaluation is different to MATLAB version
            # we evaluate PSNR in RGB channel not Y in YCbCR  
            bnd = scale
            im1 = hr[bnd:-bnd, bnd:-bnd]
            im2 = sr[bnd:-bnd, bnd:-bnd]
            mean_psnr += psnr(im1, im2) / len(test_data)

        return mean_psnr
def psnr(im1, im2):
    def im2double(im):
        min_val, max_val = 0, 255
        out = (im.astype(np.float64)-min_val) / (max_val-min_val)
        return out
        
    im1 = im2double(im1)
    im2 = im2double(im2)
    psnr = measure.compare_psnr(im1, im2)
    return psnr    


