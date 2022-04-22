import torch
import logging
from torchvision import models
import sys
f = open('a.log', 'a')
sys.stdout = f

sys.stderr = f
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from HRN import Net
model = Net(scale=4, group=1)
md = model.to(device)

summary(md,(3, 64, 64))

