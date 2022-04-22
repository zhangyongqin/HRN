import torch
import torch.nn as nn
import model.ops as ops
import torch_dct as dct

class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        
        scale = kwargs.get("scale")
        multi_scale = kwargs.get("multi_scale")
        group = kwargs.get("group", 1)

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.entry = nn.Conv2d(3, 64, 3, 1, 1)

        self.b1 = nn.Sequential(
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(inplace=True)
        )
        self.f_b1 = nn.Sequential(
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(inplace=True)
        )
        self.c1 = nn.Conv2d(64*2,64,1,1,0)
        self.fc1 = nn.Conv2d(64*2,64,1,1,0)
        self.b2 = nn.Sequential(
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(inplace=True)
        )
        self.f_b2 = nn.Sequential(
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(inplace=True)
        )
        self.c2 = nn.Conv2d(64*2,64,1,1,0)
        self.fc2 = nn.Conv2d(64*2,64,1,1,0)
        self.b3 = nn.Sequential(
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(inplace=True)
        )
        self.f_b3 = nn.Sequential(
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(inplace=True)
        )

        self.conv1x1 = nn.Conv2d(64*2,64,1,1,0)
        
        self.upsample = ops.UpsampleBlock(64, scale=scale, 
                                          multi_scale=multi_scale,
                                          group=group)
        self.exit = nn.Conv2d(64, 3, 3, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                
    def forward(self, x, scale):
        x = self.sub_mean(x)
        x = self.entry(x)
        f_x = dct.dct(x)
        
        b1 = self.b1(x)
        f_b1 = self.f_b1(f_x)
        o1 = torch.cat([b1,dct.idct(f_b1)],dim=1)
        f_o1 = torch.cat([dct.dct(b1),f_b1],dim=1)
        x1 = self.c1(o1)
        f_x1 = self.fc1(f_o1)

        b2 = self.b2(x1)
        f_b2 = self.f_b2(f_x1)
        o2 = torch.cat([b2,dct.idct(f_b2)],dim=1)
        f_o2 = torch.cat([dct.dct(b2),f_b2],dim=1)
        x2 = self.c2(o2)
        f_x2 = self.fc2(f_o2)

        b3 = self.b3(x2)
        f_b3 = self.f_b3(f_x2)
        out = torch.cat([b3,dct.idct(f_b3)],dim=1)
        out = self.conv1x1(out)
        
        out = self.upsample(out, scale=scale)

        out = self.exit(out)
        out = self.add_mean(out)

        return out
