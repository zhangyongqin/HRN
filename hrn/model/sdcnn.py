import torch
import torch.nn as nn
import model.ops as ops

class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        
        scale = kwargs.get("scale")
        multi_scale = kwargs.get("multi_scale")
        group = kwargs.get("group", 1)

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.entry = nn.Conv2d(3, 64, 3, 1, 1)

        body = nn.ModuleList()
        for i in range(3):
            op = nn.Sequential(
                nn.Conv2d(64,64,3,1,1),
                nn.ReLU(inplace=True)
            )
            body.append(op)
        self.body = nn.Sequential(*body)
        
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
        
        out = self.body(x)

        out = self.upsample(out, scale=scale)

        out = self.exit(out)
        out = self.add_mean(out)

        return out
