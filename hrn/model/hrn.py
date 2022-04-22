import torch
import torch.nn as nn
import torch_dct as dct
import math
import torch.nn.init as init
import torch.nn.functional as F


def init_weights(modules):
    pass
    
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()

        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign

        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data   = torch.Tensor([r, g, b])

        # Freeze the mean shift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x
        
class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ELU(inplace=True)
        )

        init_weights(self.modules)
        
        
    def forward(self, x):
        out = self.body(x)
        return out
        
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
        

class CA_Block(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CA_Block, self).__init__()

        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w)) 

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU(True)
        self.bn = nn.BatchNorm2d(channel//reduction)

        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):

        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2) 
        x_w = self.avg_pool_y(x)  

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2))) 
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out

class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction, patch_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        #modules_body.append(CALayer(n_feat, reduction))
        modules_body.append(CA_Block(n_feat, 48, 48, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res
        
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, patch_size, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, patch_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
        
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class Net(nn.Module):
    def __init__(self, conv=default_conv, **kwargs):
        super(Net, self).__init__()
        
        n_resgroups = 10
        n_resblocks = 20
        n_feats = 64
        kernel_size = 3
        reduction = 16 
        scale = kwargs.get("scale")
        act = nn.ReLU(True)
        patch_size = 48

        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        modules_head = [conv(3, n_feats, kernel_size)]
        
        # define body module
        self.b1 = ResidualGroup(conv, n_feats, kernel_size, reduction,patch_size, act=act, res_scale=1, n_resblocks=n_resblocks)
        self.b2 = ResidualGroup(conv, n_feats, kernel_size, reduction, patch_size, act=act, res_scale=1, n_resblocks=n_resblocks)
        self.b3 = ResidualGroup(conv, n_feats, kernel_size, reduction, patch_size, act=act, res_scale=1, n_resblocks=n_resblocks)
        self.b4 = ResidualGroup(conv, n_feats, kernel_size, reduction, patch_size, act=act, res_scale=1, n_resblocks=n_resblocks)
        self.b5 = ResidualGroup(conv, n_feats, kernel_size, reduction,patch_size, act=act, res_scale=1, n_resblocks=n_resblocks)
        self.b6 = ResidualGroup(conv, n_feats, kernel_size, reduction,patch_size, act=act, res_scale=1, n_resblocks=n_resblocks)
        self.b7 = ResidualGroup(conv, n_feats, kernel_size, reduction,patch_size, act=act, res_scale=1, n_resblocks=n_resblocks)
        self.b8 = ResidualGroup(conv, n_feats, kernel_size, reduction,patch_size, act=act, res_scale=1, n_resblocks=n_resblocks)

        self.fb1 = ResidualGroup(conv, n_feats, kernel_size, reduction,patch_size, act=act, res_scale=1, n_resblocks=n_resblocks)
        self.fb2 = ResidualGroup(conv, n_feats, kernel_size, reduction,patch_size, act=act, res_scale=1, n_resblocks=n_resblocks)
        self.fb3 = ResidualGroup(conv, n_feats, kernel_size, reduction,patch_size, act=act, res_scale=1, n_resblocks=n_resblocks)
        self.fb4 = ResidualGroup(conv, n_feats, kernel_size, reduction,patch_size, act=act, res_scale=1, n_resblocks=n_resblocks)
        
        self.c1 = BasicBlock(64 * 2, 64, 1, 1, 0)
        self.c2 = BasicBlock(64 * 2, 64, 1, 1, 0)
        self.c3 = BasicBlock(64 * 2, 64, 1, 1, 0)
        self.c4 = BasicBlock(64 * 2, 64, 1, 1, 0)
        self.c5 = BasicBlock(64 * 2, 64, 1, 1, 0)
        self.c6 = BasicBlock(64 * 2, 64, 1, 1, 0)
        self.c7 = BasicBlock(64 * 2, 64, 1, 1, 0)
        self.c8 = BasicBlock(64 * 2, 64, 1, 1, 0)
        
        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x,scale=4):
        x = self.sub_mean(x)
        x = self.head(x)
        fx = dct.dct(x)
        
        b1 = self.b1(x)
        o1 = torch.cat([b1, x], dim=1)
        c1 = self.c1(o1)
        
        fb1 = self.fb1(fx)
        fc1 = dct.idct(fb1)
        
        b2 = self.b2(c1)
        o2 = torch.cat([b2, fc1], dim=1)
        c2 = self.c2(o2)
        
        b3 = self.b3(c2)
        o3 = torch.cat([b3, c2], dim=1)
        c3 = self.c1(o3)
        
        fb2 = self.fb2(fb1)
        fc2 = dct.idct(fb2)
        
        b4 = self.b2(c3)
        o4 = torch.cat([b4, fc2], dim=1)
        c4 = self.c4(o4)
        
        b5 = self.b5(c4)
        o5 = torch.cat([b5, c4], dim=1)
        c5 = self.c5(o5)
        
        fb3 = self.fb3(fb2)
        fc3 = dct.idct(fb3)
        
        b6 = self.b6(c5)
        o6 = torch.cat([b6, fc3], dim=1)
        c6 = self.c6(o6)
        
        b7 = self.b7(c6)
        o7 = torch.cat([b7, c6], dim=1)
        c7 = self.c7(o7)

        fb4 = self.fb4(fb3)
        fc4 = dct.idct(fb4)

        b8 = self.b8(c7)
        o8 = torch.cat([b8, fc4], dim=1)
        c8 = self.c8(o8)
        
        out  = x + c8
        
        out = self.tail(out)
        out = self.add_mean(out)

        return out
        
if __name__ == '__main__':
    net = Net(scale=4)
    print(net)
    
    data = torch.rand(1,3,32,32)
    from thop import profile
    from thop import clever_format
    a,b = profile(net,inputs=(data,8))
    macs, params = clever_format([a, b], "%.3f")
    print(macs,params)
    from ptflops import get_model_complexity_info
    model_name = 'HRN'
    flops, params = get_model_complexity_info(net, (3,32,32),as_strings=True,print_per_layer_stat=True)

    print("%s |%s |%s" % (model_name,flops,params))
        
                
     
