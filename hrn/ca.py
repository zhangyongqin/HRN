import torch
from torch import nn


class CA_Block(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CA_Block, self).__init__()

        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w)) 

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()
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
        print(s_h.shape)
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        print(s_w.shape)
        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out


if __name__ == '__main__':
    x = torch.randn(1, 16, 128, 64) 
    ca_model = CA_Block(channel=16, h=128, w=64)
    y = ca_model(x)
    print(y.shape)