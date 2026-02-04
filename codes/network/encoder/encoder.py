import torch
from torch import nn
from network.encoder import resnet_1d
import numpy as np
def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, bias=False, groups=groups)
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, groups=groups)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, groups=groups)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class mix_layer(nn.Module):
    def __init__(self):
        super(mix_layer, self).__init__()
        self.mlp = nn.Linear(128, 1152)
        self.conv = nn.Conv1d(129, 128, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.res_scale = nn.Parameter(torch.ones(1))

    def forward(self, x, c):
        residual = x
        c = self.mlp(c)
        x_ = torch.cat([x, c], dim=1)
        x_ = self.conv(x_)
        x_all = x_ + residual*0
        return x_all

class Encoder(nn.Module):
    def __init__(self, backbone='resnet34', in_channel=1, use_first_pool=True, lead_num=1, init_channels=64):
        """
        :param backbone Backbone network.
        :param num_layers number of resnet layers to use, 1-5
        :param upsample_interp Interpolation to use for upscaling latent code
        :param in_channel encoder in channel
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        """
        super().__init__()

        # define backbone
        model = getattr(resnet_1d, backbone)(in_channel=in_channel, lead_num=lead_num, init_channels=init_channels)
        self.conv1 = model.conv1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1

        self.use_first_pool = use_first_pool

    def forward(self, x):
        '''
        :param x:   [B, num_lead, 512]
        :return:    [B, out_channels, 128]
        '''
        # ori_size = x.size(-1)

        x = self.conv1(x)
        x = self.relu(x)
        if self.use_first_pool:
            x = self.maxpool(x)
        x = self.layer1(x)
        return x

class layer(nn.Module):
    def __init__(self, backbone='resnet34', in_channel=1, use_first_pool=True, lead_num=1, init_channels=64):
        """
        :param backbone Backbone network.
        :param num_layers number of resnet layers to use, 1-5
        :param upsample_interp Interpolation to use for upscaling latent code
        :param in_channel encoder in channel
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        """
        super().__init__()
        # define backbone
        model = getattr(resnet_1d, backbone)(in_channel=in_channel, lead_num=lead_num, init_channels=init_channels)
        self.layer1 = model.layer1

    def forward(self, x):
        '''
        :param x:   [B, num_lead, 512]
        :return:    [B, out_channels, 128]
        '''
        x = self.layer1(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SELayer, self).__init__()
        self.channels = channels
        self.reduction = reduction
        # Squeeze: 全局平均池化 (1D-GAP)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # [B, C, L] -> [B, C, 1]
        # Excitation: 全连接层学习通道关系
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()  # 输出通道权重 [0,1]
        )
    def forward(self, x):
        # Squeeze
        b, c, l = x.size()
        y = self.avg_pool(x).view(b, c)  # [B, C, 1] -> [B, C]
        # Excitation
        y = self.fc(y).view(b, c, 1)  # [B, C] -> [B, C, 1]
        # Scale: 通道加权
        return x * y.expand_as(x)

class Encoder(nn.Module):
    def __init__(self, backbone='resnet34', in_channel=1, use_first_pool=True, lead_num=1, init_channels=64):
        """
        :param backbone Backbone network.
        :param num_layers number of resnet layers to use, 1-5
        :param upsample_interp Interpolation to use for upscaling latent code
        :param in_channel encoder in channel
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        """
        super().__init__()
        # define backbone
        model = getattr(resnet_1d, backbone)(in_channel=in_channel, lead_num=lead_num, init_channels=init_channels)
        self.lead_num = lead_num
        self.conv1 = model.conv1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.use_first_pool = use_first_pool
        self.mlp = nn.Linear(128, 1152)
        self.conv = nn.Conv1d(129, 128, kernel_size=1, stride=1, bias=False)

    def forward(self, x, c):
        '''
        :param x:   [B, num_lead, 512]
        :return:    [B, out_channels, 128]
        '''
        # ori_size = x.size(-1)
        x = self.conv1(x)
        x = self.relu(x)
        if self.use_first_pool:
            x = self.maxpool(x)

        c = self.mlp(c)
        x_list = torch.cat([x, c], dim=1)

        x_list = self.conv(x_list)
        out = self.layer1(x_list)
        return out


if __name__ == '__main__':
    x = torch.randn(8, 3, 4608)
    c = torch.randn(8,1,128)
    rois = []
    for i in range(8):
        roi, _ = torch.randint(0, 512, [10]).sort()
        roi = roi.view(5, 2)
        rois.append(roi)
    rois = torch.stack(rois)
    lead_num = 3
    net = Encoder(backbone='resnet34', in_channel=lead_num, use_first_pool=True, lead_num=lead_num,
                                 init_channels=128)
    y1 = net(x)
    print(y1.shape)
