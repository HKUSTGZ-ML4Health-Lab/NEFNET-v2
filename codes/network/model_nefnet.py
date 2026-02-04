from network.encoder.encoder import Encoder
from network.utils.theta_encoder import ThetaEncoder

import torch
import torch.nn as nn
import random


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, groups=groups)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, groups=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, groups=groups)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, groups=groups)
        self.residual_conv = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, groups=groups)
        self.stride = stride
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        if out.shape[1] != residual.shape[1]:  # if num of channel is not same
            residual = self.residual_conv(residual)

        out += residual
        out = self.relu(out)

        return out


class Model_nefnet(nn.Module):
    """
        Nef-Net implement
    """
    def __init__(self, theta_encoder_len=1, lead_num=3):
        super(Model_nefnet, self).__init__()

        self.lead_num = lead_num
        self.W_encoder = Encoder(backbone='resnet34', in_channel=lead_num, use_first_pool=True, lead_num=lead_num,
                                 init_channels=128)
        self.theta_encoder = ThetaEncoder(encoder_len=theta_encoder_len)
        self.mlp1 = nn.Linear((2*theta_encoder_len+1)*4, 128)
        self.mlp2 = nn.Linear((2*theta_encoder_len+1)*4, 256 * 1)

        self.w_feature_extractor = nn.Sequential(
            nn.Conv1d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.w_conv = nn.Sequential(
            BasicBlock(128 * self.lead_num, 128 * self.lead_num, 1, groups=self.lead_num)
        )

        self.z1_conv = nn.Sequential(
            BasicBlock(64 * self.lead_num, 128 * self.lead_num, 1, groups=self.lead_num)
        )
        self.z2_conv1 = nn.Sequential(
            BasicBlock(64 * self.lead_num, 128 * self.lead_num, 1, groups=self.lead_num)
        )
        self.z2_conv2 = nn.Sequential(
            BasicBlock(128 * 7 * self.lead_num, 128 * 7 * self.lead_num, 1, groups=self.lead_num*7),
            nn.ConvTranspose1d(128 * 7 * self.lead_num, 128 * 7 * self.lead_num // 2,
                               kernel_size=2, stride=2, groups=self.lead_num*7),
            BasicBlock(128 * 7 * self.lead_num // 2, 128 * 7 * self.lead_num, 1, groups=self.lead_num*7),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            DoubleConv(256 * 1, 128),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            DoubleConv(128, 64),
            nn.Conv1d(64, 1, 3, padding=1)
        )
        self.alpha = nn.Parameter(torch.zeros(3, 2))

    def forward(self, x, input_thetas, query_theta, rest_theta=None, phase='train'):
        '''
        :param x:   [B, lead_num, 512]
        :param input_thetas:  [B, lead_num, 2]
        :param query_theta:   [B, 2]
        :return:
        '''
        w = self.W_encoder(x)  #[B, 128 * lead_num, 128]
        if phase =='finetune':
            input_thetas = input_thetas + self.alpha.unsqueeze(0)
        input_thetas = self.theta_encoder(input_thetas)  # [B, lead_num, 12]

        w_one_lead_list = torch.chunk(w, self.lead_num, dim=1)
        encoded_theta = self.mlp1(input_thetas)
        encoded_w_list = [encoded_theta[:, i][:, :, None] * w_one_lead_list[i] for i in
                          range(self.lead_num)]  #list lead_num * [B, 128, 128]
        encoded_w_list = self.w_conv(torch.cat(encoded_w_list, dim=1)) # [B, 128*3, 128]
        encoded_w_list = torch.chunk(encoded_w_list, self.lead_num, dim=1)

        latent_z1_z2_list = [torch.chunk(x, 2, dim=1) for x in encoded_w_list]
        z1_list = [x[0] for x in latent_z1_z2_list]  # list lead_num * [B, 64, 128]
        z1 = torch.cat(z1_list, dim=1)  # [B, 64 * lead_num, 128]
        z1 = self.z1_conv(z1)   # [B, 128 * lead_num, 128]
        # get mean of z1 z2
        z1_list = torch.chunk(z1, self.lead_num, dim=1)
        z1_mean = torch.mean(torch.stack(z1_list, dim=0), dim=0)  # [B, 128, 128]
        latent_all = torch.cat([z1_mean, z1_mean], dim=1)  # [B, 256 * 1, 128]

        # get pred, patient shuffle, lead shuffle
        query_theta = self.theta_encoder(query_theta)  # [B, 12]
        query_theta = self.mlp2(query_theta.view(query_theta.size(0), -1))  # [B, 256]

        query_w = query_theta[:, :, None] * latent_all  # [B, 256, 128]
        out = self.decoder(query_w)  # [B, 1, 512]
        out = torch.sigmoid(out / 3)
        return out


