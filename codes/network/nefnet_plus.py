import torch
import torch.nn as nn
from network.encoder.encoder import Encoder,layer
import copy
from network.utils.theta_encoder import ThetaEncoder
import torch.nn.functional as F
import math
import os
from config import cfg
import numpy as np
from network.encoder import resnet_1d
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_


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
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class StableDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, L):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(in_channels, out_channels, 3, padding=1)),
            nn.LayerNorm(L),  # 替代BN，对时序更友好
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(out_channels, out_channels, 3, padding=1)),
            nn.LayerNorm(L),
            nn.GELU()
        )
        self.res_scale = nn.Parameter(torch.ones(1))  # 可学习残差缩放
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, groups=1)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        if x2.shape[1] != x.shape[1]:
            return self.residual_conv(x) + self.res_scale * x2
        else:
            return x + self.res_scale * x2
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

def attention_block(q,k):
    """
    q: query_theta[B,1,128], k: input_thetas[B,lead_num,128], v:features[B,lead_num,L]
    """
    scores = torch.matmul(q,k.transpose(-2,-1))
    d_k = k.size(-1)
    scaled_scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    attention_weights = F.softmax(scaled_scores,dim=-1)
    return attention_weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1152):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

class WeightLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))  # 可训练参数
        self.activation = nn.Sigmoid()                 # 激活函数

    def forward(self, x1, x2):
        alpha = self.activation(self.weight)
        return alpha * x1 + (1 - alpha) * x2

class view_transformer(nn.Module):
    def __init__(self, num_layers, feat_dim=128):
        super().__init__()
        self.num_layers = num_layers
        self.feat_dim = feat_dim
        self.layers = nn.ModuleDict({
            f'layer_{i}': nn.ModuleDict({
                'mlp': nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.Tanh()),
                'conv1': nn.Conv1d(feat_dim, feat_dim, kernel_size=1, groups=1),
                'conv2': nn.Conv1d(feat_dim, feat_dim, kernel_size=1, groups=1),
                'se': SELayer(feat_dim, reduction=16),
                'weights': WeightLayer()  # 假设这是自定义的加权模块
            }) for i in range(num_layers)
        })

    def forward(self, encoded_theta, query_theta, values):
        attention_weight = attention_block(query_theta, encoded_theta)
        # print(attention_weight)
        lead_n = len(values)
        v = torch.zeros_like(values[0])
        for i_ in range(lead_n):
            v += attention_weight[:, :, i_][:, :, None] * values[i_]
        if self.num_layers>0:
            values = torch.cat(values, dim=2)
            for i in range(self.num_layers):
                layer = self.layers[f'layer_{i}']
                v = layer['conv1'](v)
                values = layer['mlp'](values.transpose(2, 1))
                values = torch.chunk(values.transpose(2, 1), lead_n, dim=2)
                v1 = torch.zeros_like(v)
                for i_ in range(lead_n):
                    v1 += attention_weight[:, :, i_][:, :, None] * values[i_]
                values = torch.cat(values, dim=2)
                v1 = layer['conv2'](v1)
                v1 = layer['se'](v1)
                out = layer['weights'](v, v1)
                v = out
        else:
            out = v
        return out

class layer(nn.Module):
    """ """
    def __init__(self, cfg):
        super(layer, self).__init__()
        self.cfg = cfg
        self.view_transformer = view_transformer(feat_dim=128,num_layers=cfg.MODEL.layers)
        theta_encoder_len = cfg.MODEL.theta_L
        self.theta_encoder = ThetaEncoder(encoder_len=theta_encoder_len)

        self.mlp1 = nn.Sequential(nn.Linear((2 + 1) * 4, 128), nn.Tanh())
        self.mlp2 = nn.Sequential(nn.Linear((2 + 1) * 4, 128), nn.Tanh())
        self.mlp3 = nn.Sequential(nn.Linear((2 + 1) * 4, 128), nn.Tanh())
        self.mlpq = nn.Sequential(nn.Linear((2 + 1) * 4, 128), nn.Tanh())

        self.W_encoder1 = Encoder(
            backbone="resnet34",
            in_channel=1,
            use_first_pool=True,
            lead_num=1,
            init_channels=128,
        )
        self.W_encoder2 = Encoder(
            backbone="resnet34",
            in_channel=1,
            use_first_pool=True,
            lead_num=1,
            init_channels=128,
        )
        self.W_encoder3 = Encoder(
            backbone="resnet34",
            in_channel=1,
            use_first_pool=True,
            lead_num=1,
            init_channels=128,
        )

        if self.cfg.DATA.super_mode == 'optimization' or self.cfg.DATA.super_mode == 'transformation' or self.cfg.DATA.super_mode == 'finetune':
            self.lead_num = cfg.DATA.lead_num
            self.mlp_list = nn.ModuleList([
                copy.deepcopy(self.mlp3) for _ in range(self.lead_num-2)
            ])
            self.W_encoder_list = nn.ModuleList([
                copy.deepcopy(self.W_encoder3) for _ in range(self.lead_num-2)
            ])
            self.alpha = nn.Parameter(torch.zeros(self.lead_num, 2))

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            StableDoubleConv(128 * 1, 128, 2304),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            StableDoubleConv(128, 64, 4608),
            nn.Conv1d(64, 1, 3, padding=1),
        )

    def forward(self, x, input_thetas, query_theta, phase="train"):
        """Args
        :param x:   [B, lead_num, Length]
        :param input_thetas:  [B, lead_num, 2]
        :param query_theta:   [B, 2]
        :return: out: [B,1,Length]
        """
        if phase == "finetune":
            input_thetas = input_thetas + torch.tanh(self.alpha.unsqueeze(0))
        input_thetas = self.theta_encoder(input_thetas)  # [B, lead_num, 12]
        if self.cfg.DATA.super_mode == 'pretrain':
            lead_num = x.shape[1]
            encoded_theta1 = [self.mlp1(input_thetas[:, 0:1, :]), self.mlp2(input_thetas[:, 1:2, :])]
            encoded_theta2 = [
                self.mlp3(input_thetas[:, i:i + 1, :]) for i in range(2, lead_num)
            ]
            encoded_theta = encoded_theta1 + encoded_theta2
            encoded_theta = torch.cat(encoded_theta, dim=1)
            query_theta = self.theta_encoder(query_theta)
            query_theta = self.mlpq(query_theta)
            w_list = [
                self.W_encoder1(x[:,0:1,:], query_theta),
                self.W_encoder2(x[:, 1:2, :], query_theta)
            ] # ecg tokenizer --> [B, lead_num, 128, T]
            w_list2 = [
                self.W_encoder3(x[:,i:i+1,:], query_theta) for i in range(2,lead_num)
            ]
            w = w_list + w_list2

        else:
            encoded_theta1 = [self.mlp1(input_thetas[:, 0:1, :]), self.mlp2(input_thetas[:, 1:2, :])]
            encoded_theta2 = [
                self.mlp_list[i](input_thetas[:, i + 2:i + 3, :]) for i in range(self.lead_num-2)
            ]
            encoded_theta = encoded_theta1 + encoded_theta2
            encoded_theta = torch.cat(encoded_theta, dim=1)
            query_theta = query_theta.unsqueeze(1)
            query_theta = self.theta_encoder(query_theta)
            query_theta = self.mlpq(query_theta)
            w_list = [
                self.W_encoder1(x[:,0:1,:], query_theta),
                self.W_encoder2(x[:, 1:2, :], query_theta)
            ]
            w_list2 = [
                self.W_encoder_list[i](x[:,i+2:i+3,:], query_theta) for i in range(self.lead_num-2)
            ]
            w = w_list + w_list2

        v_q = self.view_transformer(encoded_theta,query_theta, w)

        out = self.decoder(v_q)
        out = torch.sigmoid(out / 3)
        return out

class layer1(nn.Module):
    """ """
    def __init__(self, cfg):
        super(layer1, self).__init__()
        self.cfg = cfg
        self.view_transformer = view_transformer(feat_dim=128,num_layers=cfg.MODEL.layers)
        theta_encoder_len = cfg.MODEL.theta_L
        self.theta_encoder = ThetaEncoder(encoder_len=theta_encoder_len)
        self.mlp1 = nn.Sequential(nn.Linear((2 + 1) * 4, 128), nn.Tanh())
        self.mlpq = nn.Sequential(nn.Linear((2 + 1) * 4, 128), nn.Tanh())
        self.W_encoder1 = Encoder(
            backbone="resnet34",
            in_channel=1,
            use_first_pool=True,
            lead_num=1,
            init_channels=128,
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            StableDoubleConv(128 * 1, 128, 2304),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            StableDoubleConv(128, 64, 4608),
            nn.Conv1d(64, 1, 3, padding=1),
        )
        self.alpha = nn.Parameter(torch.zeros(1, 2))
    def forward(self, x, input_thetas, query_theta, phase="train"):
        """Args
        :param x:   [B, lead_num, Length]
        :param input_thetas:  [B, lead_num, 2]
        :param query_theta:   [B, 2]
        :return: out: [B,1,Length]
        """
        if phase == "finetune":
            input_thetas = input_thetas + torch.tanh(self.alpha.unsqueeze(0))
        input_thetas = self.theta_encoder(input_thetas)  # [B, lead_num, 12]
        encoded_theta = self.mlp1(input_thetas[:, 0:1, :])
        query_theta = self.theta_encoder(query_theta)
        query_theta = self.mlpq(query_theta)
        w = [self.W_encoder1(x[:,0:1,:], query_theta)]
        v_q = self.view_transformer(encoded_theta,query_theta, w)
        out = self.decoder(v_q)
        out = torch.sigmoid(out / 3)
        return out

class layer2(nn.Module):
    """ """
    def __init__(self, cfg):
        super(layer2, self).__init__()
        self.cfg = cfg
        self.view_transformer = view_transformer(feat_dim=128,num_layers=cfg.MODEL.layers)
        theta_encoder_len = cfg.MODEL.theta_L
        self.theta_encoder = ThetaEncoder(encoder_len=theta_encoder_len)

        self.mlp1 = nn.Sequential(nn.Linear((2 + 1) * 4, 128), nn.Tanh())
        self.mlp2 = nn.Sequential(nn.Linear((2 + 1) * 4, 128), nn.Tanh())
        self.mlpq = nn.Sequential(nn.Linear((2 + 1) * 4, 128), nn.Tanh())

        self.W_encoder1 = Encoder(
            backbone="resnet34",
            in_channel=1,
            use_first_pool=True,
            lead_num=1,
            init_channels=128,
        )
        self.W_encoder2 = Encoder(
            backbone="resnet34",
            in_channel=1,
            use_first_pool=True,
            lead_num=1,
            init_channels=128,
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            StableDoubleConv(128 * 1, 128, 2304),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            StableDoubleConv(128, 64, 4608),
            nn.Conv1d(64, 1, 3, padding=1),
        )
        self.alpha = nn.Parameter(torch.zeros(2, 2))

    def forward(self, x, input_thetas, query_theta, phase="train"):
        """Args
        :param x:   [B, lead_num, Length]
        :param input_thetas:  [B, lead_num, 2]
        :param query_theta:   [B, 2]
        :return: out: [B,1,Length]
        """
        if phase == "finetune":
            input_thetas = input_thetas + torch.tanh(self.alpha.unsqueeze(0))
        input_thetas = self.theta_encoder(input_thetas)
        encoded_theta = [self.mlp1(input_thetas[:, 0:1, :]), self.mlp2(input_thetas[:, 1:2, :])]
        encoded_theta = torch.cat(encoded_theta, dim=1)
        query_theta = self.theta_encoder(query_theta)
        query_theta = self.mlpq(query_theta)

        w_list = [self.W_encoder1(x[:,0:1,:], query_theta),
            self.W_encoder2(x[:, 1:2, :], query_theta)
            ]
        w = w_list
        v_q = self.view_transformer(encoded_theta,query_theta, w)
        out = self.decoder(v_q)
        out = torch.sigmoid(out / 3)
        return out


