from .nefnet_plus import layer,layer1,layer2
from .loss import losswrapper, MSELead, lossl1, loss2
from .model_nefnet import Model_nefnet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss, L1Loss, CrossEntropyLoss

def build_model(cfg):
    model_name = cfg.MODEL.model
    print('build model.{}'.format(model_name))
    if model_name == 'nefnet':
        return Model_nefnet(theta_encoder_len=cfg.MODEL.theta_L, lead_num=cfg.DATA.lead_num)
    elif model_name == 'layer':
        return layer(cfg)
    elif model_name == 'layer1':
        return layer1(cfg)
    elif model_name == 'layer2':
        return layer2(cfg)
    else:
        raise ValueError('build model: model name error')

def build_loss(cfg):
    loss_name = cfg.MODEL.loss
    if loss_name == 'l0':
        return losswrapper
    elif loss_name == 'l1':
        return lossl1
    elif loss_name == 'l2':
        return loss2
    else:
        raise ValueError('build loss: loss name error')



