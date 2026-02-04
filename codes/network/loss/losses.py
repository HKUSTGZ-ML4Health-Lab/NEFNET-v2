import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class OurLoss1(torch.nn.Module):
    def __init__(self):
        super(OurLoss1, self).__init__()
        self.l1_loss = nn.L1Loss(reduction='mean')

    def forward(self, input0, input1):
        """
        :param input0: [B, 1, 512]
        :param input1: [B, 1, 512]
        :param target: [B, 1, 512]
        :return:
        """
        input0 = input0.detach()
        return self.l1_loss(input0, input1)
def angle_loss(x, y, z):
    vector_magnitude = torch.sqrt(x**2 + y**2 + z**2)
    cos_theta = z / vector_magnitude
    theta = torch.acos(cos_theta)
    # 将弧度转换为角度
    theta_degrees = theta * (180.0 / torch.pi)

    cos_alpha = x / vector_magnitude
    alpha = torch.acos(cos_alpha)
    # 将弧度转换为角度
    alpha_degrees = alpha * (180.0 / torch.pi)
    return alpha_degrees, theta_degrees
def lossl1(predict,target):
    l1_loss = nn.L1Loss(reduction='mean')
    loss = l1_loss(predict,target)
    return loss

def loss2(predict,target,features_all):
    l1_loss = nn.L1Loss(reduction='mean')
    loss1 = l1_loss(predict,target)
    loss2 = contrastive_loss(features_all, temperature=0.1)
    loss = loss1 + loss2
    return loss

def losswrapper(predict, predict_shuffle_1, predict_shuffle_2, predict_shuffle_3, target, cfg, rest_out=None, rest_view=None,
                loss1_gt=None, loss2_gt=None):
    loss_f_1 = OurLoss1().cuda()
    loss_f_2 = nn.L1Loss(reduction='mean').cuda() #MAE loss

    # loss1 loss2 which gt
    loss1_gt = predict if loss1_gt is None else loss1_gt
    loss2_gt = predict if loss2_gt is None else loss2_gt

    loss1 = loss_f_1(loss1_gt, predict_shuffle_1) if 1 in cfg.SOLVER.loss_using else 0.
    loss2 = loss_f_1(loss2_gt, predict_shuffle_2) if 2 in cfg.SOLVER.loss_using else 0.
    loss3 = loss_f_1(loss2_gt, predict_shuffle_3) if 3 in cfg.SOLVER.loss_using else 0.
    loss4 = loss_f_2(predict, target) if 4 in cfg.SOLVER.loss_using else 0.

    factor = cfg.SOLVER.loss_factor
    loss = loss1 * factor[0] + loss2 * factor[1] + loss3 * factor[2] + loss4 * factor[3]

    return loss, loss1 * factor[0], loss2 * factor[1], loss3 * factor[2], loss4 * factor[3]

class MSELead(nn.Module):
    def __init__(self):
        super(MSELead, self).__init__()
        self.loss_func = nn.MSELoss()

    def forward(self, input, target):
        loss_list = []
        # print('mean:', torch.mean(target, dim=(0, 2)))
        # print('std:', torch.std(target, dim=(0, 2)))
        for i in range(input.size(1)):
            loss_list.append(self.loss_func(input[:, i], target[:, i]))
        return torch.mean(torch.stack(loss_list))

def contrastive_loss(features_all, temperature=0.1):
    """
    features: (B, C, F)  # 假设已沿时序维度平均（或池化）
    目标：让同一Batch内不同导联的特征彼此远离
    """
    list = [feat.mean(dim=-1) for feat in features_all]
    features = torch.stack(list,dim=1)
    features = F.normalize(features, dim=-1)  # L2归一化
    B, C, D = features.shape

    loss = 0.0
    for b in range(B):
        # 当前样本的所有导联特征 (C, D)
        feat = features[b]  # (C, D)

        # 计算导联间的相似度矩阵 (C, C)
        sim_matrix = torch.matmul(feat, feat.T) / temperature

        # 对角线是同一导联（无损失），非对角线是不同导联（需拉开）
        mask = torch.eye(C, device=features.device)  # 对角线掩码
        neg_sim = sim_matrix * (1 - mask)  # 只保留不同导联的相似度

        # 惩罚不同导联的相似性（让它们远离）
        loss += torch.mean(neg_sim ** 2)  # 最小化相似度的平方

    return loss / B  # 按Batch平均

# class ContrastiveLoss1(nn.Module):
#     def __init__(self, temperature_init=0.07):
#         super(ContrastiveLoss1, self).__init__()
#         # 将温度参数设为可学习参数
#         self.logit_scale = nn.Parameter(torch.tensor(1/0.07))
#     def forward(self, features, index):
#         """
#         Args:
#             10 classification
#             features: w_one_lead_list, tuple:3 -> (B,128,1152).
#             index: meta['input_index'], list: 3 -> (B,).
#         """
#         # shape: (batch_size, embed_dim)
#         features_all = torch.cat(features,dim=0)
#         features_all = features_all.reshape(features_all.shape[0],-1)
#         index_all = torch.cat(index, dim=0)
#
#         # Clamp temperature such that logits are not scaled more than 100x.
#         # ln(100) = ~4.6052
#         self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
#         _scale = self.logit_scale.exp()
#
#         features_logits = _scale * features_all @ features_all.T
#
#         # Compute cross entropy loss: we compute log probabilities and take the
#         # diagonal elements as targets: image[i] should match text[i] in batch.
#         # Shift the targets according to rank of GPU process (we assume that all GPU processes have the same local batch size).
#         loss = F.cross_entropy(features_logits, index_all)
#         return loss
#
# class ContrastiveLoss2(nn.Module):
#     def __init__(self, temperature_init=0.07):
#         super(ContrastiveLoss2, self).__init__()
#         # 将温度参数设为可学习参数
#         self.logit_scale = nn.Parameter(torch.tensor(1/0.07))
#         # self.temperature = nn.Parameter(torch.tensor(temperature_init))
#     def forward(self, features):
#         """
#         Args:
#             features: w_one_lead_list, tuple:3 -> (B,128,1152).
#         """
#         # shape: (batch_size, embed_dim)
#         features_all = torch.cat(features,dim=0)
#         features_all = features_all.reshape(features_all.shape[0],-1)
#         index_all = torch.arange(features_all.shape[0], device=features_all.device)
#
#         # Clamp temperature such that logits are not scaled more than 100x.
#         # ln(100) = ~4.6052
#         self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
#         _scale = self.logit_scale.exp()
#
#         features_logits = _scale * features_all @ features_all.T
#
#         # Compute cross entropy loss: we compute log probabilities and take the
#         # diagonal elements as targets: image[i] should match text[i] in batch.
#         # Shift the targets according to rank of GPU process (we assume that all GPU processes have the same local batch size).
#         loss = F.cross_entropy(features_logits, index_all)
#
#         return loss

# def loss2(predict,target,input_position,position,input_theta,target_theta):
#     l1_loss = nn.L1Loss(reduction='mean')
#     loss1 = l1_loss(predict,target)
#
#     x1, y1, z1 = input_position[:,0:1,0],input_position[:,0:1,1],input_position[:,0:1,2]
#     x2, y2, z2 = input_position[:, 0:1, 3], input_position[:, 0:1, 4], input_position[:, 0:1, 5]
#     x3, y3, z3 = input_position[:, 0:1, 6], input_position[:, 0:1, 7], input_position[:, 0:1, 8]
#     x4, y4, z4 = position[:, 0:1, 0], position[:, 0:1, 1], position[:, 0:1, 2]
#     angle1 = angle_loss(x1, y1, z1)
#     angle2 = angle_loss(x2, y2, z2)
#     angle3 = angle_loss(x3, y3, z3)
#     angle4 = angle_loss(x4, y4, z4)
#
#     loss2 = l1_loss(angle1[0],input_theta[:,0,0:1]) + l1_loss(angle1[1],input_theta[:,0,1:]) + \
#             l1_loss(angle2[0],input_theta[:,1,0:1]) + l1_loss(angle2[1],input_theta[:,1,1:]) + \
#             l1_loss(angle3[0], input_theta[:, 2, 0:1]) + l1_loss(angle3[1], input_theta[:, 2, 1:]) + \
#             l1_loss(angle4[0], target_theta[:, 0:1]) + l1_loss(angle4[1], target_theta[:, 1:])
#     loss = loss1 + loss2
#     return loss