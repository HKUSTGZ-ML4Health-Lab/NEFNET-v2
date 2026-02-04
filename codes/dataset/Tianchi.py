import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import json
import random
from config import cfg
import neurokit2 as nk
import dill
import pickle
import matplotlib.pyplot as plt

class EcgTianChiInterval(Dataset):
    r""" 训练测试数据集
    Return
        Data:[B, 3, 4608]
        随机采样结束点end_point[2048:4608]
        有效信号为[0:end_point], [end_point:]填充为0
    """
    def __init__(self, cfg, phase, transform=None):
        self.dataset = None
        self.label_name = None
        self.label_dir = None
        self.cfg = cfg
        self.phase = phase
        self.transform = transform
        self.theta = np.array([[np.pi / 2, np.pi / 2],  # I - np.pi*4/18
                               [np.pi * 5 / 6, np.pi / 2],  # II
                               [np.pi / 2, -np.pi / 18],  # v1
                               [np.pi / 2, np.pi / 18],  # v2
                               [np.pi * (19/36), np.pi / 12],  # v3
                               [np.pi * (11/20), np.pi / 6],  # v4
                               [np.pi * (16/30), np.pi / 3],  # v5
                               [np.pi * (16/30), np.pi / 2],  # v6
                               [np.pi * (5/6), -np.pi / 2],  # III
                               [np.pi * (1/3), -np.pi / 2],  # aVR
                               [np.pi * (1/3), np.pi / 2],  # aVL
                               [np.pi * 1, np.pi / 2],  # aVF
                               ])
        self.alpha = -1*np.pi/18
        self.theta[0:2,:] = self.theta[0:2,:] - self.alpha
        self.theta[4, :] = self.theta[4, :] - self.alpha
        self._read_data(cfg, phase)
    def _read_data(self, cfg, phase):
        if phase == 'train':
            label_path = cfg.DATA.train_label_path
            with open(label_path) as f:
                self.dataset = f.read().splitlines()
            self.data_root = cfg.DATA.train_data_root

        else:
            label_path = cfg.DATA.test_label_path
            with open(label_path) as f:
                self.dataset = f.read().splitlines()
            self.data_root = cfg.DATA.test_data_root
    def angle_jitter(self, angle, jitter_factor):
        jitter_angle = jitter_factor / 180 * np.pi
        jitter = np.random.normal(scale=jitter_angle, size=angle.shape)
        angle = angle + jitter
        return angle

    def __getitem__(self, index):
        file_path = os.path.join(self.data_root, self.dataset[index])
        data = []
        with open(file_path, 'r') as file:
            i = 0
            for line in file:
                i = i + 1
                if i > 1:
                    line = line.strip().split(' ')
                    data.append(
                        [float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]),
                         float(line[6]), float(line[7])])
            source_data = np.asarray(data)
            source_data = source_data.T

        # add III, aVR, aVL, aVF. III = II - I, aVR = -0.5(I + II), aVL = I - 0.5II, aVF = II - 0.5I
        III = source_data[1:2, :] - source_data[0:1, :]
        aVR = - 0.5 * (source_data[0:1, :] + source_data[1:2, :])
        aVL = source_data[0:1, :] - 0.5 * source_data[1:2, :]
        aVF = source_data[1:2, :] - 0.5 * source_data[0:1, :]
        ECG = np.concatenate([source_data, III, aVR, aVL, aVF], axis=0)
        Lenght = len(ECG[0, :])

        # random crop
        if Lenght>4608:
            random_index_on = random.sample(range(0, Lenght - 4608), k=1)[0]
            random_index_off = random.sample(range(random_index_on + 2048 ,4608 + random_index_on), k=1)[0]
        else:
            random_index_on = 0
            random_index_off = random.sample(range(2048, Lenght), k=1)[0]

        data = ECG[:, random_index_on:random_index_off]
        # normalized
        if self.cfg.DATA.Normalize == True:
            max_, min_ = np.max(data), np.min(data)
            data = (data - min_) / (max_ - min_)
        padding_len = 4608 - (random_index_off - random_index_on)
        data = np.pad(data, ((0, 0), (0, padding_len)), mode='constant', constant_values=0)
        # get noise
        if self.cfg.DATA.noise == True:
            noise_region = data
            noise_std = np.std(noise_region, axis=1)
            noise = np.random.normal(loc=0, scale=noise_std, size=(data.shape[-1], 12))
        else:
            noise = np.zeros((4608, 12), dtype=np.float32)
        synthesis_index = random.sample(self.cfg.DATA.data_synthesis,1)[0]
        synthesis_view = data[synthesis_index]
        synthesis_theta = self.theta[synthesis_index]
        if self.cfg.DATA.super_mode == 'pretrain':
            meta = {
                'id': self.dataset[index],
                'data': data.astype(np.float32),
                'thetas': self.theta.astype(np.float32),
                'synthesis_view': np.array(synthesis_view).astype(np.float32),
                'synthesis_theta': np.array(synthesis_theta).astype(np.float32),
                'end_point': random_index_off - random_index_on,
                'noise': noise,
            }
        else:
            if self.cfg.DATA.lead_num == 3:
                input_index = [0, 1, 4]
            elif self.cfg.DATA.lead_num == 4:
                input_index = [0, 1, 2, 4]
            elif self.cfg.DATA.lead_num == 5:
                input_index = [0, 1, 2, 3, 4]
            rest_index = [x for x in range(2, 8) if x not in input_index + self.cfg.DATA.data_synthesis]
            target_index = random.sample(rest_index, 1)[0]
            meta = {
                'input_index': input_index,
                'input_data': data[input_index].astype(np.float32),
                'input_theta': np.array(self.theta[input_index]).astype(np.float32),
                'target_view': np.array(data[target_index]).astype(np.float32),
                'target_theta': np.array(self.theta[target_index]).astype(np.float32),
                'id': self.dataset[index],
                'synthesis_view': np.array(synthesis_view).astype(np.float32),
                'synthesis_theta': np.array(synthesis_theta).astype(np.float32),
                'end_point': random_index_off - random_index_on,
                'target_index': target_index,
                'noise': noise[:, target_index],
                'data': data.astype(np.float32),
            }
        return meta
    def __len__(self):
        return len(self.dataset)

class EcgTianChi_finetune(Dataset):
    r""" alpha微调验证数据集2
    Return:
        Data:[1, 3, 4608]
    """
    def __init__(self, cfg, transform=None):
        self.dataset = None
        self.label_name = None
        self.label_dir = None
        self.cfg = cfg
        self.transform = transform
        self.token_size = cfg.MODEL.token_sizes
        self.theta = np.array([[np.pi / 2, np.pi / 2],  # I
                                   [np.pi * 5 / 6, np.pi / 2],  # II
                                   [np.pi / 2, -np.pi / 18],  # v1
                                   [np.pi / 2, np.pi / 18],  # v2
                                   [np.pi * (19 / 36), np.pi / 12],  # v3
                                   [np.pi * (11 / 20), np.pi / 6],  # v4
                                   [np.pi * (16 / 30), np.pi / 3],  # v5
                                   [np.pi * (16 / 30), np.pi / 2],  # v6
                                   [np.pi * (5 / 6), -np.pi / 2],  # III
                                   [np.pi * (1 / 3), -np.pi / 2],  # aVR
                                   [np.pi * (1 / 3), np.pi / 2],  # aVL
                                   [np.pi * 1, np.pi / 2],  # aVF
                                   ])
        self.alpha = -0*np.pi/18
        self.theta[0:2,:] = self.theta[0:2,:] - self.alpha
        self.theta[4, :] = self.theta[4, :] - self.alpha
        self._read_data(cfg)
    def _read_data(self, cfg):
        label_path = cfg.DATA.test_label_path
        with open(label_path) as f:
            self.dataset = f.read().splitlines()
        self.data_root = cfg.DATA.test_data_root
    def angle_jitter(self, angle, jitter_factor):
        jitter_angle = jitter_factor / 180 * np.pi
        jitter = np.random.normal(scale=jitter_angle, size=angle.shape)
        angle = angle + jitter
        return angle

    def __getitem__(self, index):
        file_path = os.path.join(self.data_root, self.dataset[index])  ##attention!
        data = []
        with open(file_path, 'r') as file:
            i = 0
            for line in file:
                i = i + 1
                if i > 1:
                    line = line.strip().split(' ')
                    data.append(
                        [float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]),
                         float(line[5]),
                         float(line[6]), float(line[7])])
            source_data = np.asarray(data)
            source_data = source_data.T

        # add III, aVR, aVL, aVF. III = II - I, aVR = -0.5(I + II), aVL = I - 0.5II, aVF = II - 0.5I
        III = source_data[1:2, :] - source_data[0:1, :]
        aVR = - 0.5 * (source_data[0:1, :] + source_data[1:2, :])
        aVL = source_data[0:1, :] - 0.5 * source_data[1:2, :]
        aVF = source_data[1:2, :] - 0.5 * source_data[0:1, :]
        source_data = np.concatenate([source_data, III, aVR, aVL, aVF], axis=0)

        # normalized
        max, min = np.max(source_data), np.min(source_data)
        source_data = (source_data - min) / (max - min)

        end_point = 2500
        finetune_data = source_data[:, 0:end_point]
        finetune_data = np.pad(finetune_data, ((0, 0), (0, 4608-end_point)), mode='constant', constant_values=0)

        test_data = source_data[:, 5000-end_point:5000]
        test_data = np.pad(test_data, ((0, 0), (0, 4608-end_point)), mode='constant', constant_values=0)

        if self.cfg.DATA.lead_num == 3:
            input_index = [0, 1, 4]
            rest_index = [x for x in range(2, 8) if x not in input_index + self.cfg.DATA.data_synthesis]
        elif self.cfg.DATA.lead_num == 4:
            input_index = [0, 1, 2, 4]
            rest_index = [x for x in range(2, 8) if x not in input_index + self.cfg.DATA.data_synthesis]
        elif self.cfg.DATA.lead_num == 5:
            input_index = [0, 1, 2, 3, 4]
            rest_index = [x for x in range(2, 8) if x not in input_index + self.cfg.DATA.data_synthesis]
        input_theta = self.theta[input_index]
        synthesis_index = random.sample(self.cfg.DATA.data_synthesis,1)[0]
        synthesis_theta = self.theta[synthesis_index]

        meta = {
            'finetune_input': finetune_data[input_index].astype(np.float32),
            'finetune_data': finetune_data.astype(np.float32),
            'rest_index': rest_index,
            'thetas': np.array(self.theta).astype(np.float32),
            'finetune_synthesis': finetune_data[synthesis_index].astype(np.float32),

            'test_input': test_data[input_index].astype(np.float32),
            'test_target': test_data[synthesis_index],

            'input_theta': np.array(input_theta).astype(np.float32),
            'synthesis_theta': np.array(synthesis_theta).astype(np.float32),
            'end_point': end_point,
            'id': self.dataset[index]
        }
        return meta

    def __len__(self):
        return len(self.dataset)

 ##数据略微有问题
##数据有大问题
# class EcgTianChi_finetune(Dataset): #数据有问题
#     def __init__(self, cfg, phase, transform=None):
#         self.dataset = None
#         self.label_name = None
#         self.label_dir = None
#         self.cfg = cfg
#         self.phase = phase
#         self.transform = transform
#         self.token_size = 128
#         self.theta = np.array([[np.pi / 2, np.pi / 2],  # I
#                                [np.pi * 5 / 6, np.pi / 2],  # II
#                                [np.pi / 2, -np.pi / 18],  # v1
#                                [np.pi / 2, np.pi / 18],  # v2
#                                [np.pi * (19/36), np.pi / 12],  # v3
#                                [np.pi * (11/20), np.pi / 6],  # v4
#                                [np.pi * (16/30), np.pi / 3],  # v5
#                                [np.pi * (16/30), np.pi / 2],  # v6
#                                [np.pi * (5/6), -np.pi / 2],  # III
#                                [np.pi * (1/3), -np.pi / 2],  # aVR
#                                [np.pi * (1/3), np.pi / 2],  # aVL
#                                [np.pi * 1, np.pi / 2],  # aVF
#                                ])
#         self._read_data(cfg, phase)
#     def _read_data(self, cfg, phase):
#         if phase == 'train':
#             label_path = cfg.DATA.train_label_path
#             with open(label_path) as f:
#                 self.dataset = f.read().splitlines()
#             self.data_root = cfg.DATA.train_data_root
#             self.label_dir = cfg.DATA.label_root
#         else:
#             label_path = cfg.DATA.test_label_path
#             with open(label_path) as f:
#                 self.dataset = f.read().splitlines()
#             self.dataset = self.dataset[0:200]
#             self.data_root = cfg.DATA.test_data_root
#             self.label_dir = cfg.DATA.label_root
#
#     def angle_jitter(self, angle, jitter_factor):
#         jitter_angle = jitter_factor / 180 * np.pi
#         jitter = np.random.normal(scale=jitter_angle, size=angle.shape)
#         angle = angle + jitter
#
#         return angle
#
#     def __getitem__(self, index):
#         file_path = os.path.join(self.data_root, self.dataset[105]) ##attention!
#         data = []
#         with open(file_path, 'r') as file:
#             i = 0
#             for line in file:
#                 i = i + 1
#                 if i > 1:
#                     line = line.strip().split(' ')
#                     data.append(
#                         [float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]),
#                          float(line[6]), float(line[7])])
#             source_data = np.asarray(data)
#             source_data = source_data.T
#
#         # add III, aVR, aVL, aVF. III = II - I, aVR = -0.5(I + II), aVL = I - 0.5II, aVF = II - 0.5I
#         III = source_data[1:2, :] - source_data[0:1, :]
#         aVR = - 0.5 * (source_data[0:1, :] + source_data[1:2, :])
#         aVL = source_data[0:1, :] - 0.5 * source_data[1:2, :]
#         aVF = source_data[1:2, :] - 0.5 * source_data[0:1, :]
#         source_data = np.concatenate([source_data, III, aVR, aVL, aVF], axis=0)
#
#         data = source_data[:,0:4992]
#         # normalized
#         max_, min_ = np.max(data), np.min(data)
#         data = (data - min_) / (max_ - min_)
#
#         # finetune_off = random.sample(range(500,2496), k=1)[0]
#         # finetune_data = data[:,0:finetune_off]
#         # finetune_len = 4992 - finetune_off
#         # test_off = random.sample(range(3000,4992), k=1)[0]
#         # test_data = data[:,2496:test_off]
#         # test_len = 4992 - test_off + 2496
#
#         finetune_off = 3000
#         finetune_data = data[:,0:finetune_off]
#         finetune_len = 4992 - finetune_off
#         test_off = 4992
#         test_data = data[:,1992:test_off]
#         test_len = 4992 - test_off + 1992
#
#         finetune_data = np.pad(finetune_data, ((0, 0), (0, finetune_len)), mode='constant', constant_values=0)
#         test_data = np.pad(test_data, ((0, 0), (0, test_len)), mode='constant', constant_values=0)
#
#         # angle jitter
#         theta_ = self.theta
#         if self.cfg.MODEL.jitter_factor > 0 and self.phase == 'train':
#             theta_ = self.angle_jitter(theta_, self.cfg.MODEL.jitter_factor)
#
#         input_index = [0,1,2]
#         input_theta = theta_[input_index]
#
#         supervision_lead = [x for x in range(0,12)]
#         rest_index = [x for x in supervision_lead if x not in input_index]
#
#         target_index = random.sample(rest_index, 1)[0]
#
#         target_theta = theta_[target_index]
#
#         rest_view = source_data[rest_index]
#         rest_theta = theta_[rest_index]
#         meta = {
#             'finetune_input': finetune_data[input_index].astype(np.float32),
#             'finetune_target':finetune_data[target_index],
#
#             'test_input': test_data[input_index].astype(np.float32),
#             'test_target': test_data[target_index],
#
#             'input_theta': np.array(input_theta).astype(np.float32),
#             'target_theta': np.array(target_theta).astype(np.float32),
#             'id': self.dataset[index],
#
#             'ori_data': source_data,
#             'rest_view': rest_view,
#             'rest_theta': np.array(rest_theta).astype(np.float32),
#
#             'test_end_point':4992 - test_len
#         }
#         return meta
#
#     def __len__(self):
#         return len(self.dataset)