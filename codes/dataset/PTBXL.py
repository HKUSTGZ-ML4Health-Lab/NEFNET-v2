import os
import torch
import glob
from torch.utils.data import Dataset
import numpy as np
import json
import random
import pickle
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import wfdb
from scipy import signal

def get_subdirectories(path):
    subdirs = []
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            subdirs.append(full_path)
    return subdirs

# 示例用法
# folder_path = 'your/folder/path'
# subdirectories = get_subdirectories(folder_path)
# print(subdirectories)


class PTBVXL_finetune(Dataset):
    def __init__(self, cfg, transform=None):
        self.dataset = None
        self.label_name = None
        self.label_dir = None
        self.cfg = cfg
        self.transform = transform
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
        self._read_data(cfg)

    def _read_data(self, cfg):
        self.data_root = cfg.DATA.ptb_xl_root
        label_path = cfg.DATA.ptbxl_test_label_path
        with open(label_path) as f:
            self.dataset = f.read().splitlines()
##no test
    def __getitem__(self, index):
        file_path = os.path.join(self.data_root, self.dataset[index])
        record = wfdb.rdrecord(file_path)
        ECG = record.p_signal.T
        new_order = [0, 1, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5]
        ECG = ECG[new_order]
        source_data = ECG

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

    def angle_jitter(self, angle, jitter_factor):
        jitter_angle = jitter_factor / 180 * np.pi
        jitter = np.random.normal(scale=jitter_angle, size=angle.shape)
        angle = angle + jitter
        return angle

class PTBVXL(Dataset):
    def __init__(self, cfg, phase, transform=None):
        self.dataset = None
        self.label_name = None
        self.label_dir = None
        self.cfg = cfg
        self.phase = phase
        self.transform = transform
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
        self._read_data(cfg)

    def _read_data(self, cfg):
        self.data_root = cfg.DATA.ptb_xl_root
        label_path = cfg.DATA.ptbxl_test_label_path
        with open(label_path) as f:
            self.dataset = f.read().splitlines()
##no test
    def __getitem__(self, index):
        file_path = os.path.join(self.data_root, self.dataset[index])
        record = wfdb.rdrecord(file_path)
        ECG = record.p_signal.T
        Lenght = len(ECG[0, :])
        new_order = [0, 1, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5]
        ECG = ECG[new_order]

        # random crop
        if Lenght>4608:
            random_index_on = random.sample(range(0, Lenght - 4608), k=1)[0]
            random_index_off = random.sample(range(random_index_on + 2048 ,4608 + random_index_on), k=1)[0]
        else:
            random_index_on = 0
            random_index_off = random.sample(range(2048, Lenght), k=1)[0]

        data = ECG[:12, random_index_on:random_index_off]

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

    def angle_jitter(self, angle, jitter_factor):
        jitter_angle = jitter_factor / 180 * np.pi
        jitter = np.random.normal(scale=jitter_angle, size=angle.shape)
        angle = angle + jitter
        return angle