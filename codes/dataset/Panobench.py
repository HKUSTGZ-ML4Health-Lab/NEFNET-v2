import numpy as np
from torch.utils.data import Dataset
import glob
import scipy.io
import random
from scipy.signal import resample
# import json
# import pickle
# import torch
# import argparse
# import os
# import neurokit2 as nk
# from scipy import signal
# import warnings
# from scipy.interpolate import interp1d
# import pandas as pd
# from scipy.io import loadmat
# import matplotlib.pyplot as plt
# import torch.nn as nn
# import torch.nn.functional as F
# import torch

class Panobench(Dataset):
    def __init__(self,cfg, phase):
        self.cfg = cfg
        if phase == 'train':
            self.dataset = glob.glob('/home/ALL_Lab_Data/zzh/train_bspm/*.mat')
        else:
            self.dataset = glob.glob('/home/ALL_Lab_Data/zzh/test_bspm/*.mat')
        self.theta = np.array([[np.pi / 2, np.pi / 2],[np.pi * 5 / 6, np.pi / 2], # I ,II
            [np.pi * (106/180), -np.pi * (102/180)], [np.pi * (121/180), np.pi * (-101/180)],  # 1,2
                               [np.pi * (132/180), np.pi * (-99/180)], [np.pi * (52/180), np.pi * (-83/180)],  # 3,4
                               [np.pi * (68 / 180), np.pi * (-78 / 180)], [np.pi * (90 / 180), np.pi * (-74 / 180)],  # 5,6
                               [np.pi * (109 / 180), np.pi * (-75 / 180)], [np.pi * (125 / 180), np.pi * (-77 / 180)],  # 7,8
                               [np.pi * (137 / 180), np.pi * (-81 / 180)], [np.pi * (43 / 180), np.pi * (-74 / 180)],  # 9,10
                               [np.pi * (63 / 180), np.pi * (-61 / 180)], [np.pi * (90 / 180), np.pi * (-54 / 180)],  # 11,12
                               [np.pi * (113 / 180), np.pi * (-55 / 180)], [np.pi * (131 / 180), np.pi * (-62 / 180)],  # 13,14
                               [np.pi * (144 / 180), np.pi * (-70 / 180)], [np.pi * (30 / 180), np.pi * (-73 / 180)],  # 15,16
                               [np.pi * (54 / 180), np.pi * (-51 / 180)], [np.pi * (90 / 180), np.pi * (-33 / 180)],  # 17,18
                               [np.pi * (118 / 180), np.pi * (-40 / 180)], [np.pi * (137 / 180), np.pi * (-54 / 180)],  # 19,20
                               [np.pi * (149 / 180), np.pi * (-64 / 180)], [np.pi * (20 / 180), np.pi * (70 / 180)],  # 21,22
                               [np.pi * (48 / 180), np.pi * (42 / 180)], [np.pi * (90 / 180), np.pi * (11 / 180)],  # 23,24
                               [np.pi * (122 / 180), np.pi * (32 / 180)], [np.pi * (141 / 180), np.pi * (51 / 180)],  # 25,26
                               [np.pi * (153 / 180), np.pi * (63 / 180)], [np.pi * (30 / 180), np.pi * (69 / 180)],  # 27,28
                               [np.pi * (54 / 180), np.pi * (48 / 180)], [np.pi * (90 / 180), np.pi * (32 / 180)],  # 29,30
                               [np.pi * (119 / 180), np.pi * (41 / 180)], [np.pi * (139 / 180), np.pi * (55 / 180)],  # 31,32
                               [np.pi * (152 / 180), np.pi * (67 / 180)], [np.pi * (40 / 180), np.pi * (80 / 180)],  # 33,34
                               [np.pi * (60 / 180), np.pi * (71 / 180)], [np.pi * (90 / 180), np.pi * (65 / 180)],  # 35,36
                               [np.pi * (117 / 180), np.pi * (66 / 180)], [np.pi * (135 / 180), np.pi * (69 / 180)],  # 37,38
                               [np.pi * (147 / 180), np.pi * (77 / 180)], [np.pi * (112 / 180), np.pi * (105 / 180)],  # 39,40
                               [np.pi * (129 / 180), np.pi * (103 / 180)], [np.pi * (140 / 180), np.pi * (100 / 180)]   # 41,42
                               ])
        self.alpha = 0 / 18 * np.pi

    def __getitem__(self, index):
        file_path = self.dataset[index]
        mat = scipy.io.loadmat(file_path)
        data = mat['bspm'][400:,:42]
        ra, la, ll = data[:,3:4], data[:,33:34], data[:,38:39]
        l1 = la - ra
        l2 = ll - ra
        data2 = data - (ra + la + ll) / 3
        data2 = np.concatenate((l1, l2, data2), axis=1)
        num_original_samples = len(data2[:, 0])
        Length = num_original_samples * 2 #采样率由250-500
        data2 = resample(data2, Length)
        random_index_on = random.sample(range(400, Length - 4608), k=1)[0]
        random_index_off = 4608 + random_index_on
        data3 = data2[random_index_on:random_index_off,:]
        max_, min_ = np.max(data3), np.min(data3)
        ECG = (data3 - min_) / (max_ - min_)
        data = ECG.T
        if self.cfg.DATA.lead_num == 3:
            input_index = [0, 1, 29]
            synthesis_index = [x for x in range(0, 44) if x not in input_index + self.cfg.DATA.bspm_supervision]
        elif self.cfg.DATA.lead_num == 5:
            input_index = [0, 1, 2, 29, 41]
            synthesis_index = [x for x in range(0, 44) if x not in input_index + self.cfg.DATA.bspm_supervision]
        elif self.cfg.DATA.lead_num == 7:
            input_index = [0, 1, 2, 14, 29, 35, 41]
            synthesis_index = [x for x in range(0, 44) if x not in input_index + self.cfg.DATA.bspm_supervision]
        elif self.cfg.DATA.lead_num == 9:
            input_index = [0, 1, 2, 9, 14, 20, 29, 35, 41]
            synthesis_index = [x for x in range(0, 44) if x not in input_index + self.cfg.DATA.bspm_supervision]
        rest_index = self.cfg.DATA.bspm_supervision
        target_index = random.sample(rest_index,1)[0]
        synthesis_target_index = random.sample(synthesis_index,1)[0]
        synthesis_view= data[synthesis_target_index,:]
        synthesis_theta = self.theta[synthesis_target_index]
        noise = np.zeros(4608)
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
            'noise': noise,
            'data':data
        }
        return meta
    def __len__(self):
        return len(self.dataset)

def filter(raw_ecg):
    from scipy import signal
    fs = 250

    cutoff_high = 0.5
    nyq = 0.5 * fs
    high = cutoff_high / nyq
    b_high, a_high = signal.butter(2, high, btype='highpass')
    ecg_hp = np.apply_along_axis(lambda x: signal.filtfilt(b_high, a_high, x), 1, raw_ecg)

    f0 = 50.0
    Q = 30.0
    b_notch, a_notch = signal.iirnotch(f0, Q, fs)
    ecg_notch = np.apply_along_axis(lambda x: signal.filtfilt(b_notch, a_notch, x), 1, ecg_hp)

    # 3. 低通滤波：去除高频噪声
    cutoff_low = 100
    low = cutoff_low / nyq
    b_low, a_low = signal.butter(4, low, btype='lowpass')
    ecg_filtered = np.apply_along_axis(lambda x: signal.filtfilt(b_low, a_low, x), 1, ecg_notch)

    return ecg_filtered

def Split_data():
    dataset_train = glob.glob('/home/ALL_Lab_Data/zzh/train_bspm/*.mat')
    dataset_test = glob.glob('/home/ALL_Lab_Data/zzh/test_bspm/*.mat')
    import numpy as np
    from scipy.io import savemat
    window_size = 2500
    overlap = 1000
    step = window_size - overlap
    index = 0
    for file_path in dataset_test:
        mat = scipy.io.loadmat(file_path)
        data = mat['bspm'][100:, :42].T
        ra, la, ll = data[3:4, :], data[33:34, :], data[38:39, :]
        l1 = la - ra
        l2 = ll - ra
        source_data = data - (ra + la + ll) / 3
        source_data = np.concatenate((l1, l2, source_data), axis=0)
        III = source_data[1:2, :] - source_data[0:1, :]
        aVR = - 0.5 * (source_data[0:1, :] + source_data[1:2, :])
        aVL = source_data[0:1, :] - 0.5 * source_data[1:2, :]
        aVF = source_data[1:2, :] - 0.5 * source_data[0:1, :]
        ECG = np.concatenate([source_data, III, aVR, aVL, aVF], axis=0)
        ECG = filter(ECG)

        length = len(ECG[0, :])
        n_windows = (length - overlap) // step
        for i in range(n_windows):
            Panobench = ECG[:, i*step:i*step+2500]
            index+=1
            save_path = '/Panobench/test/{}.mat'.format(index)
            savemat(save_path, {'Panobench': Panobench})


## training dataset, splitted ECG
# paths_root = glob.glob('/home/ALL_Lab_Data/zzh/rawdata/*')
# path_root = paths_root[0]
# paths1 = os.path.join(path_root, 'ep1.mat')
# paths2 = os.path.join(path_root, 'ep2.mat')
# paths3 = os.path.join(path_root, 'ep3.mat')
# paths5 = os.path.join(path_root, 'ep5.mat')
# train_path0 = [paths1,paths2,paths3,paths5]
# test_path0 = [os.path.join(path_root, 'ep4.mat'),os.path.join(path_root, 'ep0.mat')]
# mat0 = scipy.io.loadmat(train_path0[0])
# mat1 = scipy.io.loadmat(train_path0[1])
# mat2 = scipy.io.loadmat(train_path0[2])
# mat3 = scipy.io.loadmat(train_path0[3])
# ##------------------------------     test     -----------------------------------
# mat1 = scipy.io.loadmat(test_path0[0])
# mat2 = scipy.io.loadmat(test_path0[1])
# # -------------------------------              ------------------------------------
# ecg = mat0['bspm'][200:,:]
# length = ecg.shape[0]
# num_chunks = length // 1000
# if length % 1000 !=0 and ((num_chunks+1)*1000 - length)>500:
#     padding_len = (num_chunks+1)*1000 - length
#     ecg = np.pad(ecg,((0, padding_len), (0, 0)), mode='constant', constant_values=0)
#     splitted_ecg0 = np.array_split(ecg, num_chunks+1)
# elif length % 1000 !=0 and ((num_chunks+1)*1000 - length)<500:
#     ecg = ecg[:num_chunks*1000,:]
#     splitted_ecg0 = np.array_split(ecg, num_chunks)
# elif length % 1000 ==0:
#     splitted_ecg0 = np.array_split(ecg, num_chunks)
#
# ##------------------------------     test     -----------------------------------
# ecg = mat1['bspm'][200:,:]
# length = ecg.shape[0]
# num_chunks = length // 1000
# if length % 1000 !=0 and ((num_chunks+1)*1000 - length)>500:
#     padding_len = (num_chunks+1)*1000 - length
#     ecg = np.pad(ecg,((0, padding_len), (0, 0)), mode='constant', constant_values=0)
#     splitted_ecg1 = np.array_split(ecg, num_chunks+1)
# elif length % 1000 !=0 and ((num_chunks+1)*1000 - length)<500:
#     ecg = ecg[:num_chunks*1000,:]
#     splitted_ecg1 = np.array_split(ecg, num_chunks)
# elif length % 1000 ==0:
#     splitted_ecg1 = np.array_split(ecg, num_chunks)
#
# ecg = mat2['bspm'][200:,:]
# length = ecg.shape[0]
# num_chunks = length // 1000
# if length % 1000 !=0 and ((num_chunks+1)*1000 - length)>500:
#     padding_len = (num_chunks+1)*1000 - length
#     ecg = np.pad(ecg,((0, padding_len), (0, 0)), mode='constant', constant_values=0)
#     splitted_ecg2 = np.array_split(ecg, num_chunks+1)
# elif length % 1000 !=0 and ((num_chunks+1)*1000 - length)<500:
#     ecg = ecg[:num_chunks*1000,:]
#     splitted_ecg2 = np.array_split(ecg, num_chunks)
# elif length % 1000 ==0:
#     splitted_ecg2 = np.array_split(ecg, num_chunks)
# # -------------------------------              ------------------------------------
# ecg = mat3['bspm'][200:,:]
# length = ecg.shape[0]
# num_chunks = length // 1000
# if length % 1000 !=0 and ((num_chunks+1)*1000 - length)>=500:
#     padding_len = (num_chunks+1)*1000 - length
#     ecg = np.pad(ecg,((0, padding_len), (0, 0)), mode='constant', constant_values=0)
#     splitted_ecg3 = np.array_split(ecg, num_chunks+1)
# elif length % 1000 !=0 and ((num_chunks+1)*1000 - length)<500:
#     ecg = ecg[:num_chunks*1000,:]
#     splitted_ecg3 = np.array_split(ecg, num_chunks)
# elif length % 1000 ==0:
#     splitted_ecg3 = np.array_split(ecg, num_chunks)
#
# splitted_ecg = splitted_ecg0 + splitted_ecg1 + splitted_ecg2 + splitted_ecg3
#
# import numpy as np
# from scipy.io import savemat
# root_path = path_root + '/train'
# os.makedirs(root_path,exist_ok=True)
# # 遍历列表中的每个数据项，并将其保存为.mat文件
# for i, data in enumerate(splitted_ecg):
#     # 将数据项转换为numpy数组，假设是一个二维数组
#     mat_data = np.array(data)
#     # 创建一个字典，键是您想在MATLAB中使用的变量名
#     mat_dict = {'bspm': mat_data}
#     # 保存.mat文件，文件名中包含索引以区分不同的文件
#     path = os.path.join(root_path,f'data_{i + 1}.mat')
#     savemat(path, mat_dict)




# i = 10
# paths = glob.glob('/home/ALL_Lab_Data/zzh/rawdata/*'+'/ep0.mat')
# mat = scipy.io.loadmat(paths[i])
# data = mat['bspm'][:, :42]
# ra, la, ll = data[:, 3:4], data[:, 33:34], data[:, 38:39]
# data2 = data - (ra + la + ll) / 3

# Noisyroom_path_root = '/home/zhanzhehui_min/Engineering_project/ECG_Panorama/data/Nijmegen-2004-12-09/Interventions/Noisy_room'
# Meg_path_root = '/home/zhanzhehui_min/Engineering_project/ECG_Panorama/data/Nijmegen-2004-12-09/Interventions/MEG_room'
# Noisyroom_paths = glob.glob(r'/home/zhanzhehui_min/Engineering_project/ECG_Panorama/data/Nijmegen-2004-12-09/Interventions/Noisy_room/*.mat') # 噪环境数据
# Meg_paths = glob.glob(r'/home/zhanzhehui_min/Engineering_project/ECG_Panorama/data/Nijmegen-2004-12-09/Interventions/MEG_room/*.mat') # 静电屏蔽室数据
#
# Meshes_paths = glob.glob(r'/home/zhanzhehui_min/Engineering_project/ECG_Panorama/data/Nijmegen-2004-12-09/Meshes/*.mat') ##所有物理模型
# bspm_65leads = '/home/zhanzhehui_min/Engineering_project/ECG_Panorama/data/Nijmegen-2004-12-09/Meshes/model_65lead.mat' ##物理模型
# electrode_65leads = '/home/zhanzhehui_min/Engineering_project/ECG_Panorama/data/Nijmegen-2004-12-09/Meshes/Electrode_LOC_65L.mat'
#
# data_paths = Meg_paths + Noisyroom_paths
# data = loadmat(Meg_paths[0])
# bspm0 = data['pots']
#
# model_65lead = loadmat(bspm_65leads)
# model_65lead_data = model_65lead['geom']
# position = model_65lead_data[0][0][0]
#
# # class bspm(Dataset):
#
# import numpy as np
#
# data = []
# with open('/home/zhanzhehui_min/Engineering_project/ECG_Panorama/data/Nijmegen-2004-12-09/BSM_(nijmegen_64).txt','r') as file:
#     i = 0
#     for line in file:
#         i = i + 1
#         if i > 1:
#             line = line.strip().split('\t')
#             data.append([float(line[0]), float(line[1]), float(line[2]), float(line[3])])
#     data2 = np.asarray(data)
#
# ##网格绘制
# # 提取 x, y, z 坐标
# x = mesh_data[:, 0]
# y = mesh_data[:, 1]
# z = mesh_data[:, 2]
# # 创建三维图形
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # 绘制点云
# ax.scatter(x, y, z, c='b', marker='o')
# # 设置标签
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# # 显示图形
# plt.show()