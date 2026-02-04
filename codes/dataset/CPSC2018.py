import numpy as np
from torch.utils.data import Dataset
import glob
import matplotlib.pyplot as plt
import scipy.io
import random
import neurokit2 as nk
import warnings
from config import cfg
import torch
import json
import os

class CPSC2018(Dataset):
    r"""
    CPSC2018:
    """
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
                               [np.pi * (19/36), np.pi / 12],  # v3
                               [np.pi * (11/20), np.pi / 6],  # v4
                               [np.pi * (16/30), np.pi / 3],  # v5
                               [np.pi * (16/30), np.pi / 2],  # v6
                               [np.pi * (5/6), -np.pi / 2],  # III
                               [np.pi * (1/3), -np.pi / 2],  # aVR
                               [np.pi * (1/3), np.pi / 2],  # aVL
                               [np.pi * 1, np.pi / 2],  # aVF
                               ])
        self._read_data(cfg, phase)

    def _read_data(self, cfg, phase):
        if phase == 'train':
            path1 = glob.glob(cfg.DATA.CPSC_path1 + '/*.mat')
            path1.remove(cfg.DATA.CPSC_path1 + '/label.mat')
            path2 = glob.glob(cfg.DATA.CPSC_path2 + '/*.mat')
            self.dataset = path1 + path2
        else:
            path3 = glob.glob(cfg.DATA.CPSC_path3 + '/*.mat')
            self.dataset = path3

    def angle_jitter(self, angle, jitter_factor):
        jitter_angle = jitter_factor / 180 * np.pi
        jitter = np.random.normal(scale=jitter_angle, size=angle.shape)
        angle = angle + jitter
        return angle

    def __getitem__(self, index):
        file_path = self.dataset[index]
        new_order = [0, 1, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5]
        mat = scipy.io.loadmat(file_path)
        ECG = mat['ECG'][0][0][2]
        ECG = ECG[new_order]
        Lenght = len(ECG[0,:])
        # random crop
        if Lenght>4608:
            random_index_on = random.sample(range(0, Lenght - 4608), k=1)[0]
            random_index_off = random.sample(range(random_index_on + 2048 ,4608 + random_index_on), k=1)[0]
        else:
            random_index_on = 0
            random_index_off = random.sample(range(2048, Lenght), k=1)[0]
        data = ECG[:, random_index_on:random_index_off]
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
            elif self.cfg.DATA.lead_num == 1:
                input_index = [3]
            elif self.cfg.DATA.lead_num == 2:
                input_index = [0, 3]
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
                'data':data.astype(np.float32),
            }
        return meta
    def __len__(self):
        return len(self.dataset)

class CPSC2018_finetune(Dataset):
    r"""
    """
    def __init__(self, cfg,transform=None):
        self.dataset = None
        self.label_name = None
        self.label_dir = None
        self.cfg = cfg
        self.transform = transform
        self.theta = np.array([[np.pi / 2, np.pi / 2],  # I
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
        self._read_data(cfg)

    def _read_data(self, cfg):
        path3 =  glob.glob(cfg.DATA.CPSC_path3 + '/*.mat')
        path3.remove('/home/ALL_Lab_Data/zzh/CPSC2018/TrainingSet3/A5277.mat')
        path3.remove('/home/ALL_Lab_Data/zzh/CPSC2018/TrainingSet3/A5524.mat')
        self.dataset = path3

    def angle_jitter(self, angle, jitter_factor):
        jitter_angle = jitter_factor / 180 * np.pi
        jitter = np.random.normal(scale=jitter_angle, size=angle.shape)
        angle = angle + jitter

        return angle

    def __getitem__(self, index):
        file_path = self.dataset[index]
        new_order = [0, 1, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5]
        mat = scipy.io.loadmat(file_path)
        ECG = mat['ECG'][0][0][2]
        ECG = ECG[new_order]
        data = ECG
        if self.cfg.DATA.Normalize == True:
            max_, min_ = np.max(data), np.min(data)
            source_data = (data - min_) / (max_ - min_)

        end_point = 4000
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

def path_classify(label,paths):
    #九类带标签数,path_9为{list:6875}，每个列表为一条数据（label和data)
    path_9 = []
    for i in range(6875):
        # b = label[i]
        # one_hot = [1 if j == b - 1 else 0 for j in range(9)]
        # path_9[b - 1].append([paths[i], one_hot])
        path_9.append([paths[i],label[i]])
    return path_9

def get_label():
    path_d = glob.glob(r'/home/ALL_Lab_Data/zzh/CPSC2018/TrainingSet1/label.mat')
    file_d = scipy.io.loadmat(path_d[0])
    a = file_d['xo']
    label = []
    for i in range(6877):
        if i != 3 and i != 2236:
            label.append(a[0][i])
    return label

def label_data():
    paths = glob.glob(r'/home/ALL_Lab_Data/zzh/CPSC2018/*/*.mat')
    paths.remove('/home/ALL_Lab_Data/zzh/CPSC2018/TrainingSet1/label.mat')
    paths = sorted(paths, key=lambda i: int(i[-8:-4]))
    label = get_label()
    path_9 = path_classify(label, paths)

class CPSC2018_c(Dataset):
    """
    CPSC2018:
    """
    def __init__(self, cfg):
        self.dataset = None
        self.label_name = None
        self.label_dir = None
        self.cfg = cfg
        self.theta = np.array([[np.pi / 2, np.pi / 2],  # I
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
        self._read_data()

    def _read_data(self):
        paths = glob.glob(r'/home/ALL_Lab_Data/zzh/CPSC2018/*/*.mat')
        paths.remove('/home/ALL_Lab_Data/zzh/CPSC2018/TrainingSet1/label.mat')
        paths = sorted(paths, key=lambda i: int(i[-8:-4]))
        label = get_label()
        path_9 = path_classify(label, paths)[4468:]
        self.dataset = []
        for path in path_9:
            if path[1] == self.cfg.DATA.CPSC_disease:
                self.dataset.append(path)

    def angle_jitter(self, angle, jitter_factor):
        jitter_angle = jitter_factor / 180 * np.pi
        jitter = np.random.normal(scale=jitter_angle, size=angle.shape)
        angle = angle + jitter
        return angle

    def __getitem__(self, index):
        file_path = self.dataset[index][0]
        new_order = [0, 1, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5]

        mat = scipy.io.loadmat(file_path)
        ECG = mat['ECG'][0][0][2]
        ECG = ECG[new_order]
        Lenght = len(ECG[0,:])
        # random crop
        # random_index_on = 0
        # even_range = range(2048, Lenght, 2)
        # random_index_off = random.choice(even_range)
        if Lenght>4608:
            random_index_on = 0
            # random_index_on = random.sample(range(0, Lenght - 4608), k=1)[0]
            random_index_off = random.sample(range(random_index_on + 2048 ,4608 + random_index_on), k=1)[0]
        else:
            random_index_on = 0
            even_range = range(2048, Lenght, 2)
            random_index_off  = random.choice(even_range)

        data = ECG[:, random_index_on:random_index_off]

        # normalized
        if self.cfg.DATA.Normalize == True:
            max_, min_ = np.max(data), np.min(data)
            data = (data - min_) / (max_ - min_)

        padding_len = 4608 - (random_index_off - random_index_on)
        data = np.pad(data, ((0, 0), (0, padding_len)), mode='constant', constant_values=0)

        # get noise
        if self.cfg.DATA.noise == True:
            noise_region = data[:, :random_index_off - random_index_on]
            noise_std = np.std(noise_region, axis=1)
            noise = np.random.normal(loc=0, scale=noise_std, size=(data.shape[-1], 12))
        else:
            noise = np.random.randn(4608, 12)

        # angle jitter
        theta_ = self.theta
        synthesis_index = random.sample(self.cfg.DATA.data_synthesis,1)[0]
        synthesis_view = data[synthesis_index]
        synthesis_theta = theta_[synthesis_index]

        if self.cfg.DATA.lead_num == 3:
            input_index = [0, 1, 4]
        elif self.cfg.DATA.lead_num == 1:
            input_index = [3]
        elif self.cfg.DATA.lead_num == 2:
            input_index = [0, 3]
        elif self.cfg.DATA.lead_num == 4:
            input_index = [0, 1, 2, 4]
        elif self.cfg.DATA.lead_num == 5:
            input_index = [0, 1, 2, 3, 4]
        rest_index = [x for x in range(2, 8) if x not in input_index + self.cfg.DATA.data_synthesis]
        target_index = random.sample(rest_index, 1)[0]
        meta = {
                'input_index': input_index,
                'input_data': data[input_index].astype(np.float32),
                'data':data.astype(np.float32),
                'input_theta': np.array(self.theta[input_index]).astype(np.float32),
                'target_view': np.array(data[target_index]).astype(np.float32),
                'target_theta': np.array(self.theta[target_index]).astype(np.float32),
                'id': self.dataset[index],
                'synthesis_view': np.array(synthesis_view).astype(np.float32),
                'synthesis_theta': np.array(synthesis_theta).astype(np.float32),
                'end_point': random_index_off - random_index_on,
                'target_index': target_index,
                'noise': noise[:, target_index],
            }
        return meta


    def __len__(self):
        return len(self.dataset)

# path1 = glob.glob(cfg.DATA.CPSC_path1 + '/*.mat')
# path1.remove(cfg.DATA.CPSC_path1 + '/label.mat')
# path2 = glob.glob(cfg.DATA.CPSC_path2 + '/*.mat')
# dataset = path1 + path2
# file_path = dataset[12]
# mat = scipy.io.loadmat(file_path)
# ECG = mat['ECG'][0][0][2]
# ECG = ECG[new_order]
# list = ["medical","minimal","blue"]
# style = list[1]
# signal = ECG[0,500:1000]
# ecg_plt(signal,style=style)
# signal = ECG[2,500:1000]
# ecg_plt(signal,style=style)
# signal = ECG[3,500:1000]
# ecg_plt(signal,style=style)
# signal = ECG[4,500:1000]
# ecg_plt(signal,style=style)
# signal = ECG[7,500:1000]
# ecg_plt(signal,style=style)
def ecg_plt(signal, fs=500, style="medical"):
    t = np.arange(len(signal)) / fs
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    # 公共：不显示坐标轴刻度
    ax.set_xticks([]);
    ax.set_yticks([])
    if style in ("medical"):
        # ECG 纸格
        ax.set_facecolor("white")
        major_grid = 0.2  # 大格：0.2 s 一条
        minor_grid = 0.04  # 小格：0.04 s 一条
        ypad = max(1e-6, 0.2 * max(abs(signal.max()), abs(signal.min())))
        ylim = (min(signal) * 1.6, max(signal) * 1.6)
        ax.set_xlim(0, t[-1])
        ax.set_ylim(signal.min() - ypad, signal.max() + ypad)
        if style == "medical":
            minor_c, minor_w = "#ffcccc", 0.8
            major_c, major_w = "#ff6666", 1.6
        for x in np.arange(0, t[-1] + minor_grid, minor_grid):
            ax.axvline(x, color=minor_c, linewidth=minor_w, zorder=0)
        for y in np.arange(ylim[0], ylim[1], 0.1):
            ax.axhline(y, color=minor_c, linewidth=minor_w, zorder=0)
        # 大格
        for x in np.arange(0, t[-1] + major_grid, major_grid):
            ax.axvline(x, color=major_c, linewidth=major_w, zorder=0)
        for y in np.arange(ylim[0], ylim[1], 0.5):
            ax.axhline(y, color=major_c, linewidth=major_w, zorder=0)
        # 波形与外框
        ax.plot(t, signal, color="black", linewidth=5, zorder=1)
        for s in ["top", "right", "bottom", "left"]:
            ax.spines[s].set_visible(False)
    elif style == "minimal":
        # 极简：细黑线 + 很淡的灰框 + 更多留白
        ax.plot(t, signal, color="black", linewidth=5)
        for s in ["top", "right", "bottom", "left"]:
            ax.spines[s].set_visible(True)
            ax.spines[s].set_color("lightgray")
            ax.spines[s].set_linewidth(2)
        ax.margins(x=0.02, y=0.25)
    elif style == "blue":
        # 黑框 + 蓝线 + 少量虚线内部网格（淡灰）
        ax.set_facecolor("white")
        ypad = max(1e-6, 0.2 * max(abs(signal.max()), abs(signal.min())))
        ax.set_ylim(signal.min() - ypad, signal.max() + ypad)
        for x in np.linspace(0, t[-1], 6):
            ax.axvline(x, color="0.85", linestyle="--", linewidth=0.5, zorder=0)
        for y in np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 6):
            ax.axhline(y, color="0.85", linestyle="--", linewidth=0.5, zorder=0)
        ax.plot(t, signal, color="blue", linewidth=5.0, zorder=1)
        for s in ["top", "right", "bottom", "left"]:
            ax.spines[s].set_visible(True)
            ax.spines[s].set_color("black")
            ax.spines[s].set_linewidth(2.0)
    # 标题（可选）
    ax.set_title("", fontsize=8)
    plt.tight_layout(pad=0.1)
    plt.show()