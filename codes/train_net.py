from dataset import build_dataset, Panobench, EcgTianChi_finetune, ChinaDB_finetune, CPSC2018_finetune, PTBVXL_finetune,CPSC2018_c
from torch.utils.data import DataLoader, Subset
from solver import Solver2
from utils import seed_torch
import os
import torch
import argparse
from config import cfg
import setproctitle
import numpy as np
from solver.optim_scheduler import get_optimizer, get_lr_scheduler
import matplotlib.pyplot as plt
from utils import CheckPointer
import loralib as lora
import dill
import random,copy
import torch.nn as nn
from tqdm import tqdm
from utils.mertic import SSIM, PSNR
import yaml
from collections import OrderedDict
import torch.nn.functional as F
from torch.utils.data.sampler import WeightedRandomSampler
from main import process_numbers_console

def main(cfg):
    parser = argparse.ArgumentParser(description='ecg generation')
    parser.add_argument(
        '--config-file',
        default="",
        metavar="FILE",
        help="path to config file",
        type=str
    )
    parser.add_argument('--a', default="", metavar="FILE", help="path to config file", type=str)
    args = parser.parse_args()
    print('Using config: ', cfg)
    if args.config_file != '':
        cfg.merge_from_file(args.config_file)
        print('config_file:.{}'.format(args.config_file))
    else:
        print('config_file:default')
    cfg.desc = args.config_file.split('/')[-1].replace('.yml', '')
    cfg.output_dir = os.path.join(cfg.output_dir, cfg.desc)
    setproctitle.setproctitle(cfg.desc)
    torch.multiprocessing.set_sharing_strategy('file_system')
    seed_torch(seed=cfg.seed)
    output_dir = os.path.join(cfg.output_dir, cfg.desc)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    solver2 = Solver2(cfg)
    train_dataset = build_dataset(cfg, phase='train')
    test_dataset = build_dataset(cfg, phase='test')
    dl_train = DataLoader(train_dataset, batch_size=cfg.DATA.batchsize, shuffle=True, num_workers=16, drop_last=True)
    dl_test = DataLoader(test_dataset, batch_size=cfg.DATA.batchsize, num_workers=8, drop_last=True)
    solver2.train(dl_train, dl_test)
    path = cfg.output_dir + 'best_valid.pkl'
    cfg.DATA.super_mode = "optimization"
    cfg.DATA.dataset = "CPSC2018"
    train_dataset = build_dataset(cfg, phase='train')
    test_dataset = build_dataset(cfg, phase='test')
    dl_train = DataLoader(train_dataset, batch_size=cfg.DATA.batchsize, shuffle=True, num_workers=16, drop_last=True)
    dl_test = DataLoader(test_dataset, batch_size=cfg.DATA.batchsize, num_workers=8, drop_last=True)
    solver2.optimization(dl_train, dl_test, path=path)

    Finetune_Dataset = CPSC2018_finetune(cfg)
    dl_finetune = DataLoader(Finetune_Dataset, batch_size=1, num_workers=8, drop_last=True)
    solver2.finetune(dl_finetune,dl_test,path=path)

def pro_ecg(data):
    max_, min_ = np.max(data), np.min(data)
    data = (data - min_) / (max_ - min_)
    return data

def application():
    import scipy
    from scipy.signal import resample
    self = Solver2(cfg)

    self.model.load_state_dict(torch.load('model_weights.pth'))
    path = '/home/zhanzhehui_min/Engineering_project/ECG_Panorama/codes/1.mat'
    mat = scipy.io.loadmat(path)
    ecg = mat['Panobench']
    ecg = resample(ecg, 5000, axis=1)
    ecg = pro_ecg(ecg)

    input_ecg = ecg[[0, 1, 29], :]
    input_ecg = torch.from_numpy(input_ecg.astype(np.float32)).unsqueeze(0)
    input_theta = torch.from_numpy(np.array([[np.pi / 2, np.pi / 2], [np.pi * 5 / 6, np.pi / 2], [np.pi * (30 / 180), np.pi * (69 / 180)]]).astype(
            np.float32)).unsqueeze(0)

    out_theta = torch.from_numpy(process_numbers_console())
    # out_theta = torch.from_numpy(np.array([[np.pi * (30 / 180), np.pi * (69 / 180)]]).astype(np.float32))


    out = self.model(input_ecg[:, :, :4608], input_theta, out_theta)
    plt.plot(out[0, 0, 1500:4000].detach().cpu())
    plt.title('ECG from view {}'.format(out_theta* 180/np.pi))
    plt.show()

def finetune():
    import scipy
    from scipy.signal import resample
    import random
    self = Solver2(cfg)

    self.load(path='/home/zhanzhehui_min/Engineering_project/ECG_Panorama/codes/layer_arch_check/layer_input03_4optimization_dataset_sizePanobenchfixPanobench_input3/best_valid.pkl')
    path = '/home/zhanzhehui_min/Engineering_project/ECG_Panorama/codes/1.mat'
    mat = scipy.io.loadmat(path)
    ecg = mat['Panobench']
    ecg = resample(ecg, 5000, axis=1)
    ecg = pro_ecg(ecg)
    ecg = torch.from_numpy(ecg.astype(np.float32)).unsqueeze(0)
    input_ecg = ecg[:,[0, 1, 29], :]
    input_theta = torch.from_numpy(np.array([[np.pi / 2, np.pi / 2], [np.pi * 5 / 6, np.pi / 2], [np.pi * (30 / 180), np.pi * (69 / 180)]]).astype(
            np.float32)).unsqueeze(0)
    q_theta = torch.from_numpy(np.array([[np.pi * (30 / 180), np.pi * (69 / 180)]]).astype(np.float32))

    batch_size = 32
    thetas = np.array([[np.pi / 2, np.pi / 2], [np.pi * 5 / 6, np.pi / 2],  # I ,II
                       [np.pi * (106 / 180), -np.pi * (102 / 180)], [np.pi * (121 / 180), np.pi * (-101 / 180)],  # 1,2
                       [np.pi * (132 / 180), np.pi * (-99 / 180)], [np.pi * (52 / 180), np.pi * (-83 / 180)],  # 3,4
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
                       [np.pi * (129 / 180), np.pi * (103 / 180)], [np.pi * (140 / 180), np.pi * (100 / 180)]  # 41,42
                       ])
    thetas = torch.from_numpy(thetas.astype(np.float32))
    Index = [x for x in range(0, 44)]

    for name, param in self.model.named_parameters():
        param.requires_grad = True
    optimizer = get_optimizer(self.cfg, filter(lambda p: p.requires_grad, self.model.parameters()))
    scheduler = get_lr_scheduler(self.cfg, optimizer)
    print(f"可训练参数数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
    for i in range(200):
        target_indice = [random.sample(Index, 1)[0] for _ in range(batch_size)]
        input_thetas = input_theta.repeat(batch_size, 1, 1).to(self.device)
        supervision_thetas = thetas[target_indice, :].to(self.device)
        supervision_output = ecg[:, target_indice, :].permute(1, 0, 2).to(self.device)
        input = []
        target = []
        for j in range(batch_size):
            StartPoint = random.sample(range(0, 5000 - 4608), k=1)[0]
            input.append(input_ecg[:, :, StartPoint:StartPoint + 4608])
            target.append(supervision_output[j, :, StartPoint:StartPoint + 4608])
        target = torch.cat(target, dim=0).unsqueeze(1)
        input = torch.cat(input, dim=0)

        result = self.model(x=input, input_thetas=input_thetas, query_theta=supervision_thetas, phase='finetune')
        loss = self.loss(result, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

def train_optiomization(cfg):
    path = cfg.output_dir + 'best_valid.pkl'
    train_dataset = build_dataset(cfg, phase='train')
    test_dataset = build_dataset(cfg, phase='test')
    dl_train = DataLoader(train_dataset, batch_size=cfg.DATA.batchsize, shuffle=True, num_workers=16, drop_last=True)
    dl_test = DataLoader(test_dataset, batch_size=cfg.DATA.batchsize, num_workers=8, drop_last=True)
    solver2 = Solver2(cfg)
    solver2.train(dl_train, dl_test)

    cfg.DATA.super_mode = "optimization"
    train_all = build_dataset(cfg, phase='train')
    test_all = build_dataset(cfg, phase='test')
    all_train = DataLoader(train_all, batch_size=cfg.DATA.batchsize, shuffle=True, num_workers=16, drop_last=True)
    all_test = DataLoader(test_all, batch_size=cfg.DATA.batchsize, num_workers=8, drop_last=True)
    cfg.SOLVER.epochs = 50
    solver2 = Solver2(cfg)
    solver2.optimization(all_train, all_test, path=path)
    path = cfg.output_dir + 'fix' + cfg.DATA.dataset + '_input' + str(cfg.DATA.lead_num) + '/best_valid.pkl'

    cfg.DATA.dataset = "Panobench"
    train_dataset = build_dataset(cfg, phase='train')
    test_dataset = build_dataset(cfg, phase='test')
    dl_train = DataLoader(train_dataset, batch_size=cfg.DATA.batchsize, shuffle=True, num_workers=16, drop_last=True)
    dl_test = DataLoader(test_dataset, batch_size=cfg.DATA.batchsize, num_workers=8, drop_last=True)
    cfg.SOLVER.epochs = 200
    solver2.optimization(dl_train, dl_test, path=path)

    cfg.DATA.dataset = "CPSC2018"
    train_dataset = build_dataset(cfg, phase='train')
    test_dataset = build_dataset(cfg, phase='test')
    dl_train = DataLoader(train_dataset, batch_size=cfg.DATA.batchsize, shuffle=True, num_workers=16, drop_last=True)
    dl_test = DataLoader(test_dataset, batch_size=cfg.DATA.batchsize, num_workers=8, drop_last=True)
    solver2.optimization(dl_train, dl_test,path=path)

    cfg.DATA.dataset = "Tianchi"
    train_dataset = build_dataset(cfg, phase='train')
    test_dataset = build_dataset(cfg, phase='test')
    dl_train = DataLoader(train_dataset, batch_size=cfg.DATA.batchsize, shuffle=True, num_workers=16, drop_last=True)
    dl_test = DataLoader(test_dataset, batch_size=cfg.DATA.batchsize, num_workers=8, drop_last=True)
    solver2.optimization(dl_train, dl_test,path=path)

    cfg.DATA.dataset = "PTBXL"
    train_dataset = build_dataset(cfg, phase='train')
    test_dataset = build_dataset(cfg, phase='test')
    dl_train = DataLoader(train_dataset, batch_size=cfg.DATA.batchsize, shuffle=True, num_workers=16, drop_last=True)
    dl_test = DataLoader(test_dataset, batch_size=cfg.DATA.batchsize, num_workers=8, drop_last=True)
    solver2.optimization(dl_train, dl_test,path=path)

    cfg.DATA.dataset = "ChinaDB"
    train_dataset = build_dataset(cfg, phase='train')
    test_dataset = build_dataset(cfg, phase='test')
    dl_train = DataLoader(train_dataset, batch_size=cfg.DATA.batchsize, shuffle=True, num_workers=16, drop_last=True)
    dl_test = DataLoader(test_dataset, batch_size=cfg.DATA.batchsize, num_workers=8, drop_last=True)
    solver2.optimization(dl_train, dl_test,path=path)

def Panobench_test(cfg):
    main(cfg)
    path = cfg.output_dir + 'best_valid.pkl'
    cfg.DATA.super_mode = "optimization"
    train_all = build_dataset(cfg, phase='train')
    test_all = build_dataset(cfg, phase='test')
    all_train = DataLoader(train_all, batch_size=cfg.DATA.batchsize, shuffle=True, num_workers=16, drop_last=True)
    all_test = DataLoader(test_all, batch_size=cfg.DATA.batchsize, num_workers=8, drop_last=True)
    cfg.SOLVER.epochs = 50
    solver2 = Solver2(cfg)
    solver2.optimization(all_train, all_test, path=path)
    path = cfg.output_dir + 'fix' + cfg.DATA.dataset + '_input' + str(cfg.DATA.lead_num) + '/best_valid.pkl'

    cfg.DATA.dataset = "Panobench"
    cfg.DATA.Panobench_supervision = [3, 15, 42]
    cfg.DATA.lead_num = 3
    train_dataset = build_dataset(cfg, phase='train')
    test_dataset = build_dataset(cfg, phase='test')
    dl_train = DataLoader(train_dataset, batch_size=cfg.DATA.batchsize, shuffle=True, num_workers=16, drop_last=True)
    dl_test = DataLoader(test_dataset, batch_size=cfg.DATA.batchsize, num_workers=8, drop_last=True)
    cfg.SOLVER.epochs = 200
    solver2.optimization(dl_train, dl_test, path=path)

    cfg.DATA.Panobench_supervision = [3, 10, 15, 24, 31, 42]
    cfg.DATA.lead_num = 3
    train_dataset = build_dataset(cfg, phase='train')
    test_dataset = build_dataset(cfg, phase='test')
    dl_train = DataLoader(train_dataset, batch_size=cfg.DATA.batchsize, shuffle=True, num_workers=16, drop_last=True)
    dl_test = DataLoader(test_dataset, batch_size=cfg.DATA.batchsize, num_workers=8, drop_last=True)
    cfg.SOLVER.epochs = 200
    solver2.optimization(dl_train, dl_test, path=path)

    cfg.DATA.Panobench_supervision = [3, 6, 10, 15, 24, 27, 31, 36, 42]
    cfg.DATA.lead_num = 3
    train_dataset = build_dataset(cfg, phase='train')
    test_dataset = build_dataset(cfg, phase='test')
    dl_train = DataLoader(train_dataset, batch_size=cfg.DATA.batchsize, shuffle=True, num_workers=16, drop_last=True)
    dl_test = DataLoader(test_dataset, batch_size=cfg.DATA.batchsize, num_workers=8, drop_last=True)
    cfg.SOLVER.epochs = 200
    solver2.optimization(dl_train, dl_test, path=path)

    cfg.DATA.Panobench_supervision = [3, 6, 8, 10, 15, 18, 24, 27, 31, 34, 36, 42]
    cfg.DATA.lead_num = 3
    train_dataset = build_dataset(cfg, phase='train')
    test_dataset = build_dataset(cfg, phase='test')
    dl_train = DataLoader(train_dataset, batch_size=cfg.DATA.batchsize, shuffle=True, num_workers=16, drop_last=True)
    dl_test = DataLoader(test_dataset, batch_size=cfg.DATA.batchsize, num_workers=8, drop_last=True)
    cfg.SOLVER.epochs = 200
    solver2.optimization(dl_train, dl_test, path=path)

    cfg.DATA.Panobench_supervision = [3, 15, 42]
    cfg.DATA.lead_num = 5
    train_dataset = build_dataset(cfg, phase='train')
    test_dataset = build_dataset(cfg, phase='test')
    dl_train = DataLoader(train_dataset, batch_size=cfg.DATA.batchsize, shuffle=True, num_workers=16, drop_last=True)
    dl_test = DataLoader(test_dataset, batch_size=cfg.DATA.batchsize, num_workers=8, drop_last=True)
    cfg.SOLVER.epochs = 200
    solver2.optimization(dl_train, dl_test, path=path)

    cfg.DATA.Panobench_supervision = [3, 10, 15, 24, 31, 42]
    cfg.DATA.lead_num = 5
    train_dataset = build_dataset(cfg, phase='train')
    test_dataset = build_dataset(cfg, phase='test')
    dl_train = DataLoader(train_dataset, batch_size=cfg.DATA.batchsize, shuffle=True, num_workers=16, drop_last=True)
    dl_test = DataLoader(test_dataset, batch_size=cfg.DATA.batchsize, num_workers=8, drop_last=True)
    cfg.SOLVER.epochs = 200
    solver2.optimization(dl_train, dl_test, path=path)

    cfg.DATA.Panobench_supervision = [3, 6, 10, 15, 24, 27, 31, 36, 42]
    cfg.DATA.lead_num = 5
    train_dataset = build_dataset(cfg, phase='train')
    test_dataset = build_dataset(cfg, phase='test')
    dl_train = DataLoader(train_dataset, batch_size=cfg.DATA.batchsize, shuffle=True, num_workers=16, drop_last=True)
    dl_test = DataLoader(test_dataset, batch_size=cfg.DATA.batchsize, num_workers=8, drop_last=True)
    cfg.SOLVER.epochs = 200
    solver2.optimization(dl_train, dl_test, path=path)

    cfg.DATA.Panobench_supervision = [3, 6, 8, 10, 15, 18, 24, 27, 31, 34, 36, 42]
    cfg.DATA.lead_num = 5
    train_dataset = build_dataset(cfg, phase='train')
    test_dataset = build_dataset(cfg, phase='test')
    dl_train = DataLoader(train_dataset, batch_size=cfg.DATA.batchsize, shuffle=True, num_workers=16, drop_last=True)
    dl_test = DataLoader(test_dataset, batch_size=cfg.DATA.batchsize, num_workers=8, drop_last=True)
    cfg.SOLVER.epochs = 200
    solver2.optimization(dl_train, dl_test, path=path)

    cfg.DATA.Panobench_supervision = [3, 15, 42]
    cfg.DATA.lead_num = 7
    train_dataset = build_dataset(cfg, phase='train')
    test_dataset = build_dataset(cfg, phase='test')
    dl_train = DataLoader(train_dataset, batch_size=cfg.DATA.batchsize, shuffle=True, num_workers=16, drop_last=True)
    dl_test = DataLoader(test_dataset, batch_size=cfg.DATA.batchsize, num_workers=8, drop_last=True)
    cfg.SOLVER.epochs = 200
    solver2.optimization(dl_train, dl_test, path=path)

    cfg.DATA.Panobench_supervision = [3, 10, 15, 24, 31, 42]
    cfg.DATA.lead_num = 7
    train_dataset = build_dataset(cfg, phase='train')
    test_dataset = build_dataset(cfg, phase='test')
    dl_train = DataLoader(train_dataset, batch_size=cfg.DATA.batchsize, shuffle=True, num_workers=16, drop_last=True)
    dl_test = DataLoader(test_dataset, batch_size=cfg.DATA.batchsize, num_workers=8, drop_last=True)
    cfg.SOLVER.epochs = 200
    solver2.optimization(dl_train, dl_test, path=path)

    cfg.DATA.Panobench_supervision = [3, 6, 10, 15, 24, 27, 31, 36, 42]
    cfg.DATA.lead_num = 7
    train_dataset = build_dataset(cfg, phase='train')
    test_dataset = build_dataset(cfg, phase='test')
    dl_train = DataLoader(train_dataset, batch_size=cfg.DATA.batchsize, shuffle=True, num_workers=16, drop_last=True)
    dl_test = DataLoader(test_dataset, batch_size=cfg.DATA.batchsize, num_workers=8, drop_last=True)
    cfg.SOLVER.epochs = 200
    solver2.optimization(dl_train, dl_test, path=path)

    cfg.DATA.Panobench_supervision = [3, 6, 8, 10, 15, 18, 24, 27, 31, 34, 36, 42]
    cfg.DATA.lead_num = 7
    train_dataset = build_dataset(cfg, phase='train')
    test_dataset = build_dataset(cfg, phase='test')
    dl_train = DataLoader(train_dataset, batch_size=cfg.DATA.batchsize, shuffle=True, num_workers=16, drop_last=True)
    dl_test = DataLoader(test_dataset, batch_size=cfg.DATA.batchsize, num_workers=8, drop_last=True)
    cfg.SOLVER.epochs = 200
    solver2.optimization(dl_train, dl_test, path=path)

def split_dataset(p,dataset):
    from torch.utils.data import random_split
    dataset_size = len(dataset)
    subset_size = int(p * dataset_size)  # 10% 数据
    remaining_size = dataset_size - subset_size
    subset, _ = random_split(dataset, [subset_size, remaining_size])
    return subset

def CPSC2018_Disease():
    self = Solver2(cfg)
    cfg.DATA.CPSC_disease = 1
    cpsc2018_C = CPSC2018_c(cfg)
    dl_cpsc2018 = DataLoader(cpsc2018_C, batch_size=32, shuffle=True, num_workers=16, drop_last=True)
    test_losses, test_gt_views, test_predict_views, test_input_views, mertics_all, mertics_synthesis = self.optimize_one_epoch(
        dl_cpsc2018, phase='test')
    psnr_gen, ssim_gen = np.mean(mertics_all, axis=0)[0], np.mean(mertics_all, axis=0)[1]
    psnr_gen2, ssim_gen2 = np.mean(mertics_synthesis, axis=0)[0], np.mean(mertics_synthesis, axis=0)[1]
    print('psnr_gen: {},  ssim_gen:{}'.format(psnr_gen, ssim_gen))
    print('psnr_gen2: {},  ssim_gen2:{}'.format(psnr_gen2, ssim_gen2))


if __name__ == '__main__':
    # solver2 = Solver2(cfg)
    # main(cfg)

    cpsc2018_C = CPSC2018_c(cfg)
    dl_cpsc2018 = DataLoader(cpsc2018_C, batch_size=32, shuffle=True, num_workers=16, drop_last=True)
    cpsc2018_finetune = CPSC2018_finetune(cfg)
    dl_finetune = DataLoader(cpsc2018_finetune, batch_size=1, num_workers=8, drop_last=True)

    # train_dataset = build_dataset(cfg, phase='train')
    # test_dataset = build_dataset(cfg, phase='test')
    # p = 0.1
    # train_dataset = split_dataset(p, train_dataset)
    # dl_train = DataLoader(train_dataset, batch_size=cfg.DATA.batchsize, shuffle=True, num_workers=16, drop_last=True)
    # dl_test = DataLoader(test_dataset, batch_size=cfg.DATA.batchsize, num_workers=8, drop_last=True)
