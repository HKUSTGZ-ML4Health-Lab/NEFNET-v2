from yacs.config import CfgNode as Node
import os
import random

cfg = Node()
cfg.seed = 123
cfg.fit_msg = 'None'

cfg.desc = 'none' # inference_with_finetune
cfg.device = '0,1' # 0,1  2,3


# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
cfg.MODEL = Node()
cfg.MODEL.model = 'layer'   #layer,nefnet
cfg.MODEL.layers = 4
cfg.MODEL.resume = ''
cfg.MODEL.loss = 'l1'
cfg.MODEL.theta_L = 1
cfg.MODEL.token_sizes = 16
cfg.MODEL.change = False
# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------
cfg.DATA = Node()
cfg.DATA.dataset = 'PTBXL' # Tianchi CPSC2018 ChinaDB PTBXL Panobench all
cfg.DATA.twopaths = False
cfg.DATA.super_mode = "optimization" # pretrain, optimization, transformation
cfg.DATA.data_mode = "3_12"
cfg.DATA.CPSC_disease = 1

cfg.DATA.noise_std = [4.37258895, 4.73799667, 5.00643047, 6.7582663, 6.57354042, 6.31023917, 6.05944371, 7.05612394]
cfg.DATA.data_synthesis = [3,7] #5 ,3
# cfg.DATA.input_index = [0,1,6]
cfg.DATA.lead_num = 3

cfg.DATA.noise = True
cfg.DATA.Normalize = True
cfg.DATA.batchsize = 32

cfg.output_dir = 'layer_arch_check/'+ cfg.MODEL.model +'_input_5' + str(cfg.MODEL.layers) + cfg.DATA.super_mode + '_dataset_size' + cfg.DATA.dataset

cfg.DATA.limb = [0, 1, 8, 9, 10, 11] # [0, 1, 8, 9, 10, 11]
cfg.DATA.breast = [2, 3, 4, 5, 6, 7] # [2, 3, 4, 5, 6, 7]
cfg.DATA.bspm_supervision = [3, 6, 10, 15, 24, 27, 31, 36, 42] #  [3,15,42] [3, 10, 15, 24, 31, 42]
# [3, 6, 10, 15, 24, 27, 31, 36, 42]  [3, 6, 8, 10, 15, 18, 24, 27, 31, 34, 36, 42]
# [3, 6, 8, 10, 13, 15, 18, 22, 24, 27, 31, 34, 36, 39, 42]
# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
cfg.SOLVER = Node()
cfg.SOLVER.optim = 'adamw' # sgd adam adamw
cfg.SOLVER.scheduler = 'MultiStep'  # MultiStep steplr
cfg.SOLVER.lr_step = [50,100,150]
cfg.SOLVER.gamma = 0.5
cfg.SOLVER.lr = 1e-3
cfg.SOLVER.epochs = 200

cfg.SOLVER.reg_loss = cfg.MODEL.loss
cfg.SOLVER.loss_using = [1, 2, 3, 4]
cfg.SOLVER.part_loss_no_grad = False
cfg.SOLVER.loss_factor = [1, 1, 1, 1]

cfg.SOLVER.finetune_iter = 100
cfg.SOLVER.finetune_batch_size = 32

cfg.DATA.train_data_root = '/home/ALL_Lab_Data/zzh/Tianchi/train'
cfg.DATA.test_data_root = '/home/ALL_Lab_Data/zzh/Tianchi/testA'
# cfg.DATA.train_data_root = '/root/autodl-tmp/Tianchi/train'
# cfg.DATA.test_data_root = '/root/autodl-tmp/Tianchi/testA'
cfg.DATA.train_label_path = '/home/zhanzhehui_min/Engineering_project/ECG_Panorama/codes/data/tianchi/train.txt'
cfg.DATA.test_label_path = '/home/zhanzhehui_min/Engineering_project/ECG_Panorama/codes/data/tianchi/test.txt'

cfg.DATA.CPSC_path1 = '/home/ALL_Lab_Data/zzh/CPSC2018/TrainingSet1'
cfg.DATA.CPSC_path2 = '/home/ALL_Lab_Data/zzh/CPSC2018/TrainingSet2'
cfg.DATA.CPSC_path3 = '/home/ALL_Lab_Data/zzh/CPSC2018/TrainingSet3'
# cfg.DATA.CPSC_path1 = '/root/autodl-tmp/CPSC2018/TrainingSet1'
# cfg.DATA.CPSC_path2 = '/root/autodl-tmp/CPSC2018/TrainingSet2'
# cfg.DATA.CPSC_path3 = '/root/autodl-tmp/CPSC2018/TrainingSet3'

cfg.DATA.ptb_xl_root = '/home/ALL_Lab_Data/zzh/ptbxl_all'
cfg.DATA.ptbxl_train_label_path = '/home/zhanzhehui_min/Engineering_project/ECG_Panorama/codes/data/ptbxl/train.txt'
cfg.DATA.ptbxl_test_label_path = '/home/zhanzhehui_min/Engineering_project/ECG_Panorama/codes/data/ptbxl/test.txt'

cfg.DATA.ChinaDb_root = '/home/ALL_Lab_Data/zzh/China_DB/ECGData'
# cfg.DATA.ChinaDb_root = '/root/autodl-tmp/China_DB/ECGData'
cfg.DATA.ChinaDb_train_label_path = '/home/zhanzhehui_min/Engineering_project/ECG_Panorama/codes/data/chinadb/train.txt'
cfg.DATA.ChinaDb_test_label_path = '/home/zhanzhehui_min/Engineering_project/ECG_Panorama/codes/data/chinadb/test.txt'

cfg.DATA.CODE_path = '/home/ALL_Lab_Data/zzh/CODE/ecg_tracings.hdf5'
# cfg.DATA.CODE_path = '/root/autodl-tmp/CODE/ecg_tracings.hdf5'

cfg.DATA.incart_root = '/home/ALL_Lab_Data/zzh/incart/files'
cfg.DATA.incart_train_label_path = 'data/incart/train.txt'
cfg.DATA.incart_test_label_path = 'data/incart/test.txt'

cfg.DATA.ptb_root = '/home/ALL_Lab_Data/zzh/ptb_all'
# cfg.DATA.ptb_root = '/root/autodl-tmp/ptb_all'
cfg.DATA.ptb_train_label_path = '/home/zhanzhehui_min/Engineering_project/ECG_Panorama/codes/data/tianchi/data/ptb/train.txt'
cfg.DATA.ptb_test_label_path = '/home/zhanzhehui_min/Engineering_project/ECG_Panorama/codes/data/tianchi/data/ptb/test.txt'




# def optimization(self, dl_train, dl_test):
#     self.output_dir = self.cfg.output_dir + 'fix'
#     path = os.path.join('/home/zhanzhehui_min/Engineering_project/ECG_Panorama/codes', self.cfg.output_dir,
#                         'best_valid.pkl')
#     self.cfg.SOLVER.lr = 1e-3
#     output_dir = os.path.join(self.output_dir, self.desc)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     self.summary_writer = tensorboardX.SummaryWriter(logdir=os.path.join(self.output_dir, 'tf_logs'))
#     optimizer = get_optimizer(self.cfg, self.model.parameters())  ##sgd
#     scheduler = get_lr_scheduler(self.cfg, optimizer)  ##steplr
#     checkpointer = CheckPointer(self.model, optimizer, scheduler, self.output_dir)
#
#     last_checkpoint_data = checkpointer.load(self.cfg.MODEL.resume)
#     # best_checkpoint_data = checkpointer.load(self.cfg.MODEL.resume,best_valid=True)
#
#     checkpointer.load_nefnet2(path=path)
#     self.sync_parameters(self.model.module.W_encoder, self.model.module.W_encoder2)
#
#     max_epochs = self.cfg.SOLVER.epochs
#     start_epoch = last_checkpoint_data['epoch'] + 1 if 'epoch' in last_checkpoint_data.keys() else 0
#
#     best_test_psnr_gen2 = last_checkpoint_data[
#         'best_test_psnr_gen2'] if 'best_test_psnr_gen2' in last_checkpoint_data.keys() else 0
#     test_ssim_gen2 = last_checkpoint_data[
#         'best_test_ssim_gen2'] if 'best_test_ssim_gen2' in last_checkpoint_data.keys() else 0
#
#     best_test_psnr_gen = last_checkpoint_data[
#         'best_test_psnr_gen'] if 'best_test_psnr_gen' in last_checkpoint_data.keys() else 0
#
#     print('the latest best_test_psnr_gen2 is {:06f}'.format(best_test_psnr_gen2))
#     print('the latest best_test_ssim_gen2 is {:06f}'.format(test_ssim_gen2))
#
#     save_arguments = {}
#
#     # for name, param in self.model.named_parameters():
#     #     if "W_encoder2" in name: # or 'mlp', W_encoder2
#     #         param.requires_grad = True
#     #     else:
#     #         param.requires_grad = False
#
#     for epoch in range(start_epoch, max_epochs):
#         print('---------------------------------{}---{}-------------------------------------'.format(self.cfg.desc,
#                                                                                                      epoch))
#         train_losses, train_gt_views, train_predict_views, train_input_views = self.run_one_epoch(dl_train,
#                                                                                                   phase='train',
#                                                                                                   optim=optimizer)
#         scheduler.step()  # 循环迭代,epoch作为计数
#         test_losses, test_gt_views, test_predict_views, test_input_views, mertics_all, mertics_synthesis = self.run_one_epoch(
#             dl_test, phase='test')
#         # test_losses, test_gt_views, test_predict_views, test_input_views, mertics_all = train_losses, train_gt_views, train_predict_views, train_input_views,[[0,1],[1,2]]
#
#         train_loss_all = np.mean(train_losses, axis=0)
#         test_loss_all = np.mean(test_losses, axis=0)
#
#         psnr_gen, ssim_gen = np.mean(mertics_all, axis=0)[0], np.mean(mertics_all, axis=0)[1]
#         psnr_gen2, ssim_gen2 = np.mean(mertics_synthesis, axis=0)[0], np.mean(mertics_synthesis, axis=0)[1]
#
#         if self.desc != 'debug':
#             scalars = [train_loss_all, test_loss_all, psnr_gen, ssim_gen, psnr_gen2, ssim_gen2]
#             names = ['train_loss_all', 'test_loss_all', 'psnr_gen', 'ssim_gen', 'psnr_gen2', 'ssim_gen2']
#
#             self.write_tensorboardx(scalars, names, epoch)
#
#         print('Epoch {}: train_loss: {}, test_loss: {}'.format(epoch, train_loss_all, test_loss_all))
#         print('psnr_gen2: {},  ssim_gen2:{}'.format(psnr_gen2, ssim_gen2))
#         print('psnr_gen: {},  ssim_gen:{}'.format(psnr_gen, ssim_gen))
#
#         # save pth every epoch
#         save_arguments['psnr_gen'] = psnr_gen
#         save_arguments['ssim_gen'] = ssim_gen
#         save_arguments['psnr_gen2'] = psnr_gen2
#         save_arguments['ssim_gen2'] = ssim_gen2
#         save_arguments['epoch'] = epoch
#         checkpointer.save('epoch_{}'.format(epoch), **save_arguments)
#         if epoch > 1:
#             os.remove(self.output_dir + '/epoch_{}.pkl'.format(epoch - 1))
#
#         if psnr_gen2 > best_test_psnr_gen2:
#             best_test_psnr_gen2 = psnr_gen2
#             best_test_ssim2 = ssim_gen2
#             save_arguments['best_test_psnr_gen2'] = best_test_psnr_gen2
#             save_arguments['best_test_ssim_gen2'] = ssim_gen2
#             save_arguments['epoch'] = epoch
#             checkpointer.save('best_valid', **save_arguments)
#         if psnr_gen > best_test_psnr_gen:
#             best_test_psnr_gen = psnr_gen
#             best_test_ssim = ssim_gen
#             save_arguments['best_test_psnr_gen'] = best_test_psnr_gen
#             save_arguments['best_test_ssim_gen'] = ssim_gen
#             save_arguments['epoch'] = epoch
#     print('psnr_gen: {},  ssim_gen:{}'.format(best_test_psnr_gen, best_test_ssim))
#     print('psnr_gen2: {},  ssim_gen2:{}'.format(best_test_psnr_gen2, best_test_ssim2))


# class nefnet2(nn.Module):
#     def __init__(self, theta_encoder_len=1, lead_num=3):
#         super(nefnet2, self).__init__()
#         self.lead_num = lead_num
#         self.theta_encoder = ThetaEncoder(encoder_len=theta_encoder_len)
#         self.mlp1_1 = nn.Sequential(nn.Linear((2 + 1) * 4, 128), nn.Tanh())
#         self.mlp1_2 = nn.Sequential(nn.Linear((2 + 1) * 4, 128), nn.Tanh())
#         self.mlp1_3 = nn.Sequential(nn.Linear((2 + 1) * 4, 128), nn.Tanh())
#         self.mlp1 = nn.Sequential(self.mlp1_1,self.mlp1_2,self.mlp1_3)
#         self.mlp2 = nn.Sequential(nn.Linear((2 + 1) * 4, 128), nn.Tanh())
#
#         self.W_encoder = Encoder2(backbone='resnet34', in_channel=lead_num, use_first_pool=True, lead_num=lead_num,
#                                  init_channels=128)
#         self.w_conv = nn.Sequential(
#             BasicBlock(128 * self.lead_num, 128 * self.lead_num, 1, groups=self.lead_num)
#         )
#         self.decoder = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
#             DoubleConv(128 * 1, 128),
#             nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
#             DoubleConv(128, 64),
#             nn.Conv1d(64, 1, 3, padding=1)
#         )
#         self.alpha = nn.Parameter(torch.zeros(3, 2))
#
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=128,
#             nhead=4,
#             dim_feedforward=128,
#             dropout=0.5,
#             activation="gelu",
#             batch_first=True,
#         )
#         self.mask = torch.triu(torch.ones(1152, 1152), diagonal=1).bool()
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
#         self.positional_encoding = PositionalEncoding(128)
#
#     def forward(self, x, input_thetas, query_theta, change=True, phase='train'):
#         '''Args
#         :param x:   [B, lead_num, Length]
#         :param input_thetas:  [B, lead_num, 2]
#         :param query_theta:   [B, 2]
#         :return: out: [B,1,Length]
#         '''
#         if phase == 'finetune' or phase == 'finetune_t':
#             input_thetas = input_thetas + torch.tanh(self.alpha.unsqueeze(0))
#         input_thetas = self.theta_encoder(input_thetas)  # [B, lead_num, 12]
#         encoded_theta = [self.mlp1_1(input_thetas[:, 0:1, :]), self.mlp1_2(input_thetas[:, 1:2, :]),
#                          self.mlp1_3(input_thetas[:, 2:, :])]
#         encoded_theta = torch.cat(encoded_theta, dim=1)
#
#         query_theta = query_theta.unsqueeze(1)
#         query_theta = self.theta_encoder(query_theta)  # [B, 1, 12]
#         query_theta = self.mlp2(query_theta)  # [B, 1, 128]
#
#         w = self.W_encoder(x,query_theta)
#
#         w_one_lead_list = torch.chunk(w, self.lead_num, dim=1)
#         encoded_w_list = [encoded_theta[:, i][:, :, None] * w_one_lead_list[i] for i in
#                           range(self.lead_num)]  # list lead_num * [B, 128, 128]
#         encoded_w_list = self.w_conv(torch.cat(encoded_w_list, dim=1))
#         encoded_w_list = torch.chunk(encoded_w_list, self.lead_num, dim=1)
#
#         attention_weight = attention_block(query_theta, encoded_theta)
#         v = attention_weight[:, :, 0][:, :, None] * encoded_w_list[0] + attention_weight[:, :, 1][:, :, None] * \
#             encoded_w_list[1] + attention_weight[:, :, 2][:, :, None] * encoded_w_list[2]
#         query_w = query_theta[:, 0][:, :, None] * v
#
#         query_w = self.positional_encoding(query_w.transpose(1,2))
#         device = query_w.device
#         mask = self.mask.to(device)
#         query_w = self.transformer(query_w,src_key_padding_mask=None, mask=mask)
#         out = self.decoder(query_w.transpose(1,2))
#         out = torch.sigmoid(out / 3)
#         return out


# class EcgTianChiInterval(Dataset):
#     r""" 训练测试数据集
#     Return
#         Data:[B, 3, 4608]
#         随机采样结束点end_point[2048:4608]
#         有效信号为[0:end_point], [end_point:]填充为0
#     """
#     def __init__(self, cfg, phase, transform=None):
#         self.dataset = None
#         self.label_name = None
#         self.label_dir = None
#         self.cfg = cfg
#         self.phase = phase
#         self.transform = transform
#         self.theta = np.array([[np.pi / 2, np.pi / 2],  # I - np.pi*4/18
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
#         self.alpha = -1*np.pi/18
#         self.theta[0:2,:] = self.theta[0:2,:] - self.alpha
#         self.theta[4, :] = self.theta[4, :] - self.alpha
#         self._read_data(cfg, phase)
#     def _read_data(self, cfg, phase):
#         if phase == 'train':
#             label_path = cfg.DATA.train_label_path
#             with open(label_path) as f:
#                 self.dataset = f.read().splitlines()
#             self.data_root = cfg.DATA.train_data_root
#
#         else:
#             label_path = cfg.DATA.test_label_path
#             with open(label_path) as f:
#                 self.dataset = f.read().splitlines()
#             self.data_root = cfg.DATA.test_data_root
#     def angle_jitter(self, angle, jitter_factor):
#         jitter_angle = jitter_factor / 180 * np.pi
#         jitter = np.random.normal(scale=jitter_angle, size=angle.shape)
#         angle = angle + jitter
#         return angle
#
#     def __getitem__(self, index):
#         file_path = os.path.join(self.data_root, self.dataset[index])
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
#         ECG = np.concatenate([source_data, III, aVR, aVL, aVF], axis=0)
#         Lenght = len(ECG[0, :])
#
#         # random crop
#         if Lenght>4608:
#             random_index_on = random.sample(range(0, Lenght - 4608), k=1)[0]
#             random_index_off = random.sample(range(random_index_on + 2048 ,4608 + random_index_on), k=1)[0]
#         else:
#             random_index_on = 0
#             random_index_off = random.sample(range(2048, Lenght), k=1)[0]
#
#         data = ECG[:, random_index_on:random_index_off]
#         padding_len = 4608 - (random_index_off - random_index_on)
#         data = np.pad(data, ((0, 0), (0, padding_len)), mode='constant', constant_values=0)
#
#         # normalized
#         if self.cfg.DATA.Normalize == True:
#             max_, min_ = np.max(data), np.min(data)
#             data = (data - min_) / (max_ - min_)
#
#         # get noise
#         if self.cfg.DATA.noise == True:
#             noise_region = data[:, :random_index_off - random_index_on]
#             noise_std = np.std(noise_region, axis=1)
#             noise = np.random.normal(loc=0, scale=noise_std, size=(data.shape[-1], 12))
#         else:
#             noise = np.random.randn(4608, 12)
#
#         # angle jitter
#         theta_ = self.theta
#         if self.cfg.MODEL.jitter_factor > 0 and self.phase == 'train':
#             theta_ = self.angle_jitter(theta_, self.cfg.MODEL.jitter_factor)
#
#         synthesis_index = random.sample(self.cfg.DATA.data_synthesis,1)[0]
#         synthesis_view = data[synthesis_index]
#         synthesis_theta = theta_[synthesis_index]
#         if self.cfg.DATA.super_mode == 'pretrain':
#             input_index1 = [0,1]
#             input_index2 = random.sample([x for x in range(2,8) if x not in self.cfg.DATA.data_synthesis], 1)
#             input_index = input_index1 + input_index2
#
#             input_theta = self.theta[input_index]
#             rest_index = [x for x in range(0,8) if x not in input_index + self.cfg.DATA.data_synthesis]
#             target_index = random.sample(rest_index, 1)[0]
#
#         if self.cfg.DATA.super_mode == 'optimization':
#             input_index = self.cfg.DATA.input_index
#             input_theta = self.theta[input_index]
#             rest_index = [x for x in range(0,8) if x not in input_index + self.cfg.DATA.data_synthesis]
#             target_index = random.sample(rest_index, 1)[0]
#
#         if self.cfg.DATA.super_mode == 'transformation':
#             input_index = self.cfg.DATA.input_index
#             input_theta = self.theta[input_index]
#             rest_index = [x for x in range(0,12) if x not in input_index + self.cfg.DATA.data_synthesis]
#             target_index = random.sample(rest_index, 1)[0]
#
#         meta = {
#             'input_index': input_index,
#
#             'data': data[input_index].astype(np.float32),
#             'input_theta': np.array(input_theta).astype(np.float32),
#             'target_view': np.array(data[target_index]).astype(np.float32),
#             'target_theta': np.array(theta_[target_index]).astype(np.float32),
#
#             'id': self.dataset[index],
#             'ori_data': data.astype(np.float32),
#             'thetas':self.theta.astype(np.float32),
#             'synthesis_view': np.array(synthesis_view).astype(np.float32),
#             'synthesis_theta': np.array(synthesis_theta).astype(np.float32),
#
#             'end_point': random_index_off - random_index_on,
#             'target_index': target_index,
#             'noise': noise[:, target_index],
#         }
#         return meta
#
#     def __len__(self):
#         return len(self.dataset)


