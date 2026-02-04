import os
from network.loss import loss2
import torch,random,copy
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import tensorboardX
import torch.nn.functional as F
from network import build_model, build_loss
from solver.optim_scheduler import get_optimizer, get_lr_scheduler
from utils.mertic import SSIM, PSNR
from utils import CheckPointer
import loralib as lora
from dataset import build_dataset
import matplotlib.pyplot as plt

class Solver2:
    def __init__(self, cfg, use_tensorboardx=True):
        self.cfg = cfg
        self.output_dir = os.path.join(cfg.output_dir, cfg.desc)
        self.desc = cfg.desc
        self.model = build_model(cfg).float()
        self.loss = build_loss(cfg)
        self.loss2 = loss2
        self._init_model_device()
        if self.desc != 'debug' and use_tensorboardx:
            self.summary_writer = tensorboardX.SummaryWriter(logdir=os.path.join(self.output_dir, 'tf_logs'))
        else:
            self.summary_writer = None
    def load(self,path):
        optimizer = get_optimizer(self.cfg, self.model.parameters())
        scheduler = get_lr_scheduler(self.cfg, optimizer) ##steplr
        checkpointer = CheckPointer(self.model, optimizer, scheduler, self.output_dir)
        checkpointer.load_nefnet2(path=path)

    def _init_model_device(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.cfg.device
        print('GPU: {}'.format(self.cfg.device))
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            if torch.cuda.device_count() > 1:
                print('Using', torch.cuda.device_count(), 'GPUs')
                self.model = nn.DataParallel(self.model)
            else:
                print('Using Single GPU')
        else:
            self.device = torch.device('cpu')
            print('cuda is not available, using cpu')
        self.model.to(self.device)

    def write_tensorboardx(self, scalars, names, epoch):
        for i in range(len(scalars)):
            self.summary_writer.add_scalar(names[i], scalars[i], global_step=epoch)

    def train(self, dl_train, dl_test):
        optimizer = get_optimizer(self.cfg, self.model.parameters())
        scheduler = get_lr_scheduler(self.cfg, optimizer) ##steplr
        checkpointer = CheckPointer(self.model, optimizer, scheduler, self.output_dir)
        last_checkpoint_data = checkpointer.load(self.cfg.MODEL.resume)
        # best_checkpoint_data = checkpointer.load(self.cfg.MODEL.resume,best_valid=True)
        # checkpointer.load_nefnet2(path='/home/zhanzhehui_min/Engineering_project/ECG_Panorama/codes/nefnet2_synthesis1_5cpsc2018/best_valid.pkl')

        max_epochs = self.cfg.SOLVER.epochs
        start_epoch = last_checkpoint_data['epoch']+1 if 'epoch' in last_checkpoint_data.keys() else 0
        best_test_psnr_gen2 = last_checkpoint_data['best_test_psnr_gen2'] if 'best_test_psnr_gen2' in last_checkpoint_data.keys() else 0
        best_test_ssim2 = last_checkpoint_data['best_test_ssim_gen2'] if 'best_test_ssim_gen2' in last_checkpoint_data.keys() else 0
        best_test_psnr_gen = last_checkpoint_data[
            'best_test_psnr_gen'] if 'best_test_psnr_gen' in last_checkpoint_data.keys() else 0
        best_test_ssim = last_checkpoint_data[
            'best_test_ssim_gen'] if 'best_test_ssim_gen' in last_checkpoint_data.keys() else 0
        print('the latest best_test_psnr_gen2 is {:06f}'.format(best_test_psnr_gen2))
        print('the latest best_test_ssim_gen2 is {:06f}'.format(best_test_ssim2))
        save_arguments = {}

        for epoch in range(start_epoch, max_epochs):
            print('---------------------------------{}---{}-------------------------------------'.format(self.cfg.desc,epoch))
            train_losses, train_gt_views, train_predict_views, train_input_views = self.pretrain_one_epoch(dl_train, phase='train', optim=optimizer)
            scheduler.step() #循环迭代,epoch作为计数
            test_losses, test_gt_views, test_predict_views, test_input_views, mertics_all, mertics_synthesis = self.pretrain_one_epoch(dl_test, phase='test')
            train_loss_all = np.mean(train_losses, axis=0)
            test_loss_all = np.mean(test_losses, axis=0)
            psnr_gen, ssim_gen= np.mean(mertics_all, axis=0)[0], np.mean(mertics_all, axis=0)[1]
            psnr_gen2, ssim_gen2 = np.mean(mertics_synthesis, axis=0)[0], np.mean(mertics_synthesis, axis=0)[1]
            if self.desc != 'debug':
                scalars = [train_loss_all, test_loss_all, psnr_gen, ssim_gen, psnr_gen2, ssim_gen2]
                names = ['train_loss_all', 'test_loss_all','psnr_gen', 'ssim_gen', 'psnr_gen2', 'ssim_gen2']
                self.write_tensorboardx(scalars, names, epoch)
            print('Epoch {}: train_loss: {}, test_loss: {}'.format(epoch, train_loss_all, test_loss_all))
            print('psnr_gen: {},  ssim_gen:{}'.format(psnr_gen, ssim_gen))
            print('psnr_gen2: {},  ssim_gen2:{}'.format(psnr_gen2, ssim_gen2))
            save_arguments['psnr_gen'] = psnr_gen
            save_arguments['ssim_gen'] = ssim_gen
            save_arguments['psnr_gen2'] = psnr_gen2
            save_arguments['ssim_gen2'] = ssim_gen2
            save_arguments['epoch'] = epoch
            checkpointer.save('epoch_{}'.format(epoch), **save_arguments)
            if epoch>1:
                os.remove(self.cfg.output_dir + '/none/epoch_{}.pkl'.format(epoch-1))
            if psnr_gen2 > best_test_psnr_gen2:
                best_test_psnr_gen2 = psnr_gen2
                best_test_ssim2 = ssim_gen2
                save_arguments['best_test_psnr_gen2'] = best_test_psnr_gen2
                save_arguments['best_test_ssim_gen2'] = ssim_gen2
                save_arguments['epoch'] = epoch
                checkpointer.save('best_valid', **save_arguments)
            if psnr_gen > best_test_psnr_gen:
                best_test_psnr_gen = psnr_gen
                best_test_ssim = ssim_gen
                save_arguments['best_test_psnr_gen'] = best_test_psnr_gen
                save_arguments['best_test_ssim_gen'] = ssim_gen
                save_arguments['epoch'] = epoch
        print('psnr_gen: {},  ssim_gen:{}'.format(best_test_psnr_gen, best_test_ssim))
        print('psnr_gen2: {},  ssim_gen2:{}'.format(best_test_psnr_gen2, best_test_ssim2))

    def optimization(self, dl_train, dl_test, path):
        if self.cfg.DATA.dataset == 'bspm':
            self.output_dir = self.cfg.output_dir + 'fix' + self.cfg.DATA.dataset + '_input' + str(self.cfg.DATA.lead_num) + '_bspm_supervision_' + str(len(self.cfg.DATA.bspm_supervision))
        else:
            self.output_dir = self.cfg.output_dir + 'fix' + self.cfg.DATA.dataset + '_input' + str(self.cfg.DATA.lead_num)
        # path = os.path.join('/home/zhanzhehui_min/Engineering_project/ECG_Panorama/codes', self.cfg.output_dir, 'best_valid.pkl')
        self.cfg.SOLVER.lr = 5e-4
        output_dir = os.path.join(self.output_dir, self.desc)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.summary_writer = tensorboardX.SummaryWriter(logdir=os.path.join(self.output_dir, 'tf_logs'))
        optimizer = get_optimizer(self.cfg, self.model.parameters())
        scheduler = get_lr_scheduler(self.cfg, optimizer) ##steplr
        checkpointer = CheckPointer(self.model, optimizer, scheduler, self.output_dir)
        last_checkpoint_data = checkpointer.load(self.cfg.MODEL.resume)
        if path != 'none':
            if last_checkpoint_data == {}:
                checkpointer.load_nefnet2(path=path)
        max_epochs = self.cfg.SOLVER.epochs
        start_epoch = last_checkpoint_data['epoch']+1 if 'epoch' in last_checkpoint_data.keys() else 0
        best_test_psnr_gen2 = last_checkpoint_data['best_test_psnr_gen2'] if 'best_test_psnr_gen2' in last_checkpoint_data.keys() else 0
        best_test_ssim2 = last_checkpoint_data['best_test_ssim_gen2'] if 'best_test_ssim_gen2' in last_checkpoint_data.keys() else 0
        best_test_psnr_gen = last_checkpoint_data[
            'best_test_psnr_gen'] if 'best_test_psnr_gen' in last_checkpoint_data.keys() else 0
        best_test_ssim = last_checkpoint_data[
            'best_test_ssim_gen'] if 'best_test_ssim_gen' in last_checkpoint_data.keys() else 0
        print('the latest best_test_psnr_gen2 is {:06f}'.format(best_test_psnr_gen2))
        print('the latest best_test_ssim_gen2 is {:06f}'.format(best_test_ssim2))
        save_arguments = {}
        for epoch in range(start_epoch, max_epochs):
            print('---------------------------------{}---{}-------------------------------------'.format(self.cfg.desc,epoch))
            train_losses, train_gt_views, train_predict_views, train_input_views = self.optimize_one_epoch(dl_train, phase='train', optim=optimizer)
            scheduler.step() #循环迭代,epoch作为计数
            test_losses, test_gt_views, test_predict_views, test_input_views, mertics_all, mertics_synthesis = self.optimize_one_epoch(dl_test, phase='test')
            train_loss_all = np.mean(train_losses, axis=0)
            test_loss_all = np.mean(test_losses, axis=0)
            psnr_gen, ssim_gen= np.mean(mertics_all, axis=0)[0], np.mean(mertics_all, axis=0)[1]
            psnr_gen2, ssim_gen2 = np.mean(mertics_synthesis, axis=0)[0], np.mean(mertics_synthesis, axis=0)[1]
            if self.desc != 'debug':
                scalars = [train_loss_all, test_loss_all, psnr_gen, ssim_gen, psnr_gen2, ssim_gen2]
                names = ['train_loss_all', 'test_loss_all','psnr_gen', 'ssim_gen', 'psnr_gen2', 'ssim_gen2']
                self.write_tensorboardx(scalars, names, epoch)
            print('Epoch {}: train_loss: {}, test_loss: {}'.format(epoch, train_loss_all, test_loss_all))
            print('psnr_gen: {},  ssim_gen:{}'.format(psnr_gen, ssim_gen))
            print('psnr_gen2: {},  ssim_gen2:{}'.format(psnr_gen2, ssim_gen2))
            # save pth every epoch
            save_arguments['psnr_gen'] = psnr_gen
            save_arguments['ssim_gen'] = ssim_gen
            save_arguments['psnr_gen2'] = psnr_gen2
            save_arguments['ssim_gen2'] = ssim_gen2
            save_arguments['epoch'] = epoch
            checkpointer.save('epoch_{}'.format(epoch), **save_arguments)
            if epoch>1:
                os.remove(self.output_dir + '/epoch_{}.pkl'.format(epoch-1))
            if psnr_gen2 > best_test_psnr_gen2:
                best_test_psnr_gen2 = psnr_gen2
                best_test_ssim2 = ssim_gen2
                save_arguments['best_test_psnr_gen2'] = best_test_psnr_gen2
                save_arguments['best_test_ssim_gen2'] = ssim_gen2
                save_arguments['epoch'] = epoch
                checkpointer.save('best_valid', **save_arguments)
            if psnr_gen > best_test_psnr_gen:
                best_test_psnr_gen = psnr_gen
                best_test_ssim = ssim_gen
                save_arguments['best_test_psnr_gen'] = best_test_psnr_gen
                save_arguments['best_test_ssim_gen'] = ssim_gen
                save_arguments['epoch'] = epoch
        print('psnr_gen: {},  ssim_gen:{}'.format(best_test_psnr_gen, best_test_ssim))
        print('psnr_gen2: {},  ssim_gen2:{}'.format(best_test_psnr_gen2, best_test_ssim2))

    def pretrain_one_epoch(self, dl, phase, optim=None):
        if phase == 'train':
            self.model.train()
        elif phase == 'test':
            self.model.eval()
        else:
            raise ValueError('phase param not found.')
        losses = []
        gt_views = []
        predict_views = []
        input_views = []
        mertics_all = []
        mertics_synthesis = []

        for meta in tqdm(dl):
            data, thetas, end_point, noise = meta['data'].to(self.device), meta['thetas'].to(self.device), meta['end_point'].to(self.device), meta['noise'].to(self.device)
            breast_num = random.sample([1,2,3], 1)[0]
            input_index1 = [0, 1]
            input_index2 = random.sample([x for x in range(2, 8) if x not in self.cfg.DATA.data_synthesis], breast_num)
            input_index = input_index1 + input_index2
            rest_index = [x for x in range(0, 8) if x not in input_index + self.cfg.DATA.data_synthesis]
            target_index = random.sample(rest_index, 1)[0]
            input_data,input_theta,target_theta,target_view,noise = data[:,input_index,:],thetas[:,input_index,:],thetas[:,target_index,:].unsqueeze(1), data[:,target_index,:].unsqueeze(1),noise[:,:,target_index].unsqueeze(1)
            result= self.model(input_data, input_theta, target_theta, phase=phase)
            if phase == 'train' and self.cfg.DATA.noise == True:
                out = result + noise
            else:
                out = result
            loss = self.loss(out, target_view)
            losses.append(loss.item())
            if phase == 'test':
                synthesis_view, synthesis_theta = meta['synthesis_view'].unsqueeze(1).to(self.device), meta['synthesis_theta'].unsqueeze(1).to(self.device)
                synthesis_out = self.model(input_data, input_theta, synthesis_theta, phase=phase)
                psnr_gen = PSNR(out.contiguous().cpu().detach().numpy(),
                                target_view.contiguous().cpu().detach().numpy(), end=end_point.cpu())
                ssim_gen = SSIM(out.contiguous().cpu().detach().numpy(),
                                target_view.contiguous().cpu().detach().numpy(), end=end_point.cpu())
                mertics_all.append([psnr_gen, ssim_gen])
                psnr_gen2 = PSNR(synthesis_out.contiguous().cpu().detach().numpy(),
                                synthesis_view.contiguous().cpu().detach().numpy(), end=end_point.cpu())
                ssim_gen2 = SSIM(synthesis_out.contiguous().cpu().detach().numpy(),
                                synthesis_view.contiguous().cpu().detach().numpy(), end=end_point.cpu())
                mertics_synthesis.append([psnr_gen2, ssim_gen2])
            if phase == 'train':
                loss.backward()
                optim.step()
                optim.zero_grad()
            gt_views += [x for x in target_view.squeeze().cpu().detach().numpy()]
            predict_views += [x for x in out.squeeze().cpu().detach().numpy()]
            input_views += [x for x in input_data.cpu().detach().numpy()]
        if phase == 'train':
            return losses, gt_views, predict_views, input_views
        else:
            return losses, gt_views, predict_views, input_views, mertics_all, mertics_synthesis

    def optimize_one_epoch(self, dl, phase, optim=None):
        if phase == 'train' or phase == 'finetune':
            self.model.train()
        elif phase == 'test' or phase == 'finetune_t':
            self.model.eval()
        else:
            raise ValueError('phase param not found.')
        losses = []
        gt_views = []
        predict_views = []
        input_views = []
        mertics_all = []
        mertics_synthesis = []
        for meta in tqdm(dl):
            input_data, input_theta, end_point, noise, target_theta, target_view = meta['input_data'].to(self.device), meta['input_theta'].to(self.device), \
                meta['end_point'].to(self.device), meta['noise'].unsqueeze(1).to(self.device), meta['target_theta'].unsqueeze(1).to(self.device), \
                meta['target_view'].unsqueeze(1).to(self.device)
            result= self.model(input_data, input_theta, target_theta, phase=phase)
            if phase == 'train' and self.cfg.DATA.noise == True:
                out = result + noise
            else:
                out = result
            loss = self.loss(out, target_view)
            losses.append(loss.item())
            if phase == 'test' or phase == 'finetune_t':
                synthesis_view, synthesis_theta = meta['synthesis_view'].unsqueeze(1).to(self.device), meta['synthesis_theta'].unsqueeze(1).to(self.device)
                synthesis_out = self.model(input_data, input_theta, synthesis_theta, phase=phase)

                psnr_gen = PSNR(out.contiguous().cpu().detach().numpy(),
                                target_view.contiguous().cpu().detach().numpy(), end=end_point.cpu())
                ssim_gen = SSIM(out.contiguous().cpu().detach().numpy(),
                                target_view.contiguous().cpu().detach().numpy(), end=end_point.cpu())
                mertics_all.append([psnr_gen, ssim_gen])
                psnr_gen2 = PSNR(synthesis_out.contiguous().cpu().detach().numpy(),
                                synthesis_view.contiguous().cpu().detach().numpy(), end=end_point.cpu())
                ssim_gen2 = SSIM(synthesis_out.contiguous().cpu().detach().numpy(),
                                synthesis_view.contiguous().cpu().detach().numpy(), end=end_point.cpu())
                mertics_synthesis.append([psnr_gen2, ssim_gen2])
            if phase == 'train' or phase == 'finetune':
                loss.backward()
                optim.step()
                optim.zero_grad()
            gt_views += [x for x in target_view.squeeze().cpu().detach().numpy()]
            predict_views += [x for x in out.squeeze().cpu().detach().numpy()]
            input_views += [x for x in input_data.cpu().detach().numpy()]
        if phase == 'train' or phase == 'finetune':
            return losses, gt_views, predict_views, input_views
        else:
            return losses, gt_views, predict_views, input_views, mertics_all,mertics_synthesis

    def finetune(self, dl_finetune, dl_test, path):
        self.output_dir = self.cfg.output_dir + 'finetune' + self.cfg.DATA.dataset + '_input' + str(self.cfg.DATA.lead_num)
        self.cfg.SOLVER.lr = 5e-5
        output_dir = os.path.join(self.output_dir, self.desc)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.summary_writer = tensorboardX.SummaryWriter(logdir=os.path.join(self.output_dir, 'tf_logs'))
        optimizer = get_optimizer(self.cfg, self.model.parameters())
        scheduler = get_lr_scheduler(self.cfg, optimizer) ##steplr
        checkpointer = CheckPointer(self.model, optimizer, scheduler, self.output_dir)
        last_checkpoint_data = checkpointer.load(self.cfg.MODEL.resume)
        if last_checkpoint_data == {}:
            checkpointer.load_nefnet2(path=path)
        test_losses, test_gt_views, test_predict_views, test_input_views, mertics_all, mertics_synthesis = self.optimize_one_epoch(dl_test, phase='test')
        psnr_gen, ssim_gen = np.mean(mertics_all, axis=0)[0], np.mean(mertics_all, axis=0)[1]
        psnr_gen2, ssim_gen2 = np.mean(mertics_synthesis, axis=0)[0], np.mean(mertics_synthesis, axis=0)[1]
        print('Optimization-test:')
        print('psnr_gen: {},  ssim_gen:{}'.format(psnr_gen, ssim_gen))
        print('psnr_gen2: {},  ssim_gen2:{}'.format(psnr_gen2, ssim_gen2))
        print('---------------------------------{}-----------------------------------------------'.format(self.cfg.desc))
        mertics_all_withoutF = []
        mertics_all = []
        for it, meta_finetune in enumerate(tqdm(dl_finetune)):  # bs must be 1
            test_input, target_theta,input_theta = meta_finetune['test_input'].to(self.device), meta_finetune['synthesis_theta'].unsqueeze(1).to(
                self.device),meta_finetune['input_theta'].to(self.device)
            test_target = meta_finetune['test_target'].unsqueeze(1).to(self.device)
            t_end_point = [meta_finetune['end_point'].item()]
            out = self.model(x=test_input, input_thetas=input_theta, query_theta=target_theta)
            psnr_gen1 = PSNR(out.contiguous().cpu().detach().numpy(),test_target.contiguous().cpu().detach().numpy(), end=t_end_point)
            ssim_gen1 = SSIM(out.contiguous().cpu().detach().numpy(),test_target.contiguous().cpu().detach().numpy(), end=t_end_point)
            mertics_all_withoutF.append([psnr_gen1, ssim_gen1])
            out = self.finetune_and_inference_one_sample(meta_finetune, it)
            psnr_gen2 = PSNR(out.contiguous().cpu().detach().numpy(),test_target.contiguous().cpu().detach().numpy(), end=t_end_point)
            ssim_gen2 = SSIM(out.contiguous().cpu().detach().numpy(),test_target.contiguous().cpu().detach().numpy(), end=t_end_point)
            mertics_all.append([psnr_gen2, ssim_gen2])
        psnr_gen1, ssim_gen1 = np.mean(mertics_all_withoutF, axis=0)[0], np.mean(mertics_all_withoutF, axis=0)[1]
        psnr_gen2, ssim_gen2 = np.mean(mertics_all, axis=0)[0], np.mean(mertics_all, axis=0)[1]
        print('Befoer Finetune:')
        print('psnr_gen1: {},  ssim_gen1:{}'.format(psnr_gen1, ssim_gen1))
        print('After Finetune:')
        print('psnr_gen2: {},  ssim_gen2:{}'.format(psnr_gen2, ssim_gen2))

    def finetune_and_inference_one_sample(self, meta_finetune, id=None):
        model_for_sample = self.prepare_for_finetune(self.model).to(self.device)

        optimizer = get_optimizer(self.cfg, filter(lambda p: p.requires_grad, model_for_sample.parameters()))
        scheduler = get_lr_scheduler(self.cfg, optimizer)

        finetune_input, finetune_data, rest_index, input_theta, thetas, end_point = meta_finetune['finetune_input'].to(self.device), \
                meta_finetune['finetune_data'].to(self.device), meta_finetune['rest_index'], \
                meta_finetune['input_theta'].to(self.device), meta_finetune['thetas'].to(self.device), meta_finetune['end_point'].to(self.device)
        end_point = end_point.item()

        batch_size = self.cfg.SOLVER.finetune_batch_size
        rest_index = [x.item() for x in rest_index]
        target_indice = [random.sample(rest_index, 1)[0] for _ in range(batch_size)]

        input_thetas = input_theta.repeat(batch_size,1,1).to(self.device)
        supervision_thetas = thetas[:,target_indice,:].permute(1,0,2).to(self.device)
        supervision_output = finetune_data[:,target_indice,:].permute(1,0,2).to(self.device)

        # Random start point for finetune
        for it in range(self.cfg.SOLVER.finetune_iter):
            random_index_off = random.choices(range(end_point-500,end_point), k=batch_size)
            input = []
            target = []
            for i in range(batch_size):
                a = F.pad(finetune_input[:,:,:random_index_off[i]],(0,4608-random_index_off[i]), mode='constant', value=0)
                input.append(a)
                b = F.pad(supervision_output[i,:,:random_index_off[i]],(0,4608-random_index_off[i]), mode='constant', value=0)
                target.append(b.unsqueeze(1))
            target = torch.cat(target, dim=0)
            input = torch.cat(input, dim=0)

            result = model_for_sample(x=input, input_thetas=input_thetas, query_theta=supervision_thetas, phase='finetune')  # target_index=target_index.item(),
            loss = self.loss(result, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        model_for_sample.eval()
        test_input, target_theta= meta_finetune['test_input'].to(self.device), meta_finetune['synthesis_theta'].unsqueeze(1).to(self.device)
        out = model_for_sample(x=test_input, input_thetas=input_theta, query_theta=target_theta, phase='finetune')
        # Release memory
        del model_for_sample
        return out

    def prepare_for_finetune(self, model):
        model_copy = copy.deepcopy(model)
        for name, param in self.model.named_parameters():
            if any(key in name for key in ['alpha2', 'mlp1', 'mlp2', 'mlp_list','view_transformer']):
                param.requires_grad = True
            else:
                param.requires_grad = False
        print(f"可训练参数数量: {sum(p.numel() for p in model_copy.parameters() if p.requires_grad)}")
        return model_copy

    def Ana_Finetune(self,meta_finetune):
        model_for_sample = self.prepare_for_finetune(self.model).to(self.device)
        optimizer = get_optimizer(self.cfg, filter(lambda p: p.requires_grad, model_for_sample.parameters()))
        scheduler = get_lr_scheduler(self.cfg, optimizer)

        data = meta_finetune['data']
        end_point = meta_finetune['end_point']
        input_index = [0,1,4]
        finetune_input =torch.from_numpy(data[input_index,:].astype(np.float32)).unsqueeze(0)
        input_theta = torch.from_numpy(np.array([[np.pi / 2, np.pi / 2], [np.pi * 5 / 6, np.pi / 2], [np.pi * (19 / 36), np.pi / 12]]).astype(
                np.float32)).unsqueeze(0)
        thetas = np.array([[np.pi / 2, np.pi / 2],  # I
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
                               ]).astype(np.float32)

        batch_size = self.cfg.SOLVER.finetune_batch_size
        rest_index = [x for x in range(2, 8) if x not in input_index + self.cfg.DATA.data_synthesis]
        rest_index = [x.item() for x in rest_index]
        target_indice = [random.sample(rest_index, 1)[0] for _ in range(batch_size)]

        input_thetas = input_theta.repeat(batch_size,1,1).to(self.device)
        supervision_thetas = thetas[:,target_indice,:].permute(1,0,2).to(self.device)
        supervision_output = data[target_indice,:].unsqueeze(0).to(self.device)

        # Random start point for finetune
        for it in range(self.cfg.SOLVER.finetune_iter):
            random_index_off = random.choices(range(end_point-500,end_point), k=batch_size)
            input = []
            target = []
            for i in range(batch_size):
                a = F.pad(finetune_input[:,:,:random_index_off[i]],(0,4608-random_index_off[i]), mode='constant', value=0)
                input.append(a)
                b = F.pad(supervision_output[i,:,:random_index_off[i]],(0,4608-random_index_off[i]), mode='constant', value=0)
                target.append(b.unsqueeze(1))
            target = torch.cat(target, dim=0)
            input = torch.cat(input, dim=0)

            result = model_for_sample(x=input, input_thetas=input_thetas, query_theta=supervision_thetas, phase='finetune')  # target_index=target_index.item(),
            loss = self.loss(result, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        return model_for_sample

    def plt(self):
        import matplotlib.pyplot as plt
        import numpy as np
        x = np.arange(3, 13, 3)  # 横轴：3, 6, 9, 12, 15
        iou_data = {
            '3-Leads': [0.989, 0.982, 0.982, 0.982],  # 深蓝色虚线 + 三角形
            '5-Leads': [0.990, 0.982, 0.985, 0.987],  # 紫色虚线 + 菱形
            '7-Leads': [0.990, 0.988, 0.987, 0.988]  # 浅蓝色虚线 + 方形
        }
        # 2. 设置画布和样式（完全匹配原图风格）
        fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
        ax.set_facecolor('white')  # 白色背景
        # 3. 定义线条样式（继承原图配色逻辑）
        styles = {
            '3-Leads': {'color': 'darkblue', 'linestyle': '--', 'marker': '^', 'markersize': 8},  # 深蓝+三角
            '5-Leads': {'color': 'purple', 'linestyle': '--', 'marker': 'D', 'markersize': 8},  # 紫+菱形
            '7-Leads': {'color': 'lightblue', 'linestyle': '--', 'marker': 's', 'markersize': 8}  # 浅蓝+方形
        }
        # 4. 绘制折线
        for mod in iou_data.keys():
            ax.plot(
                x,
                iou_data[mod],
                label=mod,
                linewidth=4,
                **styles[mod]
            )
        for x_pos in [3, 6, 9, 12]:
            ax.axvline(x=x_pos, color='lightgray', linestyle='--', alpha=0.8, linewidth=2)
        ax.set_ylim(0.978, 0.991)  # 纵轴范围调整为30-40
        ax.set_xticks(x)  # 固定横轴刻度
        ax.grid(True, linestyle=':', alpha=0.6, color='lightgray')  # 浅灰色虚线网格
        # 6. 图例位置（右下角，同原图）
        ax.legend(loc='lower right', frameon=False)
        plt.show()
        print('next one')
        for i in i_list:
            lw = 2
            plt.plot(out['out0'][i][0, 0, :3000].detach().cpu(), color='darkorange', linewidth=lw,
                     linestyle='-')  # 黑色实线
            # 3. 网格线配置
            plt.grid(True, color='lightgray', linestyle='-', alpha=0.7)  # 浅灰色半透明网格
            # 6. 显示图表
            plt.savefig("fig/Appretrain_{}.svg".format(i), format='svg', bbox_inches='tight')
            plt.show()
            plt.plot(out['out1'][i][0, 0, :3000].detach().cpu(), color='blue', linewidth=lw, linestyle='-')  # 黑色实线
            # 3. 网格线配置
            plt.grid(True, color='lightgray', linestyle='-', alpha=0.7)  # 浅灰色半透明网格
            # 6. 显示图表
            plt.savefig("fig/Device_{}.svg".format(i), format='svg', bbox_inches='tight')
            plt.show()
            plt.plot(out['out2'][i][0, 0, :3000].detach().cpu(), color='rebeccapurple', linewidth=lw,
                     linestyle='-')  # 黑色实线
            # 3. 网格线配置
            plt.grid(True, color='lightgray', linestyle='-', alpha=0.7)  # 浅灰色半透明网格
            # 6. 显示图表
            plt.savefig("fig/onthefly_{}.svg".format(i), format='svg', bbox_inches='tight')
            plt.show()
            plt.plot(list_meta[i]['test_target'][0, :3000].detach().cpu(), color='black', linewidth=lw,
                     linestyle='-')  # 黑色实线
            # 3. 网格线配置
            plt.grid(True, color='lightgray', linestyle='-', alpha=0.7)  # 浅灰色半透明网格
            # 6. 显示图表
            plt.savefig("fig/Truth_{}.svg".format(i), format='svg', bbox_inches='tight')
            plt.show()

        def plt_panorama():
        ## 用来绘制不同疾病不同视角下合成的心电信号
            panobench_thetas = np.array(
                [[np.pi * (106 / 180), -np.pi * (102 / 180)], [np.pi * (121 / 180), np.pi * (-101 / 180)],  # 1,2
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
                 ]).astype(np.float32)
            panobench_thetas = torch.from_numpy(panobench_thetas)
            breast_thetas = torch.from_numpy(np.array([
                [np.pi / 2, -np.pi / 18],  # v1
                [np.pi / 2, np.pi / 18],  # v2
                [np.pi * (19 / 36), np.pi / 12],  # v3
                [np.pi * (11 / 20), np.pi / 6],  # v4
                [np.pi * (16 / 30), np.pi / 3],  # v5
                [np.pi * (16 / 30), np.pi / 2]  # v6
            ]).astype(np.float32))

            plt.plot(out[0, 0, 1500:2000].detach().cpu(), linewidth=2)
            plt.grid(True, color='lightgray', linestyle='--', alpha=1)  # 浅灰色半透明网格
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_linewidth(2)
            plt.savefig("fig/panorama/i.svg".format(i), format='svg', bbox_inches='tight')
            plt.show()

            self.load('/home/zhanzhehui_min/Engineering_project/ECG_Panorama/codes/layer_arch_check/layer4_optimization_dataset_sizecpsc2018/fixcpsc2018_input3/best_valid.pkl')
            ECG1 = cpsc2018_C[0]
            ecg = ECG1['data']
            input_ecg = torch.from_numpy(ecg[[0, 1, 4], :].astype(np.float32)).unsqueeze(0)
            input_theta = torch.from_numpy(np.array(
                    [[np.pi / 2, np.pi / 2], [np.pi * 5 / 6, np.pi / 2], [np.pi * (19 / 36), np.pi * (1 / 12)]]).astype(
                    np.float32)).unsqueeze(0)

            out_theta = torch.from_numpy(np.array([[np.pi * (30 / 180), np.pi * (69 / 180)]]).astype(np.float32))

            i = 0
            for out_theta in panobench_thetas:
                i += 1
                out = self.model(input_ecg[:, :, :4608], input_theta, out_theta.unsqueeze(0))
                plt.plot(out[0, 0, 1500:2000].detach().cpu(), linewidth=2)
                plt.grid(True, color='lightgray', linestyle='--', alpha=1)  # 浅灰色半透明网格
                ax = plt.gca()
                for spine in ax.spines.values():
                    spine.set_linewidth(2)
                plt.savefig("fig/panorama/norm/{}.svg".format(i), format='svg', bbox_inches='tight')
                plt.show()
                break








