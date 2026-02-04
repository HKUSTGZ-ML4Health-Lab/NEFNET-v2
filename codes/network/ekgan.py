import sys
from utils.mertic import SSIM, PSNR
from dataset import build_dataset
from config import cfg
from torch.utils.data import DataLoader
sys.path.append("../")
sys.path.append(".")
sys.path.append("../models")

import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.build_trieval import compute_psnr_ssim
def label_generator_loss(lg_output, input_image):
    target_loss = torch.mean(torch.abs(input_image - lg_output))
    total_disc_loss = target_loss
    return total_disc_loss, target_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    loss_object = nn.BCEWithLogitsLoss()
    real_loss = loss_object(disc_real_output, torch.ones_like(disc_real_output))
    generated_loss = loss_object(disc_generated_output, torch.zeros_like(disc_generated_output))
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

def inference_generator_loss(disc_generated_output, ig_output, target, lambda_, ig_lv, lg_lv, alpha):
    loss_object = nn.BCEWithLogitsLoss()
    gan_loss = loss_object(disc_generated_output, torch.ones_like(disc_generated_output))
    l1_loss = F.l1_loss(ig_output, target)
    vector_loss = F.l1_loss(ig_lv, lg_lv)
    total_gen_loss = lambda_ * l1_loss + gan_loss + alpha * vector_loss
    return total_gen_loss, gan_loss, l1_loss, vector_loss

class MyLRSchedule:
    def __init__(self, initial_lr, dataset_size, batch_size):
        self.initial_lr = initial_lr
        self.path = dataset_size // batch_size

    def __call__(self, step):
        if step < self.path * 5:
            return self.initial_lr
        elif step % self.path == 0:
            return self.initial_lr * 0.95
        else:
            return self.initial_lr


def align_tensor(t1, t2):
    h = min(t1.shape[2], t2.shape[2])
    w = min(t1.shape[3], t2.shape[3])
    return t1[:, :, :h, :w], t2[:, :, :h, :w]


class InferenceGenerator(nn.Module):
    def __init__(self):
        super(InferenceGenerator, self).__init__()
        self.initializer = nn.init.normal_

        # 编码器
        self.enc1 = nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7)
        self.enc2 = nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7)
        self.enc3 = nn.Conv1d(128, 256, kernel_size=15, stride=2, padding=7)
        self.enc4 = nn.Conv1d(256, 512, kernel_size=15, stride=2, padding=7)
        self.enc5 = nn.Conv1d(512, 1024, kernel_size=15, stride=2, padding=7)

        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1024)

        # 解码器
        self.dec1 = nn.ConvTranspose1d(1024, 512, kernel_size=15, stride=2, padding=7, output_padding=1)
        self.dec2 = nn.ConvTranspose1d(1024, 256, kernel_size=15, stride=2, padding=7, output_padding=1)
        self.dec3 = nn.ConvTranspose1d(512, 128, kernel_size=15, stride=2, padding=7, output_padding=1)
        self.dec4 = nn.ConvTranspose1d(256, 64, kernel_size=15, stride=2, padding=7, output_padding=1)
        self.dec5 = nn.ConvTranspose1d(128, 12, kernel_size=15, stride=2, padding=7, output_padding=1)

        self.bn_dec1 = nn.BatchNorm1d(512)
        self.bn_dec2 = nn.BatchNorm1d(256)
        self.bn_dec3 = nn.BatchNorm1d(128)
        self.bn_dec4 = nn.BatchNorm1d(64)

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                self.initializer(m.weight, mean=0.0, std=0.02)

    def forward(self, x):  # x: (B, 12, 4096)
        e1 = self.leaky_relu(self.enc1(x))             # (B, 64, 2048)
        e2 = self.leaky_relu(self.bn2(self.enc2(e1)))  # (B, 128, 1024)
        e3 = self.leaky_relu(self.bn3(self.enc3(e2)))  # (B, 256, 512)
        e4 = self.leaky_relu(self.bn4(self.enc4(e3)))  # (B, 512, 256)
        e5 = self.leaky_relu(self.bn5(self.enc5(e4)))  # (B, 1024, 128)

        d1 = self.relu(self.bn_dec1(self.dec1(e5)))    # (B, 512, 256)
        d1 = torch.cat([d1, e4], dim=1)                # (B, 1024, 256)

        d2 = self.relu(self.bn_dec2(self.dec2(d1)))    # (B, 256, 512)
        d2 = torch.cat([d2, e3], dim=1)                # (B, 512, 512)

        d3 = self.relu(self.bn_dec3(self.dec3(d2)))    # (B, 128, 1024)
        d3 = torch.cat([d3, e2], dim=1)                # (B, 256, 1024)

        d4 = self.relu(self.bn_dec4(self.dec4(d3)))    # (B, 64, 2048)
        d4 = torch.cat([d4, e1], dim=1)                # (B, 128, 2048)

        output = self.dec5(d4)                         # (B, 12, 4096)
        return output, e5


class LabelGenerator(nn.Module):
    def __init__(self):
        super(LabelGenerator, self).__init__()
        self.enc1 = nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7)
        self.enc2 = nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7)
        self.enc3 = nn.Conv1d(128, 256, kernel_size=15, stride=2, padding=7)
        self.enc4 = nn.Conv1d(256, 512, kernel_size=15, stride=2, padding=7)
        self.enc5 = nn.Conv1d(512, 1024, kernel_size=15, stride=2, padding=7)

        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1024)

        self.dec1 = nn.ConvTranspose1d(1024, 512, kernel_size=15, stride=2, padding=7, output_padding=1)
        self.dec2 = nn.ConvTranspose1d(512, 256, kernel_size=15, stride=2, padding=7, output_padding=1)
        self.dec3 = nn.ConvTranspose1d(256, 128, kernel_size=15, stride=2, padding=7, output_padding=1)
        self.dec4 = nn.ConvTranspose1d(128, 64, kernel_size=15, stride=2, padding=7, output_padding=1)
        self.dec5 = nn.ConvTranspose1d(64, 12, kernel_size=15, stride=2, padding=7, output_padding=1)

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        e1 = self.leaky_relu(self.enc1(x))
        e2 = self.leaky_relu(self.bn2(self.enc2(e1)))
        e3 = self.leaky_relu(self.bn3(self.enc3(e2)))
        e4 = self.leaky_relu(self.bn4(self.enc4(e3)))
        e5 = self.leaky_relu(self.bn5(self.enc5(e4)))

        d1 = self.relu(self.dec1(e5))
        d2 = self.relu(self.dec2(d1))
        d3 = self.relu(self.dec3(d2))
        d4 = self.relu(self.dec4(d3))
        output = self.dec5(d4)
        return output, e5


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(24, 64, kernel_size=15, stride=2, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 1, kernel_size=15, stride=1, padding=7)  # output logits
        )

    def forward(self, x, y):
        input = torch.cat([x, y], dim=1)  # (B, 24, L)
        return self.model(input)


class EKGAN(nn.Module):
    """
    EKGAN model combining InferenceGenerator, LabelGenerator, and Discriminator.
    """
    def __init__(self):
        super(EKGAN, self).__init__()
        self.inference_generator = InferenceGenerator()
        self.label_generator = LabelGenerator()
        self.discriminator = Discriminator()
        self.inference_generator_optimizer = torch.optim.Adam(self.inference_generator.parameters(), lr=0.0001)
        self.label_generator_optimizer = torch.optim.Adam(self.label_generator.parameters(), lr=0.0001)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001)

    def forward(self, input_image):
        ig_output, ig_lv = self.inference_generator(input_image)
        lg_output, lg_lv = self.label_generator(input_image)
        return ig_output, ig_lv, lg_output, lg_lv

    def train_step(self, batch, lambda_=100, alpha=0.1):
        input_image, target = batch
        B, C, L = input_image.shape
        if C != 12:
            input_image = input_image.repeat(1, 12 // C, 1)  # Ensure input_image has 12 channels
        # Enable train mode
        self.inference_generator.train()
        self.label_generator.train()
        self.discriminator.train()

        # ----------- Step 1: Generator Forward & Loss --------------
        self.inference_generator_optimizer.zero_grad()
        self.label_generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

        ig_output, ig_lv = self.inference_generator(input_image)
        lg_output, lg_lv = self.label_generator(input_image)

        # discriminator won't track gradients through ig_output here
        with torch.no_grad():
            disc_fake_output = self.discriminator(input_image, ig_output)

        # Backward IG loss
        disc_fake_output_for_ig = self.discriminator(input_image, ig_output)  # requires_grad = True
        total_ig_loss, ig_adv_loss, ig_l1_loss, vector_loss = inference_generator_loss(
            disc_fake_output_for_ig, ig_output, target, lambda_, ig_lv, lg_lv, alpha)
        total_ig_loss.backward()
        self.inference_generator_optimizer.step()

        # ----------- Step 2: Discriminator Forward & Loss ----------
        fake_detached = ig_output.detach()
        disc_fake_output = self.discriminator(input_image, fake_detached)
        disc_real_output = self.discriminator(input_image, target)
        disc_loss = discriminator_loss(disc_real_output, disc_fake_output)
        disc_loss.backward()
        self.discriminator_optimizer.step()

        # ----------- Step 3: Label Generator -----------------------
        lg_output, lg_lv = self.label_generator(input_image)
        total_lg_loss, lg_l1_loss = label_generator_loss(lg_output, input_image)
        total_lg_loss.backward()
        self.label_generator_optimizer.step()

        # ----------- Metrics Collect -------------------------------
        return {
            "loss": total_ig_loss.item(),
            "ig_adv_loss": ig_adv_loss.item(),
            "ig_l1_loss": ig_l1_loss.item(),
            "lg_l1_loss": lg_l1_loss.item(),
            "vector_loss": vector_loss.item(),
            "disc_loss": disc_loss.item()
            }, lg_output
    
    def test_step(self, batch):
        self.inference_generator.eval()
        self.label_generator.eval()
        self.discriminator.eval()
        input_image, target = batch
        B, C, L = input_image.shape
        if C != 12:
            input_image = input_image.repeat(1, 12 // C, 1)
        with torch.no_grad():
            ig_output, _ = self.inference_generator(input_image)
            # recon_metric = compute_psnr_ssim(ig_output, target)
            # Placeholder return structure for compatibility
            return ig_output
    
    def save_ckpt(self, epoch, val_loss, val_f1, val_acc):
        """
        Save the model checkpoint.
        """
        i_state = {
            'name': 'inference_generator',
            'epoch': epoch,
            'val_loss': val_loss,
            'val_f1': val_f1,
            'val_acc': val_acc,
            'state_dict': self.inference_generator.state_dict(),

        }
        l_state = {
            'name': 'label_generator',
            'epoch': epoch,
            'val_loss': val_loss,
            'val_f1': val_f1,
            'val_acc': val_acc,
            'state_dict': self.label_generator.state_dict(),
        }
        d_state = {
            'name': 'discriminator',
            'epoch': epoch,
            'val_loss': val_loss,
            'val_f1': val_f1,
            'val_acc': val_acc,
            'state_dict': self.discriminator.state_dict(),
        }
        return i_state, l_state, d_state
    
    def load_ckpt(self, path):
        """
        Load the model checkpoint.
        """
        i_state = torch.load(path, map_location='cpu', weights_only=False)
        l_state = torch.load(path.replace('_0.pth', '_1.pth'), map_location='cpu', weights_only=False)
        d_state = torch.load(path.replace('_0.pth', '_2.pth'), map_location='cpu', weights_only=False)
        self.inference_generator.load_state_dict(i_state['state_dict'])
        self.label_generator.load_state_dict(l_state['state_dict'])
        self.discriminator.load_state_dict(d_state['state_dict'])
        print(f"Loaded model from epoch {i_state['epoch']} with val_loss {i_state['val_loss']}, "
              f"val_f1 {i_state['val_f1']}, val_acc {i_state['val_acc']}")
        
    


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EKGAN().to(device)

    keep_channels = [0, 1,2, 4,8,9,10,11]  # 保留的通道
    mask = torch.zeros(12)  # 12 通道
    mask[keep_channels] = 1  # 只保留指定通道
    # reshape mask 为 (1, 12, 1)，方便广播
    mask = mask.view(1, 12, 1).to(device)

    from tqdm import tqdm
    import numpy as np
    losses = []
    mertics_all = []
    train_dataset = build_dataset(cfg, phase='train')
    test_dataset = build_dataset(cfg, phase='test')
    dl_train = DataLoader(train_dataset, batch_size=cfg.DATA.batchsize, shuffle=True, num_workers=16, drop_last=True)
    dl_test = DataLoader(test_dataset, batch_size=cfg.DATA.batchsize, num_workers=8, drop_last=True)
    l1_loss = nn.L1Loss(reduction='mean')

    for epoch in range(200):
        for meta in tqdm(dl_train):
            input_data, data = meta['input_data'].to(device), meta['data'].to(device)
            batch = (data*mask, data)
            loss_dict, lg_output = model.train_step(batch, lambda_=100, alpha=0.1)
            losses.append(loss_dict.items())
        if epoch%20==0:
            for meta_test in tqdm(dl_test):
                mertics_test = []
                test_data = meta_test['data'].to(device)
                batch_test = (test_data * mask, test_data)
                end_point = meta_test['end_point']
                ig_output = model.test_step(batch_test)
                psnr_gen = PSNR(ig_output[:,:,:].contiguous().cpu().detach().numpy(),
                            test_data[:,:,:].contiguous().cpu().detach().numpy(), end=end_point.cpu())
                ssim_gen = SSIM(ig_output[:,:,:].contiguous().cpu().detach().numpy(),
                            test_data[:,:,:].contiguous().cpu().detach().numpy(), end=end_point.cpu())
                mertics_test.append([psnr_gen, ssim_gen])
            mertics_all.append([np.mean(mertics_test, axis=0)[0], np.mean(mertics_test, axis=0)[1]])


    # Example usage
    # psnr_gen = PSNR(ig_output[:, [2, 3, 5, 6, 7, 8, 9, 10, 11], :].contiguous().cpu().detach().numpy(),
    #                 test_data[:, [2, 3, 5, 6, 7, 8, 9, 10, 11], :].contiguous().cpu().detach().numpy(),
    #                 end=end_point.cpu())
    # ssim_gen = SSIM(ig_output[:, [2, 3, 5, 6, 7, 8, 9, 10, 11], :].contiguous().cpu().detach().numpy(),
    #                 test_data[:, [2, 3, 5, 6, 7, 8, 9, 10, 11], :].contiguous().cpu().detach().numpy(),
    #                 end=end_point.cpu())
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = EKGAN().to(device)
    #
    # # Dummy data
    # input_image = torch.randn(8, 12, 4608).to(device)  # Batch size of 8
    # target = torch.randn(8, 12, 4608).to(device)
    #
    # batch = (input_image, target)
    #
    # model.train_step(batch, lambda_=100, alpha=0.1)
    # loss_dict, lg_output = model.train_step(batch, lambda_=100, alpha=0.1)
    # # Print the loss dictionary
    # print(loss_dict)
    # ig_output = model.test_step(batch)
    # # Print the output shape
    # print("Label Generator Output Shape:", lg_output.shape)
