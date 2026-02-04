import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from utils.mertic import SSIM, PSNR
from dataset import build_dataset
from config import cfg
from torch.utils.data import DataLoader
class RMSE_Loss(torch.nn.Module):
    def __init__(self, alpha=1):
        super(RMSE_Loss, self).__init__()

    def forward(self, y_true, y_pred):
        loss1 = torch.mean(torch.square(y_true - y_pred), axis = 2)
        loss1= torch.nan_to_num(loss1)
        loss2 = pearson_correlation(y_true, y_pred)
        loss = loss1 - (0.1*loss2)
        return(torch.mean(torch.mean(loss,axis =1)), torch.mean(loss2), torch.mean(loss1))

def pearson_correlation(x,y,eps=1e-6):
    # Ensure that x and y are the same length
    assert len(x) == len(y)

    # Calculate the means of x and y
    x_mean = torch.mean(x, axis = 2)
    y_mean = torch.mean(y, axis = 2)

    # Calculate the variance of x and y
    x_variance = torch.var(x, axis = 2)
    y_variance = torch.var(y, axis = 2)

    # Calculate the standard deviations of x and y
    x_stddev = torch.sqrt(x_variance)
    y_stddev = torch.sqrt(y_variance)

    # Calculate the Pearson correlation coefficient
    r = torch.sum((x - torch.unsqueeze(x_mean, 2)) * (y - torch.unsqueeze(y_mean, 2)), axis = 2 )
    r = r/((len(x[0][0]) * x_stddev * y_stddev) + eps)
    # r = torch.nan_to_num(r)

    return r


class loss_function(torch.nn.Module):
    def __init__(self, alpha=1):
        super(loss_function, self).__init__()

    def forward(self, y_true, y_pred):
        # Calculate the MSE
        loss1 = torch.mean(torch.square(y_true - y_pred), axis = 2)

        # Calculate the mean correlation of an ECG
        loss2 = pearson_correlation(y_true, y_pred)
        
        # Calculate the loss 
        loss = loss1 + torch.ones_like(loss2, device=y_pred.device) - (0.1*loss2)

        # Return the loss, the pearson correlation and the MSE
        return(torch.mean(torch.mean(loss,axis =1)), torch.mean(loss2), torch.mean(loss1))

class Convolution1D_layer(nn.Module):
    def __init__(self, in_f, out_f):
        super(Convolution1D_layer, self).__init__()
        self.f = out_f
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_f, out_channels=out_f, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm1d(num_features=out_f),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.2)
        )

        
    def forward(self, x, device):
        b = len(x)
        new_x = torch.tensor(np.zeros((b,self.f, 12, int(x.shape[-1]/2))).astype("float32")).to(device)
        for i in range(12):
            new_x[:,:,i,:] = self.conv(x[:,:,i,:])
        return(new_x)


class Deconvolution1D_layer(nn.Module):
    def __init__(self, in_f, out_f):
        super(Deconvolution1D_layer, self).__init__()
        self.f = out_f
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(in_channels=in_f, out_channels=out_f, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm1d(num_features=out_f),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.2)
        )

        
    def forward(self, x, device):
        b = len(x)
        new_x = torch.tensor(np.zeros((b,self.f, 12, int(x.shape[-1]*2))).astype("float32")).to(device)
        for i in range(12):
            new_x[:,:,i,:] = self.deconv(x[:,:,i,:])
        return(new_x)
        
class Convolution2D_layer(nn.Module):
    def __init__(self, in_f, out_f):
        super(Convolution2D_layer, self).__init__()        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_f, out_channels=out_f, kernel_size = (13,4), stride = (1,2), padding = (6,1)),
            nn.BatchNorm2d(num_features=out_f),
            nn.LeakyReLU(0.02),
            #nn.Dropout(0.2)
        )

        
    def forward(self, x):
        new_x = self.conv(x)
        return(new_x)       


class Deconvolution2D_layer(nn.Module):
    def __init__(self, in_f, out_f):
        super(Deconvolution2D_layer, self).__init__()
        self.f = out_f
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_f, out_channels=out_f, kernel_size = (13,4), stride = (1,2), padding = (6,1)),
            nn.BatchNorm2d(num_features=out_f),
            nn.LeakyReLU(0.02),
            #nn.Dropout(0.2)
        )

        
    def forward(self, x):
        new_x = self.deconv(x)
        return(new_x)  



class Autoencoder_net(nn.Module):
    def __init__(self):
        super(Autoencoder_net, self).__init__()
        self.first_conv2D = Convolution2D_layer(1,16)
        self.first_conv1D = Convolution1D_layer(1,16)

        self.second_conv2D = Convolution2D_layer(16,32)
        self.second_conv1D = Convolution1D_layer(16,32)

        self.third_conv2D = Convolution2D_layer(32,64)
        self.third_conv1D = Convolution1D_layer(32,64)

        self.fourth_conv2D = Convolution2D_layer(64,128)
        self.fourth_conv1D = Convolution1D_layer(64,128)

        self.first_deconv1D = Deconvolution1D_layer(256,128)
        self.first_deconv2D = Deconvolution2D_layer(256,128)

        self.second_deconv1D = Deconvolution1D_layer(256,64)
        self.second_deconv2D = Deconvolution2D_layer(256,64)

        self.third_deconv1D = Deconvolution1D_layer(128,32)
        self.third_deconv2D = Deconvolution2D_layer(128,32)

        self.fourth_deconv1D = Deconvolution1D_layer(64,1)
        self.fourth_deconv2D = Deconvolution2D_layer(64,1)

        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size = (13,3), stride = (1,1), padding = (6,1)),
            nn.Tanh(),
        )

        self.transition_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size = (13,3), stride = (1,1), padding = (6,1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02)
        )
        self.loss_fn = loss_function()
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        

        
    def forward(self, x):
        conv2D_1 = self.first_conv2D(x)
        conv1D_1 = self.first_conv1D(x, x.device)
        conv_1 = torch.concat((conv1D_1,conv2D_1),axis = 1)
        #print("Conv1: ",conv1D_1.shape)
        #print("Conv1: ",conv2D_1.shape)

        conv2D_2 = self.second_conv2D(conv2D_1)
        conv1D_2 = self.second_conv1D(conv1D_1, x.device)
        #print("Conv2: ",conv1D_2.shape)
        #print("Conv2: ",conv2D_2.shape)
        conv_2 = torch.concat((conv1D_2,conv2D_2),axis = 1)
        #print("Conv2: ",conv_2.shape)

        conv2D_3 = self.third_conv2D(conv2D_2)
        conv1D_3 = self.third_conv1D(conv1D_2,x.device)
        #print("Conv3: ",conv1D_3.shape)
        #print("Conv3: ",conv2D_3.shape)
        conv_3 = torch.concat((conv1D_3,conv2D_3),axis = 1)
        #print("Conv3: ",conv_3.shape)

        conv2D_4 = self.fourth_conv2D(conv2D_3)
        conv1D_4 = self.fourth_conv1D(conv1D_3,x.device)
        #print("Conv4: ",conv1D_4.shape)
        #print("Conv4: ",conv2D_4.shape)
        conv_4 = torch.concat((conv1D_4,conv2D_4),axis = 1)
        #print("Conv4: ",conv_4.shape)

        transition = self.transition_block(conv_4)
        #print("Transition: ", transition.shape)


        deconv2D_1 = self.first_deconv2D(conv_4)
        #print("Deconv 1: ",deconv2D_1.shape)
        deconv_1 = torch.concat((deconv2D_1,conv_3),axis = 1)
        #print("Deconv 1 Concat: ",deconv_1.shape)


        deconv2D_2 = self.second_deconv2D(deconv_1)
        #print("Deconv 2: ",deconv2D_2.shape)
        deconv_2 = torch.concat((deconv2D_2,conv_2),axis = 1)
        #print("Deconv 2 Concat: ",deconv_2.shape)

        deconv2D_3 = self.third_deconv2D(deconv_2)
        #print("Deconv 3: ",deconv2D_3.shape)
        deconv_3 = torch.concat((deconv2D_3,conv_1),axis = 1)
        #print("Deconv 3 Concat: ",deconv_3.shape)

        deconv2D_4 = self.fourth_deconv2D(deconv_3)
        #print("Deconv 4: ",deconv2D_4.shape)

        out = self.final_conv(deconv2D_4)
        out = torch.squeeze(out,1)
        return(out)

    def train_step(self, batch):
        """
        Perform a single training step.
        Args:
            data (Tensor): Input ECG tensor, shape (B,1,12,L)
            target (Tensor): Ground truth ECG, shape (B,12,L)
            optimizer (Optimizer): Optimizer
            loss_fn (callable): loss_function returning (loss, corr, mse)
        Returns:
            dict: { 'loss': float, 'corr': float, 'mse': float }
        """
        data, target = batch
        # Ensure data has shape (B,1,12,L)
        B, C, L = data.shape
        # min max scalar
        data = data.unsqueeze(1) if data.dim() == 3 else data
        if data.shape[2] != 12:
            data = data.repeat(1, 1, 12 // data.shape[2], 1)
        # with torch.autograd.detect_anomaly():
        self.train()
        self.optimizer.zero_grad()
        output = self.forward(data)
        loss_val, corr, mse = self.loss_fn(target, output)
        loss_val.backward()
        self.optimizer.step()
        return {'loss': loss_val.item(), 'corr': corr.item(), 'mse': mse.item()}, None

    def test_step(self, batch):
        """
        Perform a single evaluation step.
        Args:
            data (Tensor): Input ECG tensor, shape (B,1,12,L)
            target (Tensor): Ground truth ECG, shape (B,12,L)
            loss_fn (callable): loss_function returning (loss, corr, mse)
        Returns:
            dict: { 'loss': float, 'corr': float, 'mse': float }
        """
        data, _ = batch
        B, C, L = data.shape
        data = data.unsqueeze(1) if data.dim() == 3 else data
        if data.shape[2] != 12:
            data = data.repeat(1, 1, 12 // data.shape[2], 1)
            
        self.eval()
        with torch.no_grad():
            output = self.forward(data)
        return output
    
    def save_ckpt(self, epoch, val_loss, val_f1, val_acc):
        """
        Save the model checkpoint.
        Args:
            path (str): Path to save the model
            val_loss (float): Validation loss
            val_corr (float): Validation correlation
            val_mse (float): Validation mean squared error
        """
        state = {
            'name': 'Autoencoder_net',
            'epoch': epoch,
            'val_loss': val_loss,
            'val_f1': val_f1,
            'val_acc': val_acc,
            'state_dict': self.state_dict()
        }
        return (state, )
        
    def load_ckpt(self, path):
        """
        Load the model checkpoint.
        Args:
            path (str): Path to the model checkpoint
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']} with val_loss {checkpoint['val_loss']}, "
              f"val_f1 {checkpoint['val_f1']}, val_acc {checkpoint['val_acc']}")


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder_net().to(device)

    keep_channels = [0, 1, 4]
    keep_channels = [0, 1,2, 4,8,9,10,11] # 保留的通道
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
            loss_dict, lg_output = model.train_step(batch)
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

    # model = Autoencoder_net()
    # print(model)
    # # Example input tensor with shape (B, 1, 12, L)
    # input_ecg_tensor = torch.randn(8, 12, 4608)  # Example input tensor
    # target_ecg_tensor = torch.randn(8, 12, 4608)  # Example target tensor
    # dic = model.train_step((input_ecg_tensor, target_ecg_tensor))
    # print(dic)
    # gen = model.test_step((input_ecg_tensor, None))
    # print(gen.shape)  # Should be (8, 12, 4096) if input is (8, 12, 4096)

