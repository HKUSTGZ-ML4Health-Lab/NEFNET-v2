import os, math
import sys
sys.path.append(".")

import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from S4Model import S4Layer
from utils.mertic import SSIM, PSNR
from dataset import build_dataset
from config import cfg
from torch.utils.data import DataLoader

def flatten(v):
    """
    Flatten a list of lists/tuples
    """

    return [x for y in v for x in y]


def find_max_epoch(path):
    """
    Find maximum epoch/iteration in path, formatted ${n_iter}.pkl
    E.g. 100000.pkl

    Parameters:
    path (str): checkpoint path
    
    Returns:
    maximum iteration, -1 if there is no (valid) checkpoint
    """

    files = os.listdir(path)
    epoch = -1
    for f in files:
        if len(f) <= 4:
            continue
        if f[-4:] == '.pkl':
            try:
                epoch = max(epoch, int(f[:-4]))
            except:
                continue
    return epoch


def print_size(net):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)


# Utilities for diffusion models

def std_normal(size):
    """
    Generate the standard Gaussian variable of a certain size
    """

    return torch.normal(0, 1, size=size).cuda()


def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """
    Embed a diffusion step $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

    Parameters:
    diffusion_steps (torch.long tensor, shape=(batchsize, 1)):     
                                diffusion steps for batch data
    diffusion_step_embed_dim_in (int, default=128):  
                                dimensionality of the embedding space for discrete diffusion steps
    
    Returns:
    the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """

    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).cuda()
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed),
                                      torch.cos(_embed)), 1)

    return diffusion_step_embed


def calc_diffusion_hyperparams(T, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value, 
                                where any beta_t in the middle is linearly interpolated
    
    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    Beta = torch.linspace(beta_0, beta_T, T)  # Linear schedule
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t - 1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (
                1 - Alpha_bar[t])  # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1})
        # / (1-\bar{\alpha}_t)
    Sigma = torch.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t

    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma
    diffusion_hyperparams = _dh
    return diffusion_hyperparams


def swish(x):
    return x * torch.sigmoid(x)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out
    
    
class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        return out


class Residual_block(nn.Module):
    def __init__(self, res_channels, skip_channels, 
                 diffusion_step_embed_dim_out, in_channels,
                s4_lmax,
                s4_d_state,
                s4_dropout,
                s4_bidirectional,
                s4_layernorm):
        super(Residual_block, self).__init__()
        self.res_channels = res_channels


        self.fc_t = nn.Linear(diffusion_step_embed_dim_out, self.res_channels)
        
        self.S41 = S4Layer(features=2*self.res_channels, 
                          lmax=s4_lmax,
                          N=s4_d_state,
                          dropout=s4_dropout,
                          bidirectional=s4_bidirectional,
                          layer_norm=s4_layernorm)
 
        self.conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3)

        self.S42 = S4Layer(features=2*self.res_channels, 
                          lmax=s4_lmax,
                          N=s4_d_state,
                          dropout=s4_dropout,
                          bidirectional=s4_bidirectional,
                          layer_norm=s4_layernorm)
        
        self.cond_conv = Conv(2*in_channels, 2*self.res_channels, kernel_size=1)  

        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)

        
        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = nn.utils.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.weight)

    def forward(self, input_data):
        x, cond, diffusion_step_embed = input_data
        h = x
        B, C, L = x.shape
        assert C == self.res_channels                      
                 
        part_t = self.fc_t(diffusion_step_embed)
        part_t = part_t.view([B, self.res_channels, 1])  
        h = h + part_t
        
        h = self.conv_layer(h)
        h = self.S41(h.permute(2,0,1)).permute(1,2,0)     
        
        assert cond is not None
        cond = self.cond_conv(cond)
        h += cond
        
        h = self.S42(h.permute(2,0,1)).permute(1,2,0)
        
        out = torch.tanh(h[:,:self.res_channels,:]) * torch.sigmoid(h[:,self.res_channels:,:])

        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)

        return (x + res) * math.sqrt(0.5), skip  # normalize for training stability


class Residual_group(nn.Module):
    def __init__(self, res_channels, skip_channels, num_res_layers, 
                 diffusion_step_embed_dim_in, 
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 in_channels,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm):
        super(Residual_group, self).__init__()
        self.num_res_layers = num_res_layers
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        self.fc_t1 = nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid)
        self.fc_t2 = nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)
        
        self.residual_blocks = nn.ModuleList()
        for n in range(self.num_res_layers):
            self.residual_blocks.append(Residual_block(res_channels, skip_channels, 
                                                       diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                                       in_channels=in_channels,
                                                       s4_lmax=s4_lmax,
                                                       s4_d_state=s4_d_state,
                                                       s4_dropout=s4_dropout,
                                                       s4_bidirectional=s4_bidirectional,
                                                       s4_layernorm=s4_layernorm))

            
    def forward(self, input_data):
        noise, conditional, diffusion_steps = input_data

        diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        h = noise
        skip = 0
        for n in range(self.num_res_layers):
            h, skip_n = self.residual_blocks[n]((h, conditional, diffusion_step_embed))  
            skip += skip_n  

        return skip * math.sqrt(1.0 / self.num_res_layers)  


class SSSDS4Imputer(nn.Module):
    def __init__(self, in_channels, res_channels, skip_channels, out_channels, 
                 num_res_layers,
                 diffusion_step_embed_dim_in, 
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm):
        super(SSSDS4Imputer, self).__init__()

        self.init_conv = nn.Sequential(Conv(in_channels, res_channels, kernel_size=1), nn.ReLU())
        
        self.residual_layer = Residual_group(res_channels=res_channels, 
                                             skip_channels=skip_channels, 
                                             num_res_layers=num_res_layers, 
                                             diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
                                             diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
                                             diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                             in_channels=in_channels,
                                             s4_lmax=s4_lmax,
                                             s4_d_state=s4_d_state,
                                             s4_dropout=s4_dropout,
                                             s4_bidirectional=s4_bidirectional,
                                             s4_layernorm=s4_layernorm)
        
        self.final_conv = nn.Sequential(Conv(skip_channels, skip_channels, kernel_size=1),
                                        nn.ReLU(),
                                        ZeroConv1d(skip_channels, out_channels))

    def forward(self, input_data):
        
        noise, conditional, mask, diffusion_steps = input_data 

        conditional = conditional * mask
        conditional = torch.cat([conditional, mask.float()], dim=1)

        x = noise
        x = self.init_conv(x)
        x = self.residual_layer((x, conditional, diffusion_steps))
        y = self.final_conv(x)

        return y

    def train_step(self, X, diffusion_hyperparams, loss_fn=nn.MSELoss(), only_generate_missing=1):
        """
        Compute the training loss of epsilon and epsilon_theta

        Parameters:
        net (torch network):            the wavenet model
        loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
        X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
        diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                        note, the tensors need to be cuda tensors       
        
        Returns:
        training loss
        """

        _dh = diffusion_hyperparams
        T, Alpha_bar = _dh["T"], _dh["Alpha_bar"].cuda()

        audio = X[0]
        cond = X[1]
        mask = X[2]

        B, C, L = audio.shape  # B is batchsize, C=1, L is audio length
        diffusion_steps = torch.randint(T, size=(B, 1, 1)).cuda()  # randomly sample diffusion steps from 1~T

        z = std_normal(audio.shape)
        if only_generate_missing == 1:
            z = audio * mask.float() + z * (1 - mask).float()
        transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * audio + torch.sqrt(
            1 - Alpha_bar[diffusion_steps]) * z  # compute x_t from q(x_t|x_0)
        epsilon_theta = self.forward(
            (transformed_X, cond, mask, diffusion_steps.view(B, 1),))  # predict \epsilon according to \epsilon_\theta

        return loss_fn(epsilon_theta, z)

    def test_step(self, size, diffusion_hyperparams, cond, mask, only_generate_missing=0):
        """
        Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

        Parameters:
        net (torch network):            the wavenet model
        size (tuple):                   size of tensor to be generated, 
                                        usually is (number of audios to generate, channels=1, length of audio)
        diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                        note, the tensors need to be cuda tensors 
        
        Returns:
        the generated audio(s) in torch.tensor, shape=size
        """

        _dh = diffusion_hyperparams
        T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"].cuda(), _dh["Alpha_bar"].cuda(), _dh["Sigma"].cuda()
        assert len(Alpha) == T
        assert len(Alpha_bar) == T
        assert len(Sigma) == T

        print('begin sampling, total number of reverse steps = %s' % T)

        x = std_normal(size)

        with torch.no_grad():
            for t in range(T - 1, -1, -1):
                if only_generate_missing == 1:
                    x = x * (1 - mask).float() + cond * mask.float()
                diffusion_steps = (t * torch.ones((size[0], 1))).cuda()  # use the corresponding reverse step
                epsilon_theta = self.forward((x, cond, mask, diffusion_steps,))  # predict \epsilon according to \epsilon_\theta
                # update x_{t-1} to \mu_\theta(x_t)
                x = (x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])
                if t > 0:
                    x = x + Sigma[t] * std_normal(size)  # add the variance term to x_{t-1}

        return x


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion_config = {
        "T": 100,
        "beta_0": 0.0001,
        "beta_T": 0.02
    }
    model_config = {
        "in_channels": 12, 
        "out_channels":12,
        "num_res_layers": 18,
        "res_channels": 128, 
        "skip_channels": 128,
        "diffusion_step_embed_dim_in": 128,
        "diffusion_step_embed_dim_mid": 256,
        "diffusion_step_embed_dim_out": 256,
        "s4_lmax": 100,
        "s4_d_state":64,
        "s4_dropout":0.0,
        "s4_bidirectional":1,
        "s4_layernorm":1
    }
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config)
    # model = nn.DataParallel(SSSDS4Imputer(**model_config))
    # model = model.to(device)
    # ecg = torch.randn((8, 12, 4608)).to(device)
    # mask = torch.randint(0, 2, (ecg.size())).to(device) # mask掉目标导联
    #
    # loss = model.module.train_step((ecg, mask, ecg), diffusion_hyperparams)
    # print(loss)
    # out = model.module.test_step(ecg.size(), diffusion_hyperparams, cond=ecg, mask=mask, only_generate_missing=0)
    # print(out.shape)


    