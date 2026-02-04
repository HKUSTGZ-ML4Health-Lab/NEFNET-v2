import math
import numpy as np
from skimage.metrics import structural_similarity
import torch
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score, accuracy_score, f1_score

def CC2(generated_signal,real_signal,lead):
    sim_all = []
    batch_size = len(generated_signal)
    for i_ in range(batch_size):
        x = generated_signal[i_,lead,:]
        y = real_signal[i_,lead,:]
        sim = torch.mean((x - x.mean()) * (y - y.mean())) / (torch.sqrt(torch.mean((x - x.mean()) ** 2)) * torch.sqrt(torch.mean((y - y.mean()) ** 2)))
        sim_all.append(sim)
    return sim_all

def PSNR(pred, gt, end, shave_border=0):
    psrn_all = []
    for i in range(pred.shape[0]):
        end_point = end[i]
        for j in range(pred.shape[1]):
            pred_single, gt_single = pred[i, j, :end_point], gt[i, j, :end_point]
            imdff = pred_single - gt_single
            rmse = math.sqrt(np.mean(imdff ** 2))
            if rmse == 0:
                psrn_all.append(100)
            else:
                psrn_all.append(20 * np.log10(1.0 / rmse))

    return np.mean(psrn_all)

def SSIM(pred, gt, end):
    ssim_all = []
    for i in range(pred.shape[0]):
        end_point = end[i]
        for j in range(pred.shape[1]):
            ssim = structural_similarity(pred[i, j, :end_point], gt[i, j, :end_point], data_range=1.0)
            ssim_all.append(ssim)
    return np.mean(ssim_all)
