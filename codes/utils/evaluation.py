import numpy as np
import torch
from config import cfg
def manual_correlation_matrix(data):
    n_variables = data.shape[0]
    corr_matrix = np.ones((n_variables, n_variables))

    for i in range(n_variables):
        for j in range(i + 1, n_variables):
            # 计算皮尔逊相关系数
            corr = np.corrcoef(data[i], data[j])[0, 1]
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr

    return corr_matrix

def manual_correlation_matrix_absolute(data):
    n_variables = data.shape[0]
    corr_matrix = np.ones((n_variables, n_variables))

    for i in range(n_variables):
        for j in range(i + 1, n_variables):
            corr = np.corrcoef(data[i], data[j])[0, 1]
            corr_matrix[i, j] = abs(corr)  # 取绝对值
            corr_matrix[j, i] = abs(corr)  # 取绝对值

    return corr_matrix


def attention_matrix():
    import scipy
    from scipy.signal import resample
    path = '/home/zhanzhehui_min/Engineering_project/ECG_Panorama/codes/1.mat'
    mat = scipy.io.loadmat(path)
    ecg = mat['Panobench']
    ecg = resample(ecg, 5000, axis=1)
    ecg = pro_ecg(ecg)
    input_ecg = ecg[[0, 1, 29], :]
    input_ecg = torch.from_numpy(input_ecg.astype(np.float32)).unsqueeze(0)

    self = Solver2(cfg)
    self.load('/home/zhanzhehui_min/Engineering_project/ECG_Panorama/codes/layer_arch_check/layer4_optimization_dataset_sizecpsc2018/fixcpsc2018_input3/best_valid.pkl')
    # torch.save(self.model.state_dict(), 'layer4_input5.pth')
    # self.model.load_state_dict(torch.load("layer4_012.pth"))

    input_thetas = torch.from_numpy(np.array([[np.pi / 2, np.pi / 2], [np.pi * 5 / 6, np.pi / 2], [np.pi * (19 / 36), np.pi / 12]]).astype(
            np.float32)).unsqueeze(0)
    out_theta = torch.from_numpy(np.array([
        [np.pi / 2, -np.pi / 18],  # v1
        [np.pi / 2, np.pi / 18],  # v2
        [np.pi * (19 / 36), np.pi / 12],  # v3
        [np.pi * (11 / 20), np.pi / 6],  # v4
        [np.pi * (16 / 30), np.pi / 3],  # v5
        [np.pi * (16 / 30), np.pi / 2],  # v6
    ]).astype(np.float32))

    out = self.model(input_ecg[:, :, :4608], input_thetas, out_theta[0:1,:])

def Eval_LBBB():
    import scipy
    from scipy.signal import resample
    self = Solver2(cfg)
    self.load('/home/zhanzhehui_min/Engineering_project/ECG_Panorama/codes/layer_arch_check/layer4_optimization_dataset_sizecpsc2018/fixcpsc2018_input3/best_valid.pkl')


def prepare_for_finetune(self, model):
    """
    计算神经网络模型的存储大小估计,参数大小为
    """
    model_copy = copy.deepcopy(model)
    for name, param in self.model.named_parameters():
        if any(key in name for key in ['alpha2', 'mlp1', 'mlp2', 'mlp_list','view_transformer']):
            param.requires_grad = True
        else:
            param.requires_grad = False
    print(f"可训练参数数量: {sum(p.numel() for p in model_copy.parameters() if p.requires_grad)}")
    return model_copy

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

    id_list = []
    phase = 'test'
    for meta in tqdm(dl_cpsc2018):
        input_data, input_theta, end_point, noise, target_theta, target_view = meta['input_data'].to(self.device), meta[
            'input_theta'].to(self.device), \
            meta['end_point'].to(self.device), meta['noise'].unsqueeze(1).to(self.device), meta[
            'target_theta'].unsqueeze(
            1).to(self.device), \
            meta['target_view'].unsqueeze(1).to(self.device)
        synthesis_view, synthesis_theta = meta['synthesis_view'].unsqueeze(1).to(self.device), meta[
            'synthesis_theta'].unsqueeze(1).to(self.device)
        synthesis_out = self.model(x=input_data, input_thetas=input_theta, query_theta=synthesis_theta, phase=phase)
        psnr_gen2 = PSNR(synthesis_out.contiguous().cpu().detach().numpy(),
                         synthesis_view.contiguous().cpu().detach().numpy(), end=end_point.cpu())
        if psnr_gen2 > 30:
            id_list.append(meta['id'])

        datas = []
        for meta in tqdm(dl_cpsc2018):
            input_data, input_theta, end_point, noise, target_theta, target_view = meta['input_data'].to(self.device), \
            meta[
                'input_theta'].to(self.device), \
                meta['end_point'].to(self.device), meta['noise'].unsqueeze(1).to(self.device), meta[
                'target_theta'].unsqueeze(
                1).to(self.device), \
                meta['target_view'].unsqueeze(1).to(self.device)
            synthesis_view, synthesis_theta = meta['synthesis_view'].unsqueeze(1).to(self.device), meta[
                'synthesis_theta'].unsqueeze(1).to(self.device)
            synthesis_out = self.model(x=input_data, input_thetas=input_theta, query_theta=synthesis_theta, phase=phase)
            psnr_gen2 = PSNR(synthesis_out.contiguous().cpu().detach().numpy(),
                             synthesis_view.contiguous().cpu().detach().numpy(), end=end_point.cpu())
            if meta['id'][0][0] in data_path:
                datas.append(meta)

    input_data, input_theta, end_point, noise, target_theta, target_view = meta['input_data'].to(self.device), meta[
        'input_theta'].to(self.device), \
        meta['end_point'].to(self.device), meta['noise'].unsqueeze(1).to(self.device), meta['target_theta'].unsqueeze(
        1).to(self.device), \
        meta['target_view'].unsqueeze(1).to(self.device)
    synthesis_view, synthesis_theta = meta['synthesis_view'].unsqueeze(1).to(self.device), meta[
        'synthesis_theta'].unsqueeze(1).to(self.device)
    V1 = self.model(x=input_data, input_thetas=input_theta,
                    query_theta=torch.from_numpy(np.array([[np.pi / 2, -np.pi / 18]]).astype(np.float32)), phase=phase)
    V6 = self.model(x=input_data, input_thetas=input_theta,
                    query_theta=torch.from_numpy(np.array([[np.pi * (16 / 30), np.pi / 2]]).astype(np.float32)),
                    phase=phase)

    ar1 = self.model(x=input_data, input_thetas=input_theta,
                     query_theta=torch.from_numpy(np.array([[np.pi / 2, np.pi * (13 / 18)]]).astype(np.float32)),
                     phase=phase)
    ar2 = self.model(x=input_data, input_thetas=input_theta, query_theta=torch.from_numpy(
        np.array([[np.pi * (132 / 180), -np.pi * (99 / 180)]]).astype(np.float32)), phase=phase)
    ar3 = self.model(x=input_data, input_thetas=input_theta, query_theta=torch.from_numpy(
        np.array([[np.pi * (140 / 180), np.pi * (10 / 18)]]).astype(np.float32)), phase=phase)
    L = 200
    end = 700
    plt.plot(ar1.contiguous().cpu().detach().numpy()[0, 0, L:end])
    plt.savefig("lbbb/ar1_01.svg", format="svg")
    plt.show()
    plt.plot(ar2.contiguous().cpu().detach().numpy()[0, 0, L:end])
    plt.savefig("lbbb/ar2_01.svg", format="svg")
    plt.show()
    plt.plot(ar3.contiguous().cpu().detach().numpy()[0, 0, L:end])
    plt.savefig("lbbb/ar3_01.svg", format="svg")
    plt.show()
    data = meta['data']
    plt.plot(
        0.5 * V1.contiguous().cpu().detach().numpy()[0, 0, L:end] + 0.5 * data.contiguous().cpu().detach().numpy()[0, 2,
                                                                          L:end])
    plt.savefig("lbbb/v1_g01.svg", format="svg")
    plt.show()
    plt.plot(
        0.8 * V6.contiguous().cpu().detach().numpy()[0, 0, L:end] + 0.2 * data.contiguous().cpu().detach().numpy()[0, 7,
                                                                          L:end])
    plt.savefig("lbbb/v6_g01.svg", format="svg")
    plt.show()
    plt.plot(data.contiguous().cpu().detach().numpy()[0, 2, L:end])
    plt.savefig("lbbb/v1_t01.svg", format="svg")
    plt.show()
    plt.plot(data.contiguous().cpu().detach().numpy()[0, 7, L:end])
    plt.savefig("lbbb/v6_t01.svg", format="svg")
    plt.show()


def is_lead_dropped(lead, global_std_median,
                    std_ratio_thresh=0.1,
                    nan_ratio_thresh=0.5,
                    flat_ratio_thresh=0.95):
    """
    lead: shape (L,)
    返回 True 表示这个导联“疑似脱落”
    """
    # 1) NaN 过多
    nan_ratio = np.isnan(lead).mean()
    if nan_ratio > nan_ratio_thresh:
        return True

    # 2) 去掉 NaN 方便后面统计
    lead_clean = lead[~np.isnan(lead)]
    if lead_clean.size == 0:
        return True

    # 3) 几乎平直（std 很小）：相对阈值
    std = np.std(lead_clean)
    if std < global_std_median * std_ratio_thresh:
        return True

    # 4) 大部分点都一样（全 0 或接近某个常数，比如 ADC 饱和）
    values, counts = np.unique(lead_clean, return_counts=True)
    most_common_ratio = counts.max() / lead_clean.size
    if most_common_ratio > flat_ratio_thresh:
        return True

    return False


def is_lead_drop(ecg):
    N, C, L = ecg.shape  # C 应该是 12

    bad_sample_mask = np.zeros(N, dtype=bool)  # True 表示这一条样本有问题
    bad_lead_mask = np.zeros((N, C), dtype=bool)  # 记录具体哪几个导联

    for i in range(N):  # 遍历每个样本
        for c in range(C):  # 遍历 12 个导联
            lead = ecg[i, c, :]
            if is_lead_dropped(lead, global_std_median):
                bad_sample_mask[i] = True
                bad_lead_mask[i, c] = True

    # 有导联脱落的样本索引
    bad_indices = np.where(bad_sample_mask)[0]
    # 完整正常的样本索引
    good_indices = np.where(~bad_sample_mask)[0]

    print("疑似有导联脱落的样本数:", len(bad_indices))
    print("前几个坏样本索引:", bad_indices[:10])

    # 如果你想直接把“坏样本”筛出来：
    ecg_bad = ecg[bad_indices]
    ecg_good = ecg[good_indices]

