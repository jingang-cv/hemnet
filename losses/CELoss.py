import math
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
# from zmq import device

# std = 2
def normal_sampling(mean, label_k, std):
    return math.exp(-(label_k-mean) ** 2 / (2*std**2) ) / (math.sqrt(2 * math.pi) * std)

def kl_loss(inputs : torch.Tensor, labels : torch.Tensor):
    criterion = nn.KLDivLoss(reduction='none')
    outputs = torch.log(inputs)
    loss = criterion(outputs, labels)
    loss = loss.sum()
    return loss

class TorchLossComputer(object):
    @staticmethod
    def compute_complex_absolute_given_k(output : torch.Tensor, k : torch.Tensor, N : int):
        device = output.device
        two_pi_n_over_N = Variable(2 * math.pi * torch.arange(0, N, dtype=torch.float), requires_grad=True) / N
        hanning = Variable(torch.from_numpy(np.hanning(N)).type(torch.FloatTensor), requires_grad=True).view(1, -1)

        k = k.type(torch.FloatTensor).to(device)
        two_pi_n_over_N = two_pi_n_over_N.to(device)
        hanning = hanning.to(device)

        output = output.view(1, -1) * hanning
        output = output.view(1, 1, -1).type(torch.cuda.FloatTensor)
        k = k.view(1, -1, 1)
        two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)
        complex_absolute = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 \
                           + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2

        return complex_absolute

    @staticmethod
    def complex_absolute(output : torch.Tensor, Fs : float, bpm_range=None):
        output = output.view(1, -1)

        N = output.size()[1]

        unit_per_hz = Fs / N
        feasible_bpm = bpm_range / 60.0 # 转换到频率
        k = feasible_bpm / unit_per_hz

        # only calculate feasible PSD range [0.7,4]Hz
        complex_absolute = TorchLossComputer.compute_complex_absolute_given_k(output, k, N)

        return (1.0 / complex_absolute.sum()) * complex_absolute	# Analogous Softmax operator
        # return complex_absolute


    @staticmethod
    def cross_entropy_power_spectrum_loss(inputs, target, Fs): # inputs: 160, target:1
        device = inputs.device
        inputs = inputs.view(1, -1) # 160
        target = target.view(1, -1) # 1
        bpm_range = torch.arange(40, 180, dtype=torch.float).to(device) # 140

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range) # [140,1]，对应140个类别

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0) # max返回（values, indices）
        whole_max_idx = whole_max_idx.type(torch.float) # 功率谱密度的峰值对应频率即为心率

        return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)),  torch.abs(target[0] - whole_max_idx)


    @staticmethod
    def cross_entropy_power_spectrum_DLDL_softmax2(inputs, target, Fs, std):
        device = inputs.device

        # 生成目标的心率分布
        target_distribution = [normal_sampling(int(target), i, std) for i in range(140)]
        target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
        target_distribution = torch.Tensor(target_distribution).to(device)

        rank = torch.Tensor([i for i in range(140)]).to(device)

        inputs = inputs.view(1, -1)
        target = target.view(1, -1)

        bpm_range = torch.arange(40, 180, dtype=torch.float).to(device)

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)    # 计算pred的功率谱密度

        # print(00,complex_absolute)
        fre_distribution = F.softmax(complex_absolute.view(-1), dim=0)   # 计算pred的心率分布
        # print(111,fre_distribution)
        loss_distribution_kl = kl_loss(fre_distribution, target_distribution)   # 计算两个分布之间的距离

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)

        return loss_distribution_kl, F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)),  torch.abs(target[0] - whole_max_idx)

class PSDcalculate(nn.Module):
    def __init__(self, device, low_bound=40, high_bound=180,clip_length=256, delta=3):
        super(PSDcalculate, self).__init__()

        self.clip_length = clip_length
        self.time_length = clip_length
        self.device = device
        self.delta = delta
        self.delta_distribution = [0.4, 0.25, 0.05]
        self.low_bound = low_bound
        self.high_bound = high_bound

        self.bpm_range = torch.arange(self.low_bound, self.high_bound, dtype = torch.float).to(self.device)
        self.bpm_range = self.bpm_range / 60.0

        self.pi = 3.14159265
        two_pi_n = Variable(2 * self.pi * torch.arange(0, self.time_length, dtype=torch.float))
        hanning = Variable(torch.from_numpy(np.hanning(self.time_length)).type(torch.FloatTensor), requires_grad=True).view(1, -1)

        self.two_pi_n = two_pi_n.to(self.device)
        self.hanning = hanning.to(self.device)

        self.eps = 0.0001

        self.lambda_l1 = 0.1

        self.kl_loss = nn.KLDivLoss(reduction='none')

        # 生成目标的心率分布
        hr_target = torch.arange(140).reshape(-1, 1).to(self.device)

        label_k = torch.arange(140).repeat(140, 1).to(self.device)
        std = 1.0
        # 计算正态分布的概率密度函数值
        self.probabilities = torch.exp(-(label_k - hr_target.reshape(-1, 1)) ** 2 / (2 * std ** 2)) / (math.sqrt(2 * math.pi) * std)

        # 大于1e-5
        self.probabilities[self.probabilities < 1e-5] = 1e-5


    def forward(self, wave_pr):  # all variable operation
        fps = 30
        batch_size = wave_pr.shape[0]

        f_t = self.bpm_range / fps
        preds = wave_pr * self.hanning

        preds = preds.view(batch_size, 1, -1)
        f_t = f_t.repeat(batch_size, 1).view(batch_size, -1, 1)#[B,110,1]

        tmp = self.two_pi_n.repeat(batch_size, 1)
        tmp = tmp.view(batch_size, 1, -1)

        complex_absolute_pr = torch.sum(preds * torch.sin(f_t*tmp), dim=-1) ** 2 \
                           + torch.sum(preds * torch.cos(f_t*tmp), dim=-1) ** 2 #[B ,110]

        return complex_absolute_pr

    def cal_loss(self, ecg, target):

        # 生成目标的心率分布
        target = target.view(-1).type(torch.long)
        target_distribution = self.probabilities[target]

        complex_absolute = self.forward(ecg)
        complex_absolute = (1.0 / complex_absolute.sum()) * complex_absolute

        fre_distribution = F.softmax(complex_absolute, dim=1)
        kl_loss = self.kl_loss(torch.log(fre_distribution), target_distribution)
        kl_loss = kl_loss.sum(dim=-1).mean()

        fre_loss = F.cross_entropy(complex_absolute, target)
        whole_max_idx = fre_distribution.max(1)[1]
        mae = torch.abs(target - whole_max_idx).type(torch.float).mean()
        # print(whole_max_idx)
        return kl_loss, fre_loss, mae
