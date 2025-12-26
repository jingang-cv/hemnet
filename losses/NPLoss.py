import torch
import torch.nn as nn
import numpy as np

class Neg_Pearson(nn.Module):  # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Neg_Pearson, self).__init__()
        return

    def forward(self, preds : torch.Tensor, labels : torch.Tensor):  # all variable operation
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])  # x
            sum_y = torch.sum(labels[i])  # y
            sum_xy = torch.sum(preds[i] * labels[i])  # xy
            sum_x2 = torch.sum(torch.pow(preds[i], 2))  # x^2
            sum_y2 = torch.sum(torch.pow(labels[i], 2))  # y^2
            N = preds.shape[1]
            pearson = (N * sum_xy - sum_x * sum_y) / (
               torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))
            loss += 1 - pearson

        loss = loss / preds.shape[0]
        # if torch.isnan(loss):
        #     np.savez("nan.npz", preds=preds.detach().cpu().numpy(), labels=labels.detach().cpu().numpy())
        return loss

class NP_loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, gt_lable, pre_lable):
        gt_lable = gt_lable.unsqueeze(1)
        pre_lable = pre_lable.unsqueeze(1)
        M, N, A = gt_lable.shape
        gt_lable = gt_lable - torch.mean(gt_lable, dim=2).view(M, N, 1)
        pre_lable = pre_lable - torch.mean(pre_lable, dim=2).view(M, N, 1)
        aPow = torch.sqrt(torch.sum(torch.mul(gt_lable, gt_lable), dim=2))
        bPow = torch.sqrt(torch.sum(torch.mul(pre_lable, pre_lable), dim=2))
        pearson = torch.sum(torch.mul(gt_lable, pre_lable), dim=2) / (aPow * bPow + 0.01)
        loss = 1 - torch.sum(torch.sum(pearson, dim=1), dim=0)/(gt_lable.shape[0] * gt_lable.shape[1])
        return loss
