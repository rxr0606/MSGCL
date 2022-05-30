import torch as t
from torch import nn


class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()
    '''
        输入：train_data['Y_train']（真实值）, score, model.drug_l（Ld）, 
        model.mic_l（Lm）, model.alpha1（alpha_d 训练参数）,model.alpha2（alpha_m 训练参数）, sizes
    '''
    def forward(self, target, prediction, drug_lap, mic_lap, alpha1, alpha2, sizes):
        loss_ls = t.norm((target - prediction), p='fro') ** 2

        drug_reg = t.trace(t.mm(t.mm(alpha1.T, drug_lap), alpha1))#公式18
        mic_reg = t.trace(t.mm(t.mm(alpha2.T, mic_lap), alpha2))
        graph_reg = sizes.lambda1 * drug_reg + sizes.lambda2 * mic_reg

        loss_sum = loss_ls + graph_reg

        return loss_sum.sum()
