
import torch
import numpy as np
from torch import nn
import os
import random
import logging

class PBE(object):
    """particle-based entropy based on knn normalized by running mean """
    def __init__(self, rms, knn_clip, knn_k, knn_avg, knn_rms, device):
        self.rms = rms  #用于归一化的运行均值和标准差。
        self.knn_rms = knn_rms  #用于 k-NN 计算的运行均值和标准差。
        self.knn_k = knn_k   #要考虑的最近邻数量。
        self.knn_avg = knn_avg  #一个标志，指示是否计算 k-NN 距离的平均值。
        self.knn_clip = knn_clip #用于稳定性的 k-NN 距离剪辑值。
        self.device = device

    #当使用 PBE 实例作为函数调用时（例如 pbe_instance(rep)），将调用此方法。
    def __call__(self, rep):
        #接受一个输入表示 rep，并计算基于粒子的熵。
        #rep的维度是[batch_size, c]
        source = target = rep
        #使用欧氏距离计算 source 和 target 表示之间的相似性矩阵 (sim_matrix)。
        b1, b2 = source.size(0), target.size(0)
        # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
        #对 sim_matrix 进行 k-NN 搜索，找到每个源点的 knn_k 个最近邻的距离。
        sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) -
                                target[None, :, :].view(1, b2, -1),
                                dim=-1,
                                p=2)
        #对 sim_matrix 进行 k-NN 搜索，找到每个源点的 knn_k 个最近邻的距离。
        reward, _ = sim_matrix.topk(self.knn_k,
                                    dim=1,
                                    largest=False,
                                    sorted=True)  # (b1, k)
        #处理得到的距离 (reward)，取其对数加一 (torch.log(reward + 1.0))。
        reward = reward[:, -1]
        reward = reward.reshape(-1, 1)  # (b1, 1)
        reward = torch.log(reward + 1.0)
        return reward

    
class VCSE(object):
    """particle-based entropy based on knn normalized by running mean """
    def __init__(self, knn_k,device):
        self.knn_k = knn_k
        self.device = device

    def __call__(self, state,value):
        #value => [b1 , 1]
        #state => [b1 , c]
        #z => [b1, c+1]
        # [b1] => [b1,b1]
        ds = state.size(1)
        source = target = state
        b1, b2 = source.size(0), target.size(0)
        # (b1, 1, c+1) - (1, b2, c+1) -> (b1, 1, c+1) - (1, b2, c+1) -> (b1, b2, c+1) -> (b1, b2)
        sim_matrix_s = torch.norm(source[:, None, :].view(b1, 1, -1) -
                                target[None, :, :].view(1, b2, -1),
                                dim=-1,
                                p=2)

        source = target = value
        # (b1, 1, 1) - (1, b2, 1) -> (b1, 1, 1) - (1, b2, 1) -> (b1, b2, 1) -> (b1, b2)
        sim_matrix_v = torch.norm(source[:, None, :].view(b1, 1, -1) -
                                target[None, :, :].view(1, b2, -1),
                                dim=-1,
                                p=2)
        
        sim_matrix = torch.max(torch.cat((sim_matrix_s.unsqueeze(-1),sim_matrix_v.unsqueeze(-1)),dim=-1),dim=-1)[0]
        eps, index = sim_matrix.topk(self.knn_k,
                                    dim=1,
                                    largest=False,
                                    sorted=True)  # (b1, k)
        
        state_norm, index = sim_matrix_s.topk(self.knn_k,
                                    dim=1,
                                    largest=False,
                                    sorted=True)  # (b1, k)
        
        value_norm, index = sim_matrix_v.topk(self.knn_k,
                                    dim=1,
                                    largest=False,
                                    sorted=True)  # (b1, k)
        
        eps = eps[:, -1] #k-th nearest distance
        eps = eps.reshape(-1, 1) # (b1, 1)
        
        state_norm = state_norm[:, -1] #k-th nearest distance
        state_norm = state_norm.reshape(-1, 1) # (b1, 1)

        value_norm = value_norm[:, -1] #k-th nearest distance
        value_norm = value_norm.reshape(-1, 1) # (b1, 1)
        
        sim_matrix_v = sim_matrix_v < eps
        n_v = torch.sum(sim_matrix_v,dim=1,keepdim = True) # (b1,1)
        
        sim_matrix_s = sim_matrix_s < eps
        n_s = torch.sum(sim_matrix_s,dim=1,keepdim = True) # (b1,1)        
        reward = torch.digamma((n_v+1).to(torch.float)) / ds + torch.log(eps * 2 + 0.00001)
        return reward, n_v,n_s, eps, state_norm, value_norm


class TorchRunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=(), device = torch.device("cuda:0")):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon

    def update(self, x):
        with torch.no_grad():
            batch_mean = torch.mean(x, axis=0).to("cuda:0")
            batch_var = torch.var(x, axis=0).to("cuda:0")
            batch_count = x.shape[0]
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    @property
    def std(self):
        return torch.sqrt(self.var)


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):  
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta + batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.pow(delta, 2) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count