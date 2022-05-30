import copy
import math

import torch

from graph_learners import *
from layers import GCNConv_dense, GCNConv_dgl, SparseDropout
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F

# GCN for evaluation.
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, Adj, sparse,drug_sim,drug_fea,mic_sim,mic_fea):
        super(GCN, self).__init__()
        self.drug_sim=drug_sim
        self.drug_fea=drug_fea
        self.mic_sim=mic_sim
        self.mic_fea=mic_fea
        self.layers = nn.ModuleList()
        if sparse:
            self.layers.append(GCNConv_dgl(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv_dgl(hidden_channels, hidden_channels))
            self.layers.append(GCNConv_dgl(hidden_channels, out_channels))
        else:
            self.layers.append(GCNConv_dense(in_channels, hidden_channels))#1444*32
            for i in range(num_layers - 2):
                self.layers.append(GCNConv_dense(hidden_channels, hidden_channels))#32*32
            self.layers.append(GCNConv_dense(hidden_channels, out_channels)) #32*1546

        self.dropout = dropout
        self.dropout_adj_p = dropout_adj
        self.Adj = Adj #1444*1444
        self.Adj.requires_grad = False
        self.sparse = sparse

        if self.sparse:
            self.dropout_adj = SparseDropout(dprob=dropout_adj)
        else:
            self.dropout_adj = nn.Dropout(p=dropout_adj)

    def forward(self, x):

        if self.sparse:
            Adj = copy.deepcopy(self.Adj)
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            Adj = self.dropout_adj(self.Adj)#1444*1444

        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, Adj) #1444*2888
        return x
class prediction_model(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, Adj, sparse,drug_sim,drug_fea,mic_sim,mic_fea):
        super(prediction_model, self).__init__()
        self.drug_sim=drug_sim
        self.drug_fea=drug_fea
        self.mic_sim=mic_sim
        self.mic_fea=mic_fea

        self.drug_l = []
        self.mic_l = []

        self.drug_k = []
        self.mic_k = []

        self.kernel_len =8
        self.mi_ps = torch.ones(self.kernel_len) /self.kernel_len #[0.2500,0.2500,0.2500,0.2500]
        self.dis_ps = torch.ones(self.kernel_len) / self.kernel_len #[0.2500,0.2500,0.2500,0.2500]

        self.alpha1 = torch.randn(self.drug_sim.shape[0], self.mic_sim.shape[0]).float() #1373*173
        self.alpha2 = torch.randn(self.mic_sim.shape[0], self.drug_sim.shape[0]).float() #173*1373

        self.gcn1 = GCN(in_channels, 32,256, num_layers, dropout, dropout_adj, Adj, sparse) #1546*3092 * 3092*32 *32*256
        self.gcn2 = GCN(256 , 32, 128, num_layers, dropout, dropout_adj, Adj, sparse) #1546*256 *256*32 *32*128
        self.gcn3 = GCN(128, 32, 64, num_layers, dropout, dropout_adj, Adj, sparse) #1546*128 *128*32 *32*64
        self.h1_gamma = 2 ** (-5)
        self.h2_gamma = 2 ** (-3)
        self.lambda1 = 2 ** (-3)
        self.lambda2 = 2 ** (-4)
    def forward(self, x):
        drug_kernels=[]
        mic_kernels=[]
        H1 = torch.relu(self.gcn1(x))
        drug_kernels.append(torch.FloatTensor(getGipKernel(H1[:853].clone(), 0, self.h1_gamma, True).float())) 
        mic_kernels.append(torch.FloatTensor(getGipKernel(H1[853:].clone(), 0, self.h1_gamma, True).float())) 
        H2 = torch.relu(self.gcn2(H1))
        drug_kernels.append(torch.FloatTensor(getGipKernel(H2[:853].clone(), 0, self.h2_gamma, True).float())) 
        mic_kernels.append(torch.FloatTensor(getGipKernel(H2[853:].clone(), 0, self.h2_gamma, True).float())) 
        H3 = torch.relu(self.gcn3(H2))
        drug_kernels.append(torch.FloatTensor(getGipKernel(H3[:853].clone(), 0, self.h2_gamma, True).float())) 
        mic_kernels.append(torch.FloatTensor(getGipKernel(H3[853:].clone(), 0, self.h2_gamma, True).float())) 

        H4 = torch.relu(self.gcn1(x))
        drug_kernels.append(torch.FloatTensor(getGipKernel(H4[:853].clone(), 0, self.h1_gamma, True).float())) 
        mic_kernels.append(torch.FloatTensor(getGipKernel(H4[853:].clone(), 0, self.h1_gamma, True).float())) 
        H5 = torch.relu(self.gcn2(H4))
        drug_kernels.append(torch.FloatTensor(getGipKernel(H5[:853].clone(), 0, self.h2_gamma, True).float())) 
        mic_kernels.append(torch.FloatTensor(getGipKernel(H5[853:].clone(), 0, self.h2_gamma, True).float())) 
        H6 = torch.relu(self.gcn3(H5))
        drug_kernels.append(torch.FloatTensor(getGipKernel(H6[:853].clone(), 0, self.h2_gamma, True).float())) 
        mic_kernels.append(torch.FloatTensor(getGipKernel(H6[853:].clone(), 0, self.h2_gamma, True).float())) 

        drug_kernels.append(self.drug_sim)
        drug_kernels.append(self.drug_fea)
        mic_kernels.append(self.mic_sim)
        mic_kernels.append(self.mic_fea)

        drug_k = sum([self.mi_ps[i] * drug_kernels[i] for i in range(len(self.mi_ps))])
        mic_k = sum([self.dis_ps[i] * mic_kernels[i] for i in range(len(self.dis_ps))])
        self.drug_k = normalized_kernel(drug_k)
        self.mic_k =normalized_kernel(mic_k)
        self.drug_l = laplacian(drug_k) #1373*1373 Ld
        self.mic_l = laplacian(mic_k) #Lm
        out1 = torch.mm(self.drug_k, self.alpha1) 
        out2 = torch.mm(self.mic_k, self.alpha2) 
        out = (out1 + out2.T) / 2 
        return out

class GraphEncoder(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse):

        super(GraphEncoder, self).__init__()
        self.dropout = dropout
        self.dropout_adj_p = dropout_adj
        self.sparse = sparse

        self.gnn_encoder_layers = nn.ModuleList()
        if sparse:
            self.gnn_encoder_layers.append(GCNConv_dgl(in_dim, hidden_dim))
            for _ in range(nlayers - 2):
                self.gnn_encoder_layers.append(GCNConv_dgl(hidden_dim, hidden_dim))
            self.gnn_encoder_layers.append(GCNConv_dgl(hidden_dim, emb_dim))
        else:
            self.gnn_encoder_layers.append(GCNConv_dense(in_dim, hidden_dim))
            for _ in range(nlayers - 2):
                self.gnn_encoder_layers.append(GCNConv_dense(hidden_dim, hidden_dim))
            self.gnn_encoder_layers.append(GCNConv_dense(hidden_dim, emb_dim))

        if self.sparse:
            self.dropout_adj = SparseDropout(dprob=dropout_adj)
        else:
            self.dropout_adj = nn.Dropout(p=dropout_adj)

        self.proj_head = Sequential(Linear(emb_dim, proj_dim), ReLU(inplace=True),
                                           Linear(proj_dim, proj_dim))

    def forward(self,x, Adj_, branch=None):

        if self.sparse:
            if branch == 'anchor':
                Adj = copy.deepcopy(Adj_)
            else:
                Adj = Adj_
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            Adj = self.dropout_adj(Adj_)

        for conv in self.gnn_encoder_layers[:-1]:
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gnn_encoder_layers[-1](x, Adj)
        z = self.proj_head(x)
        return z, x
'''
    作用：分别学习锚视图和学习视图的节点级嵌入
'''
class GCL(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse):
        super(GCL, self).__init__()

        self.encoder = GraphEncoder(nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse)

    def forward(self, x, Adj_, branch=None):
        z, embedding = self.encoder(x, Adj_, branch)
        return z, embedding

    @staticmethod
    def calc_loss(x, x_aug, temperature=0.2, sym=True): #公式14 节点级对比损失函数
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        if sym:
            loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

            loss_0 = - torch.log(loss_0).mean()
            loss_1 = - torch.log(loss_1).mean()
            loss = (loss_0 + loss_1) / 2.0
            return loss
        else:
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss_1 = - torch.log(loss_1).mean()
            return loss_1