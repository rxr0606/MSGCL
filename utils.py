import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
import dgl
from sklearn import metrics
from munkres import Munkres

EOS = 1e-10
'''
    输入：1373*173
    功能：将药物与微生物的特征阵与0阵堆叠
    输出：1546*1546
'''
def kernelToDistance(k):
    di = torch.diag(k).T
    d = di.repeat(len(k)).reshape(len(k), len(k)).T + di.repeat(len(k)).reshape(len(k), len(k)) - 2 * k
    return d

def normalized_kernel(K):
    K = abs(K)
    k = K.flatten().sort()[0] #按行展开 1*（1373*1373）由小到大排序
    min_v = k[torch.nonzero(k, as_tuple=False)[0]] #取出最小值
    K[torch.where(K == 0)] = min_v #将k中为0的元素换成这个最小值
    D = torch.diag(K)  #1373的列表 tensor 全为1 取出对角线的元素
    D = D.sqrt()
    S = K / (D * D.T)
    return S

def normalized_embedding(embeddings):
    [row, col] = embeddings.size() #1373,128
    ne = torch.zeros([row, col])
    for i in range(row):
        ne[i, :] = (embeddings[i, :] - min(embeddings[i, :])) / (max(embeddings[i, :]) - min(embeddings[i, :]))
    return ne

def laplacian(kernel):
    d1 = sum(kernel) #tensor 1373的一维列表 每一列求和
    D_1 = torch.diag(d1) #求对角阵 得到度矩阵D 1373*1373 只有对角线上有值
    L_D_1 = D_1 - kernel #Δd
    D_5 = D_1.rsqrt() #每个元素取平方根再取倒数 因为0取不了平方根 所以为inf值
    D_5 = torch.where(torch.isinf(D_5), torch.full_like(D_5, 0), D_5) #如果值为inf则将其变成0
    L_D_11 = torch.mm(D_5, L_D_1) #这里开始是Ld的计算
    L_D_11 = torch.mm(L_D_11, D_5)
    return L_D_11

def getGipKernel(y, trans, gamma, normalized=False): #H1[0:1373]，0，0.03125，False
        if trans:
            y = y.T
        if normalized:
            y = normalized_embedding(y)
        y_=y.detach().numpy()
        y=np.nan_to_num(y_)
        y=torch.FloatTensor(y)
        krnl = torch.mm(y, y.T)#1373*1546 * 1546*1373=1373*1373
        krnl = krnl / torch.mean(torch.diag(krnl)) #torch.diag（取矩阵对角元素）
        krnl = torch.exp(-kernelToDistance(krnl) * gamma) #公式10
        return krnl
def constructNet(mat1,mat2): 
    mat = np.vstack((mat1, mat2))
    return mat

def apply_non_linearity(tensor, non_linearity, i):
    if non_linearity == 'elu':
        return F.elu(tensor * i - i) + 1
    elif non_linearity == 'relu':
        return F.relu(tensor)
    elif non_linearity == 'none':
        return tensor
    else:
        raise NameError('We dont support the non-linearity yet')

'''
    init_list=features.shape[0]=2708
    batch_size=0
    return:?
'''
def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list


def edge_deletion(adj, drop_r):
    edge_index = np.array(np.nonzero(adj))
    half_edge_index = edge_index[:, edge_index[0,:] < edge_index[1,:]]
    num_edge = half_edge_index.shape[1]
    samples = np.random.choice(num_edge, size=int(drop_r * num_edge), replace=False)
    dropped_edge_index = half_edge_index[:, samples].T
    adj[dropped_edge_index[:,0],dropped_edge_index[:,1]] = 0.
    adj[dropped_edge_index[:,1],dropped_edge_index[:,0]] = 0.
    return adj

def edge_addition(adj, add_r):
    edge_index = np.array(np.nonzero(adj))
    half_edge_index = edge_index[:, edge_index[0,:] < edge_index[1,:]]
    num_edge = half_edge_index.shape[1]
    num_node = adj.shape[0]
    added_edge_index_in = np.random.choice(num_node, size=int(add_r * num_edge), replace=True)
    added_edge_index_out = np.random.choice(num_node, size=int(add_r * num_edge), replace=True)
    adj[added_edge_index_in,added_edge_index_out] = 1.
    adj[added_edge_index_out,added_edge_index_in] = 1.
    return adj


def get_feat_mask(features, mask_rate):
    feat_node = features.shape[1]
    mask = torch.zeros(features.shape)
    samples = np.random.choice(feat_node, size=int(feat_node * mask_rate), replace=False)
    mask[:, samples] = 1
    return mask.cuda(), samples


def accuracy(preds, labels):
    pred_class = torch.max(preds, 1)[1]
    return torch.sum(torch.eq(pred_class, labels)).float() / labels.shape[0]


def nearest_neighbors(X, k, metric):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    return adj


def nearest_neighbors_sparse(X, k, metric):
    adj = kneighbors_graph(X, k, metric=metric)
    loop = np.arange(X.shape[0])
    [s_, d_, val] = sp.find(adj)
    s = np.concatenate((s_, loop))
    d = np.concatenate((d_, loop))
    return s, d


def nearest_neighbors_pre_exp(X, k, metric, i):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    adj = adj * i - i
    return adj

'''
    X:feature 2708*1433
    k：30 后处理k近邻
    metric：cosine
    i=6
'''
def nearest_neighbors_pre_elu(X, k, metric, i):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    adj = adj * i - i
    return adj


def normalize(adj, mode, sparse=False):
    if not sparse:
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :] #
        elif mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
            return inv_degree[:, None] * adj
        else:
            exit("wrong norm mode")
    else:
        adj = adj.coalesce()
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()))
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]

        elif mode == "row":
            aa = torch.sparse.sum(adj, dim=1)
            bb = aa.values()
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        new_values = adj.values() * D_value

        return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())


def symmetrize(adj):  # only for non-sparse
    return (adj + adj.T) / 2


def cal_similarity_graph(node_embeddings):
    similarity_graph = torch.mm(node_embeddings, node_embeddings.t())
    return similarity_graph


def top_k(raw_graph, K):
    values, indices = raw_graph.topk(k=int(K), dim=-1)
    assert torch.max(indices) < raw_graph.shape[1]
    mask = torch.zeros(raw_graph.shape).cuda()
    mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices] = 1.

    mask.requires_grad = False
    sparse_graph = raw_graph * mask
    return sparse_graph


def knn_fast(X, k, b):
    X = F.normalize(X, dim=1, p=2)
    index = 0
    values = torch.zeros(X.shape[0] * (k + 1)).cuda()
    rows = torch.zeros(X.shape[0] * (k + 1)).cuda()
    cols = torch.zeros(X.shape[0] * (k + 1)).cuda()
    norm_row = torch.zeros(X.shape[0]).cuda()
    norm_col = torch.zeros(X.shape[0]).cuda()
    while index < X.shape[0]:
        if (index + b) > (X.shape[0]):
            end = X.shape[0]
        else:
            end = index + b
        sub_tensor = X[index:index + b]
        similarities = torch.mm(sub_tensor, X.t())
        vals, inds = similarities.topk(k=k + 1, dim=-1)
        values[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
        cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
        rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index, end).view(-1, 1).repeat(1, k + 1).view(-1)
        norm_row[index: end] = torch.sum(vals, dim=1)
        norm_col.index_add_(-1, inds.view(-1), vals.view(-1))
        index += b
    norm = norm_row + norm_col
    rows = rows.long()
    cols = cols.long()
    values *= (torch.pow(norm[rows], -0.5) * torch.pow(norm[cols], -0.5))
    return rows, cols, values


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def torch_sparse_to_dgl_graph(torch_sparse_mx):
    torch_sparse_mx = torch_sparse_mx.coalesce()
    indices = torch_sparse_mx.indices()
    values = torch_sparse_mx.values()
    rows_, cols_ = indices[0,:], indices[1,:]
    dgl_graph = dgl.graph((rows_, cols_), num_nodes=torch_sparse_mx.shape[0], device='cuda')
    dgl_graph.edata['w'] = values.detach().cuda()
    return dgl_graph


def dgl_graph_to_torch_sparse(dgl_graph):
    values = dgl_graph.edata['w'].cpu().detach()
    rows_, cols_ = dgl_graph.edges()
    indices = torch.cat((torch.unsqueeze(rows_, 0), torch.unsqueeze(cols_, 0)), 0).cpu()
    torch_sparse_mx = torch.sparse.FloatTensor(indices, values)
    return torch_sparse_mx


def torch_sparse_eye(num_nodes):
    indices = torch.arange(num_nodes).repeat(2, 1)
    values = torch.ones(num_nodes)
    return torch.sparse.FloatTensor(indices, values)

def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.shape[0]): #1546
        for j in range(matrix.shape[1]): #1546
            if matrix[i, j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)


class Sizes(object):
    def __init__(self):
        self.drug_size = 853
        self.mic_size = 591
        self.F1 = 128
        self.F2 = 64
        self.F3 = 32
        self.k_fold = 5
        self.epoch = 10
        self.learn_rate = 0.01
        self.seed = 1
        self.h1_gamma = 2 ** (-5) #0.03125
        self.h2_gamma = 2 ** (-3) #0.125
        self.h3_gamma = 2 ** (-3) #0.125

        self.lambda1 = 2 ** (-3)
        self.lambda2 = 2 ** (-4)
