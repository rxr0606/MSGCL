import numpy as np
import scipy.sparse as sp
import random
import gc
import argparse
from clac_metric import get_metrics
from utils import get_edge_index,Sizes
import torch as t
from torch import optim
import dgl
from utils import *
import copy
from model import GCN, GCL,prediction_model
import torch.nn.functional as F
import torch.nn as nn
from graph_learners import *
from load_data import *
from sklearn.cluster import KMeans
from loss import Myloss
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

class Experiment:
    def __init__(self):
        super(Experiment, self).__init__()


    def setup_seed(self, seed):
        t.manual_seed(seed)
        t.cuda.manual_seed_all(seed)
        t.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)
        dgl.seed(seed)
        dgl.random.seed(seed)

   
    def loss_cls(self, model,features,train_matrix,regression_crit):
        
        drug_mic_res=model(features) #1444*1444
        loss1 = nn.MSELoss()
        loss = loss1(drug_mic_res,train_matrix)
        # loss = regression_crit(train_matrix.cpu(), drug_mic_res, model.drug_l, model.mic_l, model.alpha1,
        #                        model.alpha2, sizes) #drug_mic_res 853*591 train_matix:1444*1444
        # model.alpha1 = t.mm(
        #     t.mm((t.mm(model.drug_k, model.drug_k) + model.lambda1 * model.drug_l).inverse(), model.drug_k),
        #     2 * train_matrix.cpu() - t.mm(model.alpha2.T, model.mic_k.T)).detach() #公式19 令损失为0 解得αd
        # model.alpha2 = t.mm(t.mm((t.mm(model.mic_k, model.mic_k) + model.lambda2 * model.mic_l).inverse(), model.mic_k),
        #                     2 * train_matrix.cpu().T - t.mm(model.alpha1.T, model.drug_k.T)).detach() #公式20 αm
        #print(loss)
        return loss,drug_mic_res


    def loss_gcl(self, model, graph_learner, features, anchor_adj):

       
        if args.maskfeat_rate_anchor:
            mask_v1, _ = get_feat_mask(features, args.maskfeat_rate_anchor)
            features_v1 = features * (1 - mask_v1)
        else:
            features_v1 = copy.deepcopy(features) #深拷贝
        
        z1, _ = model(features_v1, anchor_adj, 'anchor')  

        
        if args.maskfeat_rate_learner:
            mask, _ = get_feat_mask(features, args.maskfeat_rate_learner) 
            features_v2 = features * (1 - mask) 
        else:
            features_v2 = copy.deepcopy(features)

        learned_adj = graph_learner(features) 
        if not args.sparse:
            learned_adj = symmetrize(learned_adj)
            learned_adj = normalize(learned_adj, 'sym', args.sparse) 

        z2, _ = model(features_v2, learned_adj, 'learner')  

        # compute loss
        loss = model.calc_loss(z1, z2) 

        return loss, learned_adj

    
    def evaluate_adj_by_cls(self, Adj, features, nfeats,args,train_matrix,out1,drug_sim,drug_fea,mic_sim,mic_fea):
        model = GCN(in_channels=nfeats, hidden_channels=args.hidden_dim_cls, out_channels=out1, num_layers=args.nlayers_cls,
                    dropout=args.dropout_cls, dropout_adj=args.dropedge_cls, Adj=Adj, sparse=args.sparse,drug_sim=drug_sim,drug_fea=drug_fea,mic_sim=mic_sim,mic_fea=mic_fea)

        optimizer = t.optim.Adam(model.parameters(), lr=args.lr_cls, weight_decay=args.w_decay_cls)

        

        if t.cuda.is_available():
            model = model.cuda()
            features = features.cuda()
            #train_matrix=t.FloatTensor(train_matrix)

        
        for epoch in range(1, args.epochs_cls + 1): #1-201
            model.train()
            regression_crit = Myloss()
            
            loss,drug_mic_res= self.loss_cls(model, features,train_matrix,regression_crit)
            optimizer.zero_grad()
            #loss = loss.requires_grad_()
            loss.backward()
            optimizer.step()
           
        return drug_mic_res 

    
    def train(self, args,sizes):

        t.cuda.set_device(args.gpu)
        
        if args.gsl_mode == 'structure_refinement': 
            features, nfeats, drug_mic_matrix,adj_original,drug_sim,drug_fea,mic_sim,mic_fea = load_data(args)
        elif args.gsl_mode == 'structure_inference':
            features, nfeats, drug_mic_matrix, _ ,drug_sim,drug_fea,mic_sim,mic_fea= load_data(args)

        if args.downstream_task == 'linkpre':
            AUPR_accuracies = []
            AUC_accuracies = []
            ACC_accuracies=[]

        
        index = crossval_index(drug_mic_matrix, sizes)  
        metric = np.zeros((1, 7)) 
        pre_matrix = np.zeros(drug_mic_matrix.shape)
        print("seed=%d, evaluating drug-microbe...." % (sizes.seed))
        for k in range(args.k_fold):
            print("------this is %dth cross validation------" % (k + 1))
            train_matrix = np.matrix(drug_mic_matrix, copy=True) 
            train_matrix[tuple(np.array(index[k]).T)] = 0 
            self.setup_seed(k)

            if args.gsl_mode == 'structure_inference':
                if args.sparse:
                    anchor_adj_raw = torch_sparse_eye(features.shape[0])
                else:
                    anchor_adj_raw = t.eye(features.shape[0]) 
            elif args.gsl_mode == 'structure_refinement':
                if args.sparse:
                    anchor_adj_raw = adj_original
                else:
                    anchor_adj_raw = t.from_numpy(adj_original) 

            anchor_adj = normalize(anchor_adj_raw, 'sym', args.sparse) 
            anchor_adj = anchor_adj.float()
            if args.sparse:
                anchor_adj_torch_sparse = copy.deepcopy(anchor_adj)
                anchor_adj = torch_sparse_to_dgl_graph(anchor_adj)

            
            if args.type_learner == 'fgp':
                graph_learner = FGP_learner(features.cpu(), args.k, args.sim_function, 6, args.sparse) 
            elif args.type_learner == 'mlp':
                graph_learner = MLP_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                     args.activation_learner)
            elif args.type_learner == 'att':
                graph_learner = ATT_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                          args.activation_learner)
            elif args.type_learner == 'gnn':
                graph_learner = GNN_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                     args.activation_learner, anchor_adj)
           
            model = GCL(nlayers=args.nlayers, in_dim=nfeats, hidden_dim=args.hidden_dim,
                         emb_dim=args.rep_dim, proj_dim=args.proj_dim,
                         dropout=args.dropout, dropout_adj=args.dropedge_rate, sparse=args.sparse)

           
            optimizer_cl = t.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
            optimizer_learner = t.optim.Adam(graph_learner.parameters(), lr=args.lr, weight_decay=args.w_decay)


            if t.cuda.is_available():
               
                model = model.cuda()
                graph_learner = graph_learner.cuda()
                features = features.cuda()
                train_matrix = t.FloatTensor(train_matrix).cuda()
                if not args.sparse:
                    anchor_adj = anchor_adj.cuda() #Aa
               

            
            for epoch in range(1, args.epochs + 1):

                model.train()
                graph_learner.train()
                
                loss, Adj = self.loss_gcl(model, graph_learner, features, anchor_adj)

                optimizer_cl.zero_grad()
                optimizer_learner.zero_grad()
                loss.backward()
                optimizer_cl.step()
                optimizer_learner.step()

               
                if (1 - args.tau) and (args.c == 0 or epoch % args.c == 0):
                    anchor_adj = anchor_adj * args.tau + Adj.detach() * (1 - args.tau) 

                print("Epoch {:05d} | CL Loss {:.4f}".format(epoch, loss.item()))
               
                if epoch % args.epochs == 0:
                    if args.downstream_task == 'linkpre':
                        model.eval() 
                        graph_learner.eval()
                        f_adj = Adj 
                        f_adj = f_adj.detach() 
                        out1=train_matrix.shape[1] 
                        drug_mic_res= self.evaluate_adj_by_cls(f_adj, features, nfeats,args,train_matrix,out1,drug_sim,drug_fea,mic_sim,mic_fea)

            
            predict_y_proba = drug_mic_res.reshape(sizes.drug_size+sizes.mic_size,sizes.drug_size+sizes.mic_size).cpu().detach().numpy() 
            pre_matrix[tuple(np.array(index[k]).T)] = predict_y_proba[tuple(np.array(index[k]).T)]
            real_score = drug_mic_matrix[tuple(np.array(index[k]).T)]
            metric_tmp = get_metrics(real_score,predict_y_proba[tuple(np.array(index[k]).T)]) 
            print(metric_tmp)
            metric += metric_tmp 
        if args.downstream_task == 'linkpre' and k == 4: 
            print(metric / sizes.k_fold)

    
        

def random_index(index_matrix, sizes):
    association_nam = index_matrix.shape[1] 
    random_index = index_matrix.T.tolist() 
    random.seed(sizes.seed)
    random.shuffle(random_index)
    k_folds = args.k_fold #5折
    CV_size = int(association_nam / k_folds) 
    temp = np.array(random_index[:association_nam - association_nam %
                                  k_folds]).reshape(k_folds, CV_size, -1).tolist() 

    temp[k_folds - 1] = temp[k_folds - 1] + \
                        random_index[association_nam - association_nam % k_folds:]
    return temp

def crossval_index(drug_mic_matrix, sizes):
    random.seed(sizes.seed)
    pos_index_matrix = np.mat(np.where(drug_mic_matrix == 1)) #np.where（输出每个元素值为1的坐标）
    neg_index_matrix = np.mat(np.where(drug_mic_matrix == 0)) 
    pos_index = random_index(neg_index_matrix, sizes)
    neg_index = random_index(pos_index_matrix, sizes)
    index = [pos_index[i] + neg_index[i] for i in range(args.k_fold)] #两个列表按行拼接起来 
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experimental setting
    parser.add_argument('-dataset', type=str, default='MDAD',
                        choices=['MDAD', 'aBiofilm', 'DrugVirus'])
    parser.add_argument('-k_fold', type=int, default=5)
    parser.add_argument('-sparse', type=int, default=0)
    parser.add_argument('-gsl_mode', type=str, default="structure_inference",
                        choices=['structure_inference', 'structure_refinement'])
    parser.add_argument('-eval_freq', type=int, default=5)
    parser.add_argument('-downstream_task', type=str, default='linkpre',
                        choices=['linkpre', 'classification'])
    parser.add_argument('-gpu', type=int, default=0)

    # GCL Module - Framework
    parser.add_argument('-epochs', type=int, default=1000)
    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('-w_decay', type=float, default=0.0)
    parser.add_argument('-hidden_dim', type=int, default=512)
    parser.add_argument('-rep_dim', type=int, default=64)
    parser.add_argument('-proj_dim', type=int, default=64)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-contrast_batch_size', type=int, default=0)
    parser.add_argument('-nlayers', type=int, default=2)

    # GCL Module -Augmentation
    parser.add_argument('-maskfeat_rate_learner', type=float, default=0.2)
    parser.add_argument('-maskfeat_rate_anchor', type=float, default=0.2)
    parser.add_argument('-dropedge_rate', type=float, default=0.5)

    # GSL Module
    parser.add_argument('-type_learner', type=str, default='fgp', choices=["fgp", "att", "mlp", "gnn"])
    parser.add_argument('-k', type=int, default=30)
    parser.add_argument('-sim_function', type=str, default='cosine', choices=['cosine', 'minkowski'])
    parser.add_argument('-gamma', type=float, default=0.9)
    parser.add_argument('-activation_learner', type=str, default='relu', choices=["relu", "tanh"])

    # Evaluation Network (Classification)
    parser.add_argument('-epochs_cls', type=int, default=200)
    parser.add_argument('-lr_cls', type=float, default=0.001)
    parser.add_argument('-w_decay_cls', type=float, default=0.0005)
    parser.add_argument('-hidden_dim_cls', type=int, default=32)
    parser.add_argument('-dropout_cls', type=float, default=0.5)
    parser.add_argument('-dropedge_cls', type=float, default=0.25)
    parser.add_argument('-nlayers_cls', type=int, default=2)
    parser.add_argument('-patience_cls', type=int, default=10)

    # Structure Bootstrapping
    parser.add_argument('-tau', type=float, default=1)
    parser.add_argument('-c', type=int, default=0)

    args = parser.parse_args()
    
    experiment = Experiment()
    sizes = Sizes()
    experiment.train(args,sizes)
    
    
