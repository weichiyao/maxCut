import numpy as np
# os module provides dozens of functions for interacting with the operating system
import os
# import dependencies
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import networkx
import pickle
import scipy.io

#Pytorch requirements
import unicodedata
import string
import re
import random
import argparse


import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

class Generator(object):
    def __init__(self, args = None):
        if args is None:
            self.p = 0.5
            self.N = 50
            self.J = 3
            self.generative_model = "ErdosRenyi"
            self.bs = 1
            self.path_output = '/home/jss2/wy635/Graph/GraphGNN/myfile/temp/output/'
        else:
            self.p = args.edge_density
            self.N = args.num_nodes
            self.J = args.J
            self.generative_model = args.generative_model
            self.bs = args.batch_size
            self.path_output = args.path_output


        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
            self.dtype_l = torch.cuda.LongTensor

        else:
            self.dtype = torch.FloatTensor
            self.dtype_l = torch.LongTensor

    def ErdosRenyi(self):
        g = networkx.erdos_renyi_graph(self.N, self.p)
        W = networkx.adjacency_matrix(g).todense()
        W = torch.tensor(W).type(self.dtype)
        return W

    def RegularGraph(self):
        """ Generate random regular graph """
        d = self.p * self.N # with input p
        d = int(d)
        g = networkx.random_regular_graph(d, self.N)
        W = networkx.adjacency_matrix(g).todense()
        W = torch.tensor(W).type(self.dtype)
        return W

    def get_operators(self, W):
        # operators: {Id, D, W, W^2, ..., W^{J-1}}
        W = W.type(self.dtype)
        n = W.shape[0]
        d = W.sum(1)
        D = torch.diag(d)
        D = D.type(self.dtype)
    
        Wnew = torch.zeros([n, n, self.J]).type(self.dtype)
        for j in range(self.J):
            Wnew[:, :, j] = W
            W = torch.min(torch.mm(W, W), torch.ones([n, n]).type(self.dtype))
            W = W.type(self.dtype)
        
        WWtemp = torch.stack((torch.eye(n).type(self.dtype), D), dim = -1)
        WWtemp = WWtemp.type(self.dtype)
        WWres = torch.cat((WWtemp, Wnew), dim = -1)
        WWres = WWres.type(self.dtype)
        x = torch.reshape(d, [n, 1])
        x = x.type(self.dtype)
        return WWres, x

    def get_Pm(self, W):
        W = W.type(self.dtype)
        n = W.shape[0]
        W = W * (torch.ones([n, n]).type(self.dtype) - torch.eye(n).type(self.dtype))
        M = int(W.sum()) // 2
        p = 0
        Pm = torch.zeros([n, M * 2]).type(self.dtype)
        for i in range(n):
            for j in range(i+1, n):
                if (W[i][j]==1):
                    Pm[i][p] = 1
                    Pm[j][p] = 1
                    Pm[i][p + M] = 1
                    Pm[j][p + M] = 1
                    p += 1
        Pm = Pm.type(self.dtype)
        return Pm

    def get_Pd(self, W):
        W = W.type(self.dtype)
        n = W.shape[0]
        W = W * (torch.ones([n, n]).type(self.dtype) - torch.eye(n).type(self.dtype))
        M = int(W.sum()) // 2
        p = 0
        Pd = torch.zeros([n, M * 2]).type(self.dtype)
        for i in range(n):
            for j in range(i+1, n):
                if (W[i][j]==1):
                    Pd[i][p] = 1
                    Pd[j][p] = -1
                    Pd[i][p + M] = -1
                    Pd[j][p + M] = 1
                    p += 1
        Pd = Pd.type(self.dtype)
        return Pd

    def get_NB_2(self, W):
        Pm = self.get_Pm(W)
        Pd = self.get_Pd(W)
        Pf = (Pm + Pd) / 2
        Pt = (Pm - Pd) / 2
        NB = torch.mm(torch.transpose(Pt,0,1),Pf) * (1-torch.mm(torch.transpose(Pf,0,1),Pt))
        return NB


    def get_P(self, W):
        W = W.type(self.dtype)
        P = torch.stack((self.get_Pm(W), self.get_Pd(W)), dim = -1)
        P = P.type(self.dtype)
        return P
    
    def compute_sample_i(self):
        sample_i = {}
        if self.generative_model == 'ErdosRenyi':
            W = self.ErdosRenyi()
        elif self.generative_model == 'RegularGraph':
            W = self.RegularGraph()
        else:
            raise ValueError('Generative model {} not supported'
                             .format(self.generative_model))

        WW, x = self.get_operators(W)
        W_lg = self.get_NB_2(W)
        WW_lg, y = self.get_operators(W_lg)
        P = self.get_P(W)

        sample_i['WW'], sample_i['x'] = WW, x
        sample_i['WW_lg'], sample_i['y'] = WW_lg, y
        sample_i['P'] = P
        return sample_i

    # def load_dataset(self):
    #     # load train dataset
    #     filename = 'BIStrain_{}' + str(self.generative_model) + '_N' + str(self.N) + '_p' + str(self.p) + '_J' + str(self.J) + '_bs' + str(self.bs) + '.pickle'
    #     path_plus_name = os.path.join(self.path_dataset, filename)
        
    #     if os.path.exists(path_plus_name):
    #         print('Reading training dataset at {}'.format(path_plus_name))
    #         with open(path_plus_name, 'rb') as f:
    #             self.data_train = pickle.load(f)
    #     else:
    #         print('Creating training dataset.')
    #         self.create_dataset_train()
    #         print('Saving training datatset at {}'.format(path_plus_name))
    #         with open(path_plus_name, 'wb') as f:
    #             pickle.dump(self.data_train, f, pickle.HIGHEST_PROTOCOL)
    #     # load test dataset
     
    #     filename = 'BIStest_{}' + str(self.generative_model) + '_N' + str(self.N) + '_p' + str(self.p) + '_J' + str(self.J) + '_bs' + str(self.bs) + '.pickle'
    #     path_plus_name = os.path.join(self.path_dataset, filename)
    #     if os.path.exists(path_plus_name):
    #         print('Reading testing dataset at {}'.format(path_plus_name))
    #         self.data_test = np.load(open(path_plus_name, 'rb'))
    #     else:
    #         print('Creating testing dataset.')
    #         self.create_dataset_test()
    #         print('Saving testing datatset at {}'.format(path))
    #         np.save(open(path_plus_name, 'wb'), self.data_test)


    def sample_batch(self):
        # generate a list of bs elements 
        batch_i = [self.compute_sample_i() for _ in range(self.bs)]
        WW = torch.stack([element['WW'] for element in batch_i])
        x = torch.stack([element['x'] for element in batch_i])
        WW_lg = torch.stack([element['WW_lg'] for element in batch_i])
        y = torch.stack([element['y'] for element in batch_i])
        P = torch.stack([element['P'] for element in batch_i])
        return WW, x, WW_lg, y, P

if __name__ == '__main__':
    # execute only if run as a script
    ################### Test graph generators ########################
    gen = Generator()
    WW, x, WW_lg, y, P = gen.sample_batch()
    L = WW[:,:,:,1] - WW[:,:,:,2]
    resname = 'testdata_' + str(gen.generative_model) + '_N' + str(gen.N) + '_p' + str(gen.p) + '_num' + str(gen.bs)
    path_plus_name = os.path.join(gen.path_output, resname)


           