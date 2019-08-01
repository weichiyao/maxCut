import numpy as np
import math
import os
# import dependencies
import time

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
from torch.distributions.categorical import Categorical 

softmax = nn.Softmax(dim = -1) # pred of size bs x N x 2 or N x 2
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor

else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor

def compute_loss_rlx(pred, args, L, Lambda):
    L = L.type(dtype)
    pred_prob = softmax(pred).type(dtype) # pred of size bs x N x 2
    pp = pred_prob[:, :, 0] # pp of size bs x N
    yy = 2*pp-1
    c = 1/4 * torch.bmm(yy.unsqueeze(-2), torch.bmm(L, yy.unsqueeze(-1))) 

    if args.problem == 'max':
        loss = torch.mean(- c.view([args.batch_size]) + Lambda * (pp.sum(dim = -1) - args.num_nodes/2).pow(2))
    else:
        loss = torch.mean(c.view([args.batch_size]) + Lambda * (pp.sum(dim = -1) - args.num_nodes/2).pow(2))
    return loss

def compute_loss_acc(pred, args, L):  
    L = L.type(dtype)
    d = int(args.num_nodes * args.edge_density)
    labels = torch.argmax(pred, dim = -1).type(dtype) * 2 - 1 # of size bs x N
    acc = torch.mean(1/4 * torch.bmm(labels.unsqueeze(-2), torch.bmm(L, labels.unsqueeze(-1))))
    z = (acc/args.num_nodes - d/4)/np.sqrt(d/4)
    inb = torch.abs(torch.mean(torch.abs(labels.sum(dim = -1))))
    return acc, z, inb, labels.squeeze()

def compute_loss_policy(pred, args, L, Lambda): 
    L = L.type(dtype)
    pred_prob = softmax(pred).type(dtype)  # pred of size bs x N x 2
    d = int(args.num_nodes * args.edge_density)
    if args.batch_size == 1:
        m = Categorical(pred_prob[0,:,:])
        y_sampled = m.sample((args.num_ysampling,)).type(dtype)
        #  y of size: args.num_ysampling x N
        pred_prob_sampled_log = m.log_prob(y_sampled).type(dtype)
        # of size: args.num_ysampling x N
        pred_prob_sampled_sum_log = pred_prob_sampled_log.sum(dim = -1) 
        # of size args.num_ysampling
        y_sampled_label = y_sampled * 2 - 1
        #  y of size: args.num_ysampling x N
        L = L.squeeze(0).type(dtype)
        #  L of size: N x N
        c = torch.mm(y_sampled_label, torch.mm(L, torch.t(y_sampled_label)))
        c = 1/4 * torch.diagonal(c, offset = 0) 
        # c of size args.num_ysampling
        if args.problem == 'max':
            c_plus_penalty = - c + Lambda * y_sampled_label.sum(dim = 1).pow(2)
        else:
            c_plus_penalty = c + Lambda * y_sampled_label.sum(dim = 1).pow(2)             
        loss = pred_prob_sampled_sum_log.dot(c_plus_penalty)
        w = torch.exp(pred_prob_sampled_sum_log)/torch.exp(pred_prob_sampled_sum_log).sum(dim = -1)
        acc = w.dot(c)
        z = (acc/args.num_nodes - d/4)/np.sqrt(d/4)
        inb = torch.dot(torch.abs(y_sampled_label.sum(dim = 1)), w)
    else:
        m = Categorical(pred_prob)
        y_sampled = m.sample((args.num_ysampling,)).type(dtype)
        # y_sampled of size: args.num_ysampling x bs x N
        pred_prob_sampled_log = m.log_prob(y_sampled) 
        # of size: args.num_ysampling x bs x N
        y_sampled = y_sampled.permute(1,2,0)
        # y_sampled of size: bs x N x args.num_ysampling
        pred_prob_sampled_sum_log = pred_prob_sampled_log.sum(dim = -1).permute(1,0)
        # of size args.num_ysampling x bs -> bs x args.num_ysampling
        y_sampled_label = y_sampled * 2 - 1
        c = torch.bmm(y_sampled_label.permute(0,2,1), torch.bmm(L, y_sampled_label))
        # c of size bs x args.num_ysampling x args.num_ysampling
        c = 1/4 * torch.diagonal(c, offset = 0, dim1 = -2, dim2 = -1) 
        c_plus_penalty = c + Lambda * y_sampled_label.sum(dim = 1).pow(2)
        # c_plus_penalty of size bs x args.num_ysampling
        loss = torch.bmm(c_plus_penalty.view([args.batch_size, 1, args.num_ysampling]), pred_prob_sampled_sum_log.view([args.batch_size, args.num_ysampling, 1]))
        # loss of size bs
        loss = torch.mean(loss)
        w = torch.exp(pred_prob_sampled_sum_log)/torch.exp(pred_prob_sampled_sum_log).sum(dim = -1).view([args.batch_size, 1])
        acc = torch.dot(c.view([args.batch_size, 1, args.num_ysampling]), w.view([args.batch_size, args.num_ysampling, 1]))
        inb = torch.dot(torch.abs(y_sampled_label.sum(dim=1)).view([args.batch_size, 1, args.num_ysampling]), w.view([args.batch_size, args.num_ysampling, 1]))
        acc = torch.mean(acc)
        z = (acc/args.num_nodes - d/4)/np.sqrt(d/4)
        inb = torch.mean(inb)
    inb = torch.round(inb)
    return loss, acc, z, inb




    




 

