import numpy as np
import os
# import dependencies
from data_generator import Generator
from model import lGNN_multiclass
from log_definition import Logger
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#Pytorch requirements
import unicodedata
import string
import re
import random
import argparse
import pickle

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from loss import compute_loss_rlx, compute_loss_acc, compute_loss_policy
import pandas

template1 = '{:<10} {:<15} {:<10} {:<10} {:<10} {:<15} {:<15} {:<10} '
template2 = '{:<10} {:<15.4g} {:<10.3f} {:<10.4f} {:<10} {:<15.4g} {:<15} {:<10.3f} \n'
template3 = '{:<10} {:<10} {:<10} {:<10} {:<15} {:<15} {:<10} '
template4 = '{:<10} {:<10.3f} {:<10.4f} {:<10} {:<15} {:<15.3f} {:<10.3f} \n'
template5 = '{:<10} {:<10} {:<10} {:<10} {:<10} {:<11} {:<10} {:<10} {:<10} '
template6 = '{:<10} {:<10} {:<10.2f} {:<10} {:<10.3f} {:<11.12} {:<10.1f} {:<10} {:<10.3f} \n'


def train_single(gnn, optimizer, logger, gen, Lambda, it, args):
    start = time.time()
    WW, x, WW_lg, y, P = gen.sample_batch()
    pred = gnn(WW, x, WW_lg, y, P)
    L = WW[:,:,:,1] - WW[:,:,:,2]
    del WW
    del WW_lg
    del x
    del y
    del P
    if args.loss_method == 'relaxation':
        loss = compute_loss_rlx(pred, args, L, Lambda)
        acc, z, inb = compute_loss_acc(pred, args, L) 
    elif args.loss_method == 'policy':
        loss, acc, z, inb = compute_loss_policy(pred, args, L, Lambda)
    del L
    gnn.zero_grad()
    loss.backward()
    # Clips gradient norm of an iterable of parameters.
    # The norm is computed over all gradients together, as if they were concatenated into a single vector. Gradients are modified in-place.
    nn.utils.clip_grad_norm_(gnn.parameters(), args.clip_grad_norm)
    optimizer.step()
    
    elapsed = time.time() - start

    if(torch.cuda.is_available()):
        loss_value = float(loss.data.cpu().numpy())
        acc_value = float(acc.data.cpu().numpy())
        z_value = float(z.data.cpu().numpy())
        inb_value = float(inb.data.cpu().numpy())
    else:
        loss_value = float(loss.data.numpy())
        acc_value = float(acc.data.numpy())
        z_value = float(z.data.numpy())
        inb_value = float(inb.data.numpy())
    
    info = ['epoch', 'avg loss', 'avg acc', 'avg nacc', 'avg inb', 'Lambda', '(avg)degree', 'elapsed']
    out = [it, loss_value, acc_value, z_value, inb_value, Lambda, int(args.edge_density*args.num_nodes), elapsed]
    print(template1.format(*info))
    print(template2.format(*out))
    return loss_value, acc_value, z_value, inb_value

def train(gnn, logger, gen, args, iters=None):
    if iters is None:
        iters = args.num_examples_train
    gnn.train()
    optimizer = torch.optim.Adamax(gnn.parameters(), lr = args.lr)
    loss_lst = np.zeros([iters])
    acc_lst = np.zeros([iters])
    z_lst = np.zeros([iters])
    inb_lst = np.zeros([iters])
    #lastIDX = iters
    for it in range(iters):
        if args.problem0 == 'Bisection':
            Lambda = args.Lambda * pow((1 + args.LambdaIncRate),it)
        elif args.problem0 == 'Cut':
            Lambda = 0
        else:
            raise ValueError('problem0 {} not supported'
                             .format(args.problem0))
        loss_single, acc_single, z_single, inb_single = train_single(gnn, optimizer, logger, gen, Lambda, it, args)
        loss_lst[it] = loss_single
        acc_lst[it] = acc_single
        z_lst[it] = z_single 
        inb_lst[it] = inb_single
        torch.cuda.empty_cache()
        #if it > 100 and np.sum(inb_lst[(it-20):it]) < 38:
        #    lastIDX = it
        #    break
        if (it % 100 == 0) and (it >= 100):
            # print ('Testing at check_pt begins')
            print ('Check_pt at iteration ' + str(it) + ' :')
            c1 = np.mean(loss_lst[it-100:it])
            print ('Avg train loss {:.4f}'.format(c1))
            c2 = np.mean(acc_lst[it-100:it])
            print ('Avg train acc {:.4f}'.format(c2))
            c3 = np.std(acc_lst[it-100:it])
            print ('Std train acc {:.4f}'.format(c3))
            c4 = np.mean(z_lst[it-100:it])
            print ('Avg train nacc {:.4f}'.format(c4))
            c5 = np.mean(inb_lst[it-100:it])
            print ('Avg train inbalance {:.1f}'.format(c5))
    #loss_lst = loss_lst[0:lastIDX]
    #acc_lst = acc_lst[0:lastIDX]
    #z_lst = z_lst[0:lastIDX]
    #inb_lst = inb_lst[0:lastIDX]
    c1 = np.mean(loss_lst)
    print ('Final avg train loss {:.4f}'.format(c1))
    c2 = np.mean(acc_lst)
    print ('Final avg train acc {:.4f}'.format(c2))
    c3 = np.std(acc_lst)
    print ('Final std train acc {:.4f}'.format(c3))
    c4 = np.mean(z_lst)
    print ('Final avg train nacc {:.4f}'.format(c4))
    c5 = np.mean(inb_lst)
    print ('Final avg train inbalance {:.1f}'.format(c5))
    #return loss_lst, acc_lst, z_lst, inb_lst, lastIDX

def test_single(gnn, logger, gen, it, args):
    start = time.time()
    random.seed(it)
    WW, x, WW_lg, y, P = gen.sample_batch()
    pred = gnn(WW, x, WW_lg, y, P)
    L = WW[:,:,:,1] - WW[:,:,:,2]
    del WW
    del WW_lg
    del x
    del y
    del P
    acc, z, inb, label = compute_loss_acc(pred, args, L)  
    del L

    elapsed = time.time() - start
    if (torch.cuda.is_available()):
        acc_value = float(acc.data.cpu().numpy())
        z_value = float(z.data.cpu().numpy())
        inb_value = float(inb.data.cpu().numpy())
        label_value = label.data.cpu().numpy()
    else:
        acc_value = float(acc.data.numpy())
        z_value = float(z.data.numpy())
        inb_value = float(inb.data.numpy())
        label_value = label.data.numpy()

    info = ['epoch', 'avg acc', 'avg nacc', 'avg inb', 'num_nodes', 'edge_den', 'elapsed']
    out = [it, acc_value, z_value, inb_value, args.num_nodes, args.edge_density, elapsed]
    print(template3.format(*info))
    print(template4.format(*out))
    return acc_value, z_value, inb_value, label_value

def test(gnn, logger, gen, args, iters=None):
    if iters is None:
        iters = args.num_examples_test
    gnn.train()
    acc_lst = np.zeros([iters])
    z_lst = np.zeros([iters])
    inb_lst = np.zeros([iters])
    label_lst = [[]]*iters
    for it in range(iters):
        acc_single, z_single, inb_single, label_single = test_single(gnn, logger, gen, it, args)
        acc_lst[it] = acc_single
        z_lst[it] = z_single
        inb_lst[it] = inb_single
        label_lst[it] = label_single
        torch.cuda.empty_cache()
    print ('Testing results:')
    c1 = np.mean(acc_lst)
    print ('Final avg test acc {:.4f}'.format(c1))
    c2 = np.std(acc_lst)
    print ('Final std test acc {:.4f}'.format(c2))
    c3 = np.mean(z_lst)
    print ('Final avg test nacc {:.4f}'.format(c3))
    c4 = np.std(z_lst)
    print ('Final std test nacc {:.4f}'.format(c4))
    c5 = np.mean(inb_lst)
    print ('Final avg test inbalance {:.1f}'.format(c5))
    return acc_lst, z_lst, inb_lst, label_lst

def read_args_commandline():
    parser = argparse.ArgumentParser()
    # Parser for command-line options, arguments and subcommands
    # The argparse module makes it easy to write user-friendly command-line interfaces.
    
    ###############################################################################
    #                             General Settings                                #
    ###############################################################################
    parser.add_argument('--num_examples_train', nargs='?', const=1, type=int,
                        default=int(1000))
    parser.add_argument('--loss_method', nargs='?', const=1, type=str,
                        default='relaxation')
    parser.add_argument('--num_examples_test', nargs='?', const=1, type=int,
                        default=int(1000))
    parser.add_argument('--edge_density', nargs='?', const=1, type=float,
                        default=0.5)
    parser.add_argument('--generative_model', nargs='?', const=1, type=str,
                        default='ErdosRenyi')
    parser.add_argument('--num_nodes', nargs='?', const=1, type=int,
                        default=50)
    parser.add_argument('--batch_size', nargs='?', const=1, type=int, default=1)
    parser.add_argument('--mode', nargs='?', const=1, type=str, default='train')
    parser.add_argument('--path_output', nargs='?', const=1, type=str, default='')
    parser.add_argument('--path_logger', nargs='?', const=1, type=str, default='')
    parser.add_argument('--path_gnn', nargs='?', const=1, type=str, default='')
    parser.add_argument('--filename_existing_gnn', nargs='?', const=1, type=str, default='')
    parser.add_argument('--filename_test_segm', nargs='?', const=1, type=str, default='')
    parser.add_argument('--print_freq', nargs='?', const=1, type=int, default=100)
    parser.add_argument('--test_freq', nargs='?', const=1, type=int, default=500)
    parser.add_argument('--save_freq', nargs='?', const=1, type=int, default=2000)
    parser.add_argument('--clip_grad_norm', nargs='?', const=1, type=float, default=40.0)
    parser.add_argument('--Lambda', nargs='?', const=1, type=float, default=10)
    parser.add_argument('--LambdaIncRate', nargs='?', const=1, type=float, default=0.05)
    parser.add_argument('--num_ysampling', nargs='?', const=1, type=int, default=10000)
    parser.add_argument('--problem', nargs='?', const=1, type=str, default='max')
    parser.add_argument('--problem0', nargs='?', const=1, type=str, default='Cut')
    ###############################################################################
    #                                 GNN Settings                                #
    ###############################################################################

    parser.add_argument('--num_features', nargs='?', const=1, type=int,
                        default=20)
    parser.add_argument('--num_layers', nargs='?', const=1, type=int,
                        default=20)
    parser.add_argument('--num_classes', nargs='?', const=1, type=int,
                        default=2)
    parser.add_argument('--J', nargs='?', const=1, type=int, default=4)
    parser.add_argument('--lr', nargs='?', const=1, type=float, default=1e-3)

    return parser.parse_args()


def main():
    args = read_args_commandline()
    
    logger = Logger(args.path_logger)
    logger.write_settings(args)

    gen = Generator(args)
    torch.backends.cudnn.enabled=False
    
    if (args.mode == 'test'):
        print ('In testing mode')
        # filename = 'gnn_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_Ntr' + str(gen.N_test) + '_it' + str(args.iterations)
        filename = args.filename_existing_gnn
        path_plus_name = os.path.join(args.path_gnn, filename)
        if ((filename != '') and (os.path.exists(path_plus_name))):
            print ('Loading gnn ' + filename)
            gnn = torch.load(path_plus_name)
            if torch.cuda.is_available():
                gnn.cuda()
            acc, z, inb, label = test(gnn, logger, gen, args, iters = None)
            res = {
                'acc': acc,
                'nacc': z,
                'inb': inb,
                'label': label
            }
            path_plus_name = os.path.join(args.path_output, args.filename_test_segm)
            print ('Saving acc, nacc, inb...' + args.filename_test_segm)
            with open(path_plus_name, 'wb') as f:
                pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
        else:
            print ('No such a gnn exists; please first create one')

    elif (args.mode == 'train'):           
        print ('Creating the gnn ...')
        if args.loss_method == 'policy':
            filename = 'lgnn_' + str(args.problem) + str(args.problem0) + '_plc' + str(args.num_ysampling) + '_N' + str(args.num_nodes) + '_p' + str(args.edge_density) + '_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_ftr' + str(args.num_features) + '_Lbd' + str(args.Lambda) + '_LbdR' + str(args.LambdaIncRate) + '_lr' + str(args.lr)
        else:
            filename = 'lgnn_' + str(args.problem) + str(args.problem0) + '_N' + str(args.num_nodes) + '_p' + str(args.edge_density) + '_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_ftr' + str(args.num_features) + '_Lbd' + str(args.Lambda) + '_LbdR' + str(args.LambdaIncRate) + '_lr' + str(args.lr)

        path_plus_name = os.path.join(args.path_gnn, filename)
        gnn = lGNN_multiclass(args.num_features, args.num_layers, args.J + 2, args.num_classes)
    
        if torch.cuda.is_available():
            gnn.cuda()
        print ('Training begins')
        train(gnn, logger, gen, args, iters = None)
        print ('Saving gnn ' + filename)
        if torch.cuda.is_available():
            torch.save(gnn.cpu(), path_plus_name)
            gnn.cuda()
        else:
            torch.save(gnn, path_plus_name)
        #res = {
        #    'loss': loss,
        #    'acc': acc,
        #    'nacc': z,
        #    'inb': inb,
        #    'lastIDX': lastIDX
        #}
        #if args.loss_method == 'policy':
        #    resname = 'segm_' + str(args.problem) + str(args.problem0) + '_plc' + str(args.num_ysampling) + '_N' + str(args.num_nodes) + '_p' + str(args.edge_density) + '_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_ftr' + str(args.num_features) + '_Lbd' + str(args.Lambda) + '_LbdR' + str(args.LambdaIncRate) + '_lr' + str(args.lr) + '.pickle'
        #else:
        #    resname = 'segm_' + str(args.problem) + str(args.problem0) + '_N' + str(args.num_nodes) + '_p' + str(args.edge_density) + '_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_ftr' + str(args.num_features) + '_Lbd' + str(args.Lambda) + '_LbdR' + str(args.LambdaIncRate) + '_lr' + str(args.lr) + '.pickle'
        #path_plus_name = os.path.join(args.path_output, resname)
        #print ('Saving loss, acc, nacc, inb, lastIDX...' + resname)
        #with open(path_plus_name, 'wb') as f:
        #    pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()

