import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import ConvexHull

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from loss import compute_loss_rlx, compute_loss_acc

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    # torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor
    # torch.manual_seed(0)


class Logger(object):
    def __init__(self, path_logger):
        directory = os.path.join(path_logger, 'plots/')
        self.path = path_logger
        self.path_dir = directory
        # Create directory if necessary
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)
        self.args = None
    
    def write_settings(self, args):
        self.args = {}
        # write info
        path = os.path.join(self.path, 'experiment.txt')
        with open(path, 'w') as file:
            for arg in vars(args):
                file.write(str(arg) + ' : ' + str(getattr(args, arg)) + '\n')
                self.args[str(arg)] = getattr(args, arg)

    def save_model(self, model):
        save_dir = os.path.join(self.path, 'parameters/')
        # Create directory if necessary
        try:
            os.stat(save_dir)
        except:
            os.mkdir(save_dir)
        path = os.path.join(save_dir, 'gnn.pt')
        torch.save(model, path)
        print('Model Saved.')

    def load_model(self):
        load_dir = os.path.join(self.path, 'parameters/')
        # check if any training has been done before.
        try:
            os.stat(load_dir)
        except:
            print("Training has not been done before testing. This session will be terminated.")
            sys.exit()
        path = os.path.join(load_dir, 'gnn.pt')
        print('Loading the most recent model...')
        gnn = torch.load(path)
        return gnn