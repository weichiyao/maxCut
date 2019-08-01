import re
import heapq
import numpy as np
import random 
import scipy.stats as stats

class EOCut:
    """
    EOCut is a class for extremal optimization algorithm applying on cut problem.
    """
    def __init__(self, problem, W, tau, t_max, matrix):
        self.problem = problem # -1 => 'max', 1 => 'min'
        self.W = W # adjacency matrix
        self.n = W.shape[0]
        self.fitness = np.zeros(W.shape[0])
        self.tau = tau
        self.t_max = t_max  ## TODO: customize const 1 << A << n
        self.matrix = matrix # 'Adjacency' or 'Wigner'

    def node_kthSmallest(self, inputarr, k): 
        # idx = np.argpartition(self.fitness, k)
        # return idx[k-1]
        heapq.heapify(inputarr.tolist())    
        # get list of first k smallest element because 
        # nsmallest(k,list) method returns first k  
        # smallest element now print last element of  
        # that list 
        kSmallest = heapq.nsmallest(k, inputarr.tolist())[-1]
        nodeID_kthSmallest = random.choice(np.where(inputarr == kSmallest)[0])
        return nodeID_kthSmallest



    def truncated_power_law(self, m):
        """
        Generates the distribution according to a power law controlled by
        parameter tau.
        """

        x = np.arange(1, m + 1, dtype='float')
        pmf = 1 / (x ** self.tau)
        pmf /= pmf.sum()
        return stats.rv_discrete(values=(range(1, m + 1), pmf))

    def select_rank(self):
        """
        Randomly select the rank with a power law chance.
        """

        distribution = self.truncated_power_law(self.n)
        k = distribution.rvs()
        return k 


    def evaluate_fitness(self, nodeID, label):
        if self.matrix is 'Adjacency':
            for i in nodeID:
                nbr_arr = self.nbrs[i]
                if nbr_arr.size < 1:
                    self.fitness[i] = self.problem * 1
                else:
                    good = np.sum(label[nbr_arr] == label[i])
                    self.fitness[i] = self.problem * good / label[nbr_arr].size
        elif self.matrix is 'Wigner':
            for i in nodeID:
                self.fitness[i] = - self.problem * label[i] * (self.W[i,:].dot(label))
        else:
            print('Wrong matrix is given')

    def find_nbrs(self):
        # n,_ = self.W.shape
        self.nbrs = np.empty(self.n, dtype = object)
        if self.matrix is 'Adjacency':
            for i in range(self.n):
                self.nbrs[i] = np.where(self.W[i,] == 1)[0]
        elif self.matrix is 'Wigner':
            for i in range(self.n):
                nbrsi = np.arange(self.n)
                self.nbrs[i] = nbrsi[nbrsi !=i]
                # self.nbrs[i] = np.where(self.W[i,] > 0)[0]
        else:
            print('Wrong matrix is given')

    def compute_cut(self, label):
        if self.matrix is 'Adjacency':
            d = self.W.sum(1)
            D = np.diag(d)
            L = D - self.W
            cut = 1/4 * label.dot(L.dot(label))
        elif self.matrix is 'Wigner':
            cut = label.dot(self.W.dot(label))
        else:
            print('Wrong matrix is given')
        
        return cut
 
        
    def eo(self):
        print(self.problem)
        self.find_nbrs()
        if self.matrix is 'Adjacency':
            label = 2 * np.random.randint(2, size = self.n) - 1
        elif self.matrix is 'Wigner':
            label = (2 * np.random.randint(2, size = self.n) - 1)/np.sqrt(self.n)
        else:
            print('Wrong matrix is given')
        cut0 = self.compute_cut(label)
        label0 = label
        ## initial fitness for all the nodes
        self.evaluate_fitness(np.arange(self.n), label)
        print ('Round {:.0f}'.format(0) + ', ' + 'Cut = {:.2f}'.format(cut0))

        for t in range(self.t_max):
            ## select the rank to flip by law tau
            k = self.select_rank()
            nodetoflip = self.node_kthSmallest(self.fitness, k)
            ## flip the node
            label[nodetoflip] = - label[nodetoflip]
 
            ## update the fitness 
            node_to_update = np.concatenate((nodetoflip, self.nbrs[nodetoflip]), axis=None)
            self.evaluate_fitness(node_to_update, label)

            ## compute the new cut
            cut1 = self.compute_cut(label)
            if (cut0 - cut1)*self.problem >0 :
                cut0 = cut1
                label0 = label
            
            if (t % 10000 == 0) and (t >= 10000):
                print ('Round {:.0f}'.format(t) + ', ' + 'Cut = {:.2f}'.format(cut0))
        
        return cut0, label0
        
        


