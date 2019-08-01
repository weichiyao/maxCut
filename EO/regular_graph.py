import numpy as np
import networkx as nx
import random

def regular_graph(n, degree, seed=None):
    """Python module creates the random d-regular graph that can be passed to Matlab"""
    if seed is not None:
        random.seed(seed - 1) ## In python, i-th is (i-1)-th
    
    g = nx.random_regular_graph(degree, n)
    W = nx.adjacency_matrix(g).todense()
    W = np.array(W)
    return W


# module purge
# module load matlab/2016b
# matlab