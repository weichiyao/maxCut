import re
import heapq
import pickle
import numpy as np
import random 
import os
import scipy.stats as stats
import regular_graph as reg
from extremal_optimization import EOCut
import sys

def main(argv):
    # if int(argv[0]) < 1:
    #     print("the first input should be larger than 0, which is the job number")
    #     exit()


    # job_number = int(argv[0]) ## the second argument value
    resname = 'eo_label_n1000_' + str(argv[0])  + '.pickle'
    path_plus_name = os.path.join(argv[1], resname)


    try: #fh = open(path_plus_name, 'r') # Store configuration file values
        with open(path_plus_name, 'rb') as f:
            res = pickle.load(f)


        z1 = res['cut1']
        z2 = res['cut2']
        l1 = res['label1']
        l2 = res['label2']
    except FileNotFoundError:
        z1 = np.zeros(7)
        z2 = np.zeros(7)
        l1 = [[]] * 7
        l2 = [[]] * 7
        
    
    
    znonzero = np.where(z1==0)[0]

    
    if znonzero.shape[0] > 0:
        znonzero = znonzero[znonzero != 11]
        znonzero = znonzero[znonzero != 12]
        for jj in znonzero:
            job_number = jj + 1 
            print('Job number is {:.0f}'.format(job_number))
            N = 1000
            t_max = 5000 * N
            if job_number == 1:
                d = 3
            elif job_number == 2:
                d = 20
            elif job_number == 3:
                d = 50
            elif job_number == 4:
                d = 100
            elif job_number == 5:
                d = 200
            elif job_number == 6:
                d = 400
            else:
                d = 600

            W = reg.regular_graph(N, d, seed = int(argv[0])) 

            tau = 1.4 
            problem = -1
            

            ## extremal optimization
            eocut = EOCut(problem, W, tau, t_max, matrix = 'Adjacency')
            cut1, l1[jj] = eocut.eo()
            cut2, l2[jj] = eocut.eo()
            
            ## normalize the cut
            z1[jj] = (cut1/N - d/4)/np.sqrt(d/4)
            z2[jj] = (cut2/N - d/4)/np.sqrt(d/4)
            res = {
                'cut1': z1,
                'label1': l1,
                'cut2': z2,
                'label2': l2
            }

            print('Node size {:.0f}'.format(N) + ', ' + 'degree {:.0f}'. format(d))
    
            with open(path_plus_name, 'wb') as f:
                pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)




if __name__ == '__main__':
    main(sys.argv[1:]) ## argv[0] 

