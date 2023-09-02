import numpy as np
from scipy.stats import bernoulli
#from numpy import random
import math
#import igraph
import random
import time
#import functorch
import pandas as pd
import seaborn as sns
import torch
import ot
import sys
import itertools
from scipy.spatial.distance import pdist, squareform


def generate_adj(pm):
    nodes = pm.shape[0]
    A = torch.bernoulli(pm)
    i, j = torch.triu_indices(nodes, nodes)
    vals = A[i,j]
    A.T[i, j] = vals
    return A

def get_norm_laplacian(A):
    dval = torch.sum(A,dim=1)
    D = 1/torch.sqrt(dval)
    D[D == float("Inf")] = 0
    LA = D*A*D
    return LA

def U_center_gpu(mean_A):
    nodes = mean_A.shape[0]
    row_sum = torch.sum(mean_A, 1)
    col_sum = torch.sum(mean_A, 0)
    all_sum = torch.sum(mean_A)
    i, j = torch.triu_indices(nodes,nodes)
    center_val = mean_A[i, j]- row_sum[i]/nodes - col_sum[j]/nodes + all_sum/(nodes*nodes)
    mean_A[i, j] = center_val
    mean_A.T[i,j] = center_val
    mean_A.fill_diagonal_(0)
    return mean_A

def U_product_gpu(U,V):
    n = U.shape[0]
    i, j = torch.tril_indices(n,n,-1)
    U_val = U[i,j]*V[i,j]
    re = 2*torch.sum(U_val)/(n*n)
    return re

def compute_dist(p1,p2,p,q):
    sample = p1.shape[0]
    dist = torch.zeros(sample, sample,dtype=torch.float64)    
    for i in range(sample):
        for j in range(i,sample):
            dist[i, j] = np.abs(ot.gromov.gromov_wasserstein(p1[i], p2[j], p, q, 'square_loss', verbose=True, log=True)[1]['gw_dist'].item())
            dist[j, i] = dist[i,j]
    return dist

def sample_graph_gpu(adj_A, sel_nodes):
    A11 = torch.index_select(adj_A, dim = 0, index = sel_nodes)
    A12 = torch.index_select(A11, dim = 1, index = sel_nodes)
    return A12


def get_pval(dist_yy,stat,g1,tilde_p1,bootsam_gpu):
    sum_gre = 0
    
    for k in range(bootsam_gpu.shape[0]):
        dist_yy1 = sample_graph_gpu(dist_yy, bootsam_gpu[k,:])
        tilde_p21 = U_center_gpu(dist_yy1)
        gcov1 = U_product_gpu(tilde_p1,tilde_p21)
        g21 = U_product_gpu(tilde_p21,tilde_p21)
        gcorr1 = gcov1/torch.sqrt(g1*g21)
        sum_gre = sum_gre + np.greater_equal(gcorr1.item(), stat)
        
    pval = (1+sum_gre)/(1+n_boot)
    return pval


list1 = ['sort_ADHD_subnet1','sort_ADHD_subnet2','sort_ADHD_subnet3','sort_ADHD_subnet4','sort_ADHD_subnet5']

comb1 = []
for x in itertools.combinations(list1, 2):
    comb1.append(x)

list2 = ['health_subnet1','health_subnet2','health_subnet3','health_subnet4','health_subnet5']

comb2 = []
for x in itertools.combinations(list2, 2):
    comb2.append(x)

compare_list = comb1 + comb2

gcor_list = torch.zeros(len(compare_list))
gcov_list = torch.zeros(len(compare_list))

sample = 50
nodes = 1000

for l in compare_list:
    file1 =  '.../Data/' + l[0] + '.np.npy'
    file2 =  '.../Data/' + l[1] + '.np.npy'

    pose1 = np.load(file1)
    pose2 = np.load(file2)

    x_gpu = torch.from_numpy(pose1)
    y_gpu = torch.from_numpy(pose2)

    pm1_tensor = x_gpu
    pm2_tensor = y_gpu

    sample = x_gpu.shape[0]
    nodes1 = x_gpu.shape[1]
    nodes2 = y_gpu.shape[1]

    p = torch.from_numpy(ot.unif(nodes1))
    q = torch.from_numpy(ot.unif(nodes2))

    save_stdout = sys.stdout
    sys.stdout = open('trash', 'w')
    dist_xx = compute_dist(pm1_tensor,pm1_tensor,p,p)
    dist_yy = compute_dist(pm2_tensor,pm2_tensor,q,q)
    sys.stdout = save_stdout

    tilde_p1 = U_center_gpu(dist_xx)
    tilde_p2 = U_center_gpu(dist_yy)
    gcov = U_product_gpu(tilde_p1,tilde_p2)
    #gcov_list[l] = gcov

    g1 = U_product_gpu(tilde_p1,tilde_p1)
    g2 = U_product_gpu(tilde_p2,tilde_p2)
    gcorr0 = gcov/torch.sqrt(g1*g2)
    #gcor_list[l] = gcorr

    n_boot = 1000
    sel_n = sample
    bootsam = np.zeros((n_boot, sel_n), int)
    for m in range(n_boot):
        bootsam[m,:] = random.sample(range(0,sample),sel_n)
    bootsam_gpu = torch.from_numpy(bootsam)

    gcorr_pval = get_pval(dist_yy,gcorr0,g1,tilde_p1,bootsam_gpu)

    print(l,gcov,gcorr0,gcorr_pval)

                                               
    
#d1 = gcov_list 
#d2 = gcor_list

#df1 = pd.DataFrame(d1)
#df2 = pd.DataFrame(d2)

    
    
    
    
    