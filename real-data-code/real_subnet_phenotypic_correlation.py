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
import torch.nn.functional as F


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

def get_pval(dist_xx,stat,g2,tilde_p2,bootsam_gpu):
    sum_gre = 0
    
    for k in range(bootsam_gpu.shape[0]):
        dist_xx_1 = sample_graph_gpu(dist_xx, bootsam_gpu[k,:])
        tilde_p12 = U_center_gpu(dist_xx_1)
        gcov1_2 = U_product_gpu(tilde_p12,tilde_p2)
        g21 = U_product_gpu(tilde_p12,tilde_p12)
        gcorr1 = gcov1_2/torch.sqrt(g21*g2)
        sum_gre = sum_gre + np.greater_equal(torch.abs(gcorr1), torch.abs(stat))
        
    pval = (1+sum_gre)/(1+n_boot)
    return pval


Attr = ['Age','ADHD Index','Inattentive','Hyper/Impulsive','Verbal IQ','Performance IQ','Full4 IQ']

adhd_df = pd.read_csv(".../Data/Peking_1_phenotypic.csv")

subnet1 = np.load('.../Data/sort_ADHD_subnet1.np.npy')
subnet2 = np.load('.../Data/sort_ADHD_subnet2.np.npy')
subnet3 = np.load('.../Data/sort_ADHD_subnet3.np.npy')
subnet4 = np.load('.../Data/sort_ADHD_subnet4.np.npy')
subnet5 = np.load('.../Data/sort_ADHD_subnet5.np.npy')


x_gpu1 = torch.from_numpy(subnet1)
x_gpu2 = torch.from_numpy(subnet2)
x_gpu3 = torch.from_numpy(subnet3)
x_gpu4 = torch.from_numpy(subnet4)
x_gpu5 = torch.from_numpy(subnet5)

pm1_tensor = x_gpu1
pm2_tensor = x_gpu2
pm3_tensor = x_gpu3
pm4_tensor = x_gpu4
pm5_tensor = x_gpu5

sample = x_gpu1.shape[0]

nodes1 = x_gpu1.shape[1]
nodes2 = x_gpu2.shape[1]
nodes3 = x_gpu3.shape[1]
nodes4 = x_gpu4.shape[1]
nodes5 = x_gpu5.shape[1]

p1 = torch.from_numpy(ot.unif(nodes1))
p2 = torch.from_numpy(ot.unif(nodes2))
p3 = torch.from_numpy(ot.unif(nodes3))
p4 = torch.from_numpy(ot.unif(nodes4))
p5 = torch.from_numpy(ot.unif(nodes5))

save_stdout = sys.stdout
sys.stdout = open('trash', 'w')
dist_xx1 = compute_dist(pm1_tensor,pm1_tensor,p1,p1)
dist_xx2 = compute_dist(pm2_tensor,pm2_tensor,p2,p2)
dist_xx3 = compute_dist(pm3_tensor,pm3_tensor,p3,p3)
dist_xx4 = compute_dist(pm4_tensor,pm4_tensor,p4,p4)
dist_xx5 = compute_dist(pm5_tensor,pm5_tensor,p5,p5)
sys.stdout = save_stdout

gcor_list = torch.zeros(5,len(Attr))
gcov_list = torch.zeros(5,len(Attr))
gcorr_pval = torch.zeros(5,len(Attr))

for l in Attr:
    y_gpu = torch.tensor(list(adhd_df[l]),dtype=torch.float)
    y_gpu = F.normalize(y_gpu, p=2, dim=0)

    dist_yy = torch.zeros(sample, sample,dtype=torch.float64) 

    for i in range(sample):
        for j in range(i,sample):
            dist_var = torch.exp(-torch.pow(y_gpu[i]-y_gpu[j],2))
            dist_yy[i, j] = dist_var
            dist_yy.T[i,j] = dist_var

    tilde_p1_1 = U_center_gpu(dist_xx1)
    tilde_p1_2 = U_center_gpu(dist_xx2)
    tilde_p1_3 = U_center_gpu(dist_xx3)
    tilde_p1_4 = U_center_gpu(dist_xx4)
    tilde_p1_5 = U_center_gpu(dist_xx5)
    tilde_p2 = U_center_gpu(dist_yy)
    gcov1 = U_product_gpu(tilde_p1_1,tilde_p2)
    gcov2 = U_product_gpu(tilde_p1_2,tilde_p2)
    gcov3 = U_product_gpu(tilde_p1_3,tilde_p2)
    gcov4 = U_product_gpu(tilde_p1_4,tilde_p2)
    gcov5 = U_product_gpu(tilde_p1_5,tilde_p2)
    
    g1_1 = U_product_gpu(tilde_p1_1,tilde_p1_1)
    g1_2 = U_product_gpu(tilde_p1_2,tilde_p1_2)
    g1_3 = U_product_gpu(tilde_p1_3,tilde_p1_3)
    g1_4 = U_product_gpu(tilde_p1_4,tilde_p1_4)
    g1_5 = U_product_gpu(tilde_p1_5,tilde_p1_5)

    g2 = U_product_gpu(tilde_p2,tilde_p2)
    gcorr0_1 = gcov1/torch.sqrt(g1_1*g2)
    gcorr0_2 = gcov2/torch.sqrt(g1_2*g2)
    gcorr0_3 = gcov3/torch.sqrt(g1_3*g2)
    gcorr0_4 = gcov4/torch.sqrt(g1_4*g2)
    gcorr0_5 = gcov5/torch.sqrt(g1_5*g2)
    #gcor_list[1,l] = torch.abs(gcorr0_1)
    #gcor_list[2,l] = torch.abs(gcorr0_2)
    #gcor_list[3,l] = torch.abs(gcorr0_3)
    #gcor_list[4,l] = torch.abs(gcorr0_4)
    #gcor_list[5,l] = torch.abs(gcorr0_5)
    print(l,torch.abs(gcorr0_1))
    print(l,torch.abs(gcorr0_2))
    print(l,torch.abs(gcorr0_3))
    print(l,torch.abs(gcorr0_4))
    print(l,torch.abs(gcorr0_5))
    

    n_boot = 10000
    sel_n = sample
    bootsam = np.zeros((n_boot, sel_n), int)
    for m in range(n_boot):
        bootsam[m,:] = random.sample(range(0,sample),sel_n)
    bootsam_gpu = torch.from_numpy(bootsam)

    gcorr_pval1 = get_pval(dist_xx1,gcorr0_1,g2,tilde_p2,bootsam_gpu)
    gcorr_pval2 = get_pval(dist_xx2,gcorr0_2,g2,tilde_p2,bootsam_gpu)
    gcorr_pval3 = get_pval(dist_xx3,gcorr0_3,g2,tilde_p2,bootsam_gpu)
    gcorr_pval4 = get_pval(dist_xx4,gcorr0_4,g2,tilde_p2,bootsam_gpu)
    gcorr_pval5 = get_pval(dist_xx5,gcorr0_5,g2,tilde_p2,bootsam_gpu)

    print(l,gcorr_pval1)
    print(l,gcorr_pval2)
    print(l,gcorr_pval3)
    print(l,gcorr_pval4)
    print(l,gcorr_pval5)
    

                                               
    
    
    
    
    