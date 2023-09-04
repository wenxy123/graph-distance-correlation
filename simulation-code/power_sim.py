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

def generate_xy(nodes,sample):
    x = np.zeros((sample, nodes))
    y = np.zeros((sample, nodes))
    for i in range(sample):
        #x[i] = np.random.beta(1, 2, size=nodes)
        #y[i] = np.random.normal(0, 1, nodes)
        #y[i] = np.exp(x[i]) 
        #x[i] = np.random.beta(1, 2, size=nodes)
        #y[i] = 128*np.power(x[i]-1/3,3) + 48*np.power(x[i]-1/3,2) -12*(x[i]-1/3) 
        #mean = [0, 0]
        #cov = [[1, 0.5], [0.5, 1]]
        #x[i], y[i] = np.random.multivariate_normal(mean, cov, nodes).T
        #x[i] = np.random.uniform(-1, 1, size=nodes)
        #y[i] = 4*(np.power(np.power(x[i],2)-1/2,2)+np.random.uniform(-1,1,nodes)/500)
        #x[i] = np.random.normal(0, 1, size=nodes)
        #y[i] = 2*np.log2(np.abs(x[i])) + np.random.normal(0, 0.05, nodes)
        #u = np.random.uniform(-1, 1, size=nodes)
        #v = np.random.uniform(-1, 1, size=nodes)
        #theta = -np.pi/4
        #x[i] = u*np.cos(theta) + v*np.sin(theta) 
        #y[i] = -u*np.sin(theta) + v*np.cos(theta)
        #x[i] = np.random.normal(0, 1, size=nodes)
        #u = np.random.normal(0, 1, size=nodes)
        #y[i] = x[i]*u
        u = np.random.normal(0, 1, size=nodes)
        v = np.random.normal(0, 1, size=nodes)
        u1 = np.random.binomial(size=nodes, n=1, p= 0.5)
        v1 = np.random.binomial(size=nodes, n=1, p= 0.5)

        x[i] = u/3 + 2*u1 - 1
        y[i] = v/3 + 2*v1 - 1
    return x,y

def generate_adj(pm):
    nodes = pm.shape[0]
    A = torch.bernoulli(pm)
    i, j = torch.triu_indices(nodes, nodes)
    vals = A[i,j]
    A.T[i, j] = vals
    return A


def U_center_gpu(mean_A):
    nodes = mean_A.shape[0]
    row_sum = torch.sum(mean_A, 1)
    col_sum = torch.sum(mean_A, 0)
    all_sum = torch.sum(mean_A)
    i, j = torch.triu_indices(nodes,nodes)
    center_val = mean_A[i, j]- row_sum[i]/(nodes-2) - col_sum[j]/(nodes-2) + all_sum/((nodes-1)*(nodes-2))
    mean_A[i, j] = center_val
    mean_A.T[i,j] = center_val
    mean_A.fill_diagonal_(0)
    return mean_A

def U_product_gpu(U,V):
    n = U.shape[0]
    i, j = torch.tril_indices(n,n,-1)
    U_val = U[i,j]*V[i,j]
    re = 2*torch.sum(U_val)/(n*(n-3))
    return re

def compute_dist(p1,p2):
    sample = p1.shape[0]
    dist = torch.zeros(sample, sample,dtype=torch.float64)    
    for i in range(sample):
        for j in range(i,sample):
            dist[i, j] = math.sqrt(np.abs(ot.gromov.gromov_wasserstein(p1[i], p2[j], p, q, 'square_loss', verbose=True, log=True)[1]['gw_dist'].item()))
            dist[j, i] = dist[i,j]
    return dist

def sample_graph_gpu(adj_A, sel_nodes):
    A11 = torch.index_select(adj_A, dim = 0, index = sel_nodes)
    A12 = torch.index_select(A11, dim = 1, index = sel_nodes)
    return A12

def sample_tensor_gpu(tensor, sel_sample):
    t1 = torch.index_select(tensor, dim = 0, index = sel_sample)
    return t1

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

def get_pval1(y,stat,g1,tilde_p1,bootsam_gpu):
    sum_gre=0
    for m in range(bootsam_gpu.shape[0]):
        y1 = sample_tensor_gpu(y_gpu, bootsam_gpu[m,:])
        pm21_tensor = torch.zeros(sample,nodes, nodes,dtype=torch.float64)
        for k in range(sample):
            i, j = torch.triu_indices(nodes,nodes)
            pm21_var = torch.exp(-torch.abs(y1[k][i]-y1[k][j]))
            pm21_tensor[k][i, j] = pm21_var
            pm21_tensor[k].T[i,j] = pm21_var
            pm21_tensor[k].fill_diagonal_(0)
        save_stdout = sys.stdout
        sys.stdout = open('trash', 'w')
        dist_yy1 = compute_dist(pm21_tensor,pm21_tensor)
        sys.stdout = save_stdout
        tilde_p21 = U_center_gpu(dist_yy1)
        gcov1 = U_product_gpu(tilde_p1,tilde_p21)
        g21 = U_product_gpu(tilde_p21,tilde_p21)
        gcorr1 = gcov1/torch.sqrt(g1*g21)
        sum_gre = sum_gre + np.greater_equal(gcorr1.item(), stat)
        
    pval = (1+sum_gre)/(1+n_boot)
    return pval



nodes = 300
sample_list = [5,15,25,35,45,50]
#sample_list =[50]
tcov_pow_list = torch.zeros(len(sample_list))

alpha = 0.05
pt = 500

for n in range(len(sample_list)):
    sample = sample_list[n]
    pow_list = torch.zeros(50)

    p = torch.from_numpy(ot.unif(nodes))
    q = torch.from_numpy(ot.unif(nodes)) 
    
    for s in range(50):
        #gcorr_pow = 0
        x, y = generate_xy(nodes,sample)

        x_gpu = torch.from_numpy(x)
        y_gpu = torch.from_numpy(y)
        
        pm1_tensor = torch.zeros(sample,nodes, nodes,dtype=torch.float64)
        pm2_tensor = torch.zeros(sample,nodes, nodes,dtype=torch.float64)

        for k in range(sample):

            i, j = torch.triu_indices(nodes,nodes)
            pm1_var = torch.exp(-torch.pow(x_gpu[k][i]-x_gpu[k][j],2))
            pm1_tensor[k][i, j] = pm1_var
            pm1_tensor[k].T[i,j] = pm1_var
            pm1_tensor[k].fill_diagonal_(0)

            pm2_var = torch.exp(-torch.abs(y_gpu[k][i]-y_gpu[k][j]))
            pm2_tensor[k][i, j] = pm2_var
            pm2_tensor[k].T[i,j] = pm2_var
            pm2_tensor[k].fill_diagonal_(0)

        save_stdout = sys.stdout
        sys.stdout = open('trash', 'w')
        dist_xx = compute_dist(pm1_tensor,pm1_tensor)
        dist_yy = compute_dist(pm2_tensor,pm2_tensor)
        sys.stdout = save_stdout

        tilde_p1 = U_center_gpu(dist_xx)
        tilde_p2 = U_center_gpu(dist_yy)
        gcov = U_product_gpu(tilde_p1,tilde_p2)
        g1 = U_product_gpu(tilde_p1,tilde_p1)
        g2 = U_product_gpu(tilde_p2,tilde_p2)
        gcorr0 = gcov/torch.sqrt(g1*g2)
        
        gcorr_pow = 0

        for l in range(pt):
            n_boot = 1000
            sel_n = sample
            bootsam = np.zeros((n_boot, sel_n), int)
            for m in range(n_boot):
                bootsam[m,:] = random.sample(range(0,sample),sel_n)
            bootsam_gpu = torch.from_numpy(bootsam)
        
            gcorr_pval = get_pval(dist_yy,gcorr0,g1,tilde_p1,bootsam_gpu)
            gcorr_pow = gcorr_pow + np.greater_equal(alpha, gcorr_pval)
            
            #clean the memroy
            #gc.collect()
            #print(n,l,gcorr_pval,gcorr0)
        print(n,s,gcorr_pow/pt,gcorr0)
        pow_list[s] = gcorr_pow/pt

    tcov_pow_list[n] = torch.mean(pow_list)
    print(tcov_pow_list[n])
    
df1 = pd.DataFrame(tcov_pow_list)
#df2 = pd.DataFrame(d2)

df1.to_csv('.../tcor_pow.csv')

