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

def generate_hd_xy(nodes,sample,d):
    x = np.zeros((sample, nodes,d))
    y = np.zeros((sample, nodes,d))
    for i in range(sample):
        for j in range(nodes):
            #mean = [0, 0]
            #cov = [[1, 0.75], [0.75, 2]]
            #x[i][j], y[i][j] = np.random.multivariate_normal(mean, cov, d).T
            #x[i][j] = np.random.beta(1, 2, size=d)
            #y[i][j] = np.random.normal(0, 1, d)
            #y[i][j] = np.exp(x[i][j])
            #y[i][j] = 128*np.power(x[i][j]-1/3,3) + 48*np.power(x[i][j]-1/3,2) -12*(x[i][j]-1/3) 
            #x[i][j] = np.random.uniform(-1, 1, size=d)
            #y[i][j] = 4*(np.power(np.power(x[i][j],2)-1/2,2)+np.random.uniform(-1,1,d)/500) 
            #x[i][j] = np.random.normal(0, 1, size=d)
            #y[i][j] = np.log(np.abs(x[i][j])) 
            #theta = np.linspace(0 , 2 * np.pi , nodes)
            #radius = 1
            #x[i] = radius * np.cos(theta) 
            #y[i] = radius * np.sin(theta)
            #x[i] = np.random.normal(0, 1, size=nodes)
            #u = np.random.normal(0, 1, size=nodes)
            #y[i] = x[i]*u
            u = np.random.uniform(-1, 1, size=d)
            v = np.random.uniform(-1, 1, size=d)
            theta = -np.pi/4
            x[i][j] = u*np.cos(theta) + v*np.sin(theta)   
            # Simulation with noise
            # x[i][j] = u*np.cos(theta) + v*np.sin(theta) + np.random.normal(0, 0.05, d)    
            y[i][j] = -u*np.sin(theta) + v*np.cos(theta)
    return x,y

def generate_hd_linear(nodes,sample,d):
    x = np.zeros((sample, nodes, d))
    y = np.zeros((sample, nodes, d))
    for i in range(sample):
        x[i] = np.random.uniform(-1,1,size=(nodes,d))
        y[i] = x[i]
    return x,y

def generate_hd_cubic(nodes,sample,d):
    x = np.zeros((sample, nodes, d))
    y = np.zeros((sample, nodes, d))
    for i in range(sample):
        x[i] = np.random.uniform(-1,1,size=(nodes,d))
        y[i] = 128*np.power(x[i]-1/3,3) + 48*np.power(x[i]-1/3,2) -12*(x[i]-1/3)
    return x,y

def generate_hd_Wshape(nodes,sample,d):
    x = np.zeros((sample, nodes, d))
    y = np.zeros((sample, nodes, d))
    for i in range(sample):
        x[i] = np.random.uniform(-1,1,size=(nodes,d))
        u = np.random.uniform(-1,1,size=(nodes,d))
        y[i] = 4*(np.power(np.power(x[i],2)-1/2,2)+u/500)
    return x,y

def generate_hd_Diamond(nodes,sample,d):
    x = np.zeros((sample, nodes, d))
    y = np.zeros((sample, nodes, d))
    for i in range(sample):
        u = np.random.uniform(-1, 1, size=(nodes,d))
        v = np.random.uniform(-1, 1, size=(nodes,d))
        theta = -np.pi/4
        x[i] = u*np.cos(theta) + v*np.sin(theta) 
        y[i] = -u*np.sin(theta) + v*np.cos(theta)
    return x,y

def generate_hd_Indep(nodes,sample,d):
    x = np.zeros((sample, nodes, d))
    y = np.zeros((sample, nodes, d))
    mean = np.zeros(d)
    cov = np.eye(d)
    for i in range(sample):
        u = np.random.multivariate_normal(mean, cov, nodes)
        v = np.random.multivariate_normal(mean, cov, nodes)
        u1 = np.random.binomial(size=(nodes,d), n=1, p= 0.5)
        v1 = np.random.binomial(size=(nodes,d), n=1, p= 0.5)
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

sample = 50
d_list = [1,2,3,4,5,6,7,8,9,10]
nodes_list = [10,20,30,40,50,60,70,80,90,100]

gcor_list = torch.zeros(len(d_list),len(nodes_list))
gcov_list = torch.zeros(len(d_list),len(nodes_list))

for l in range(len(d_list)):
    d = d_list[l]
    for m in range(len(nodes_list)):
        nodes = nodes_list[m]         
        p = torch.from_numpy(ot.unif(nodes))
        q = torch.from_numpy(ot.unif(nodes))
        
        gcov = []
        gcorr = []

        for s in range(10):
            x, y = generate_hd_Indep(nodes,sample,d)

            pm1_tensor = torch.zeros(sample,nodes, nodes,dtype=torch.float64)
            pm2_tensor = torch.zeros(sample,nodes, nodes,dtype=torch.float64)

            for k in range(sample):
                x_gpu = torch.from_numpy(x)
                y_gpu = torch.from_numpy(y)

                for i in range(nodes):
                    for j in range(i,nodes):
                        pm1_var = torch.exp(-torch.pow(torch.norm(x_gpu[k][i]-x_gpu[k][j]),2))
                        pm1_tensor[k][i, j] = pm1_var
                        pm1_tensor[k].T[i,j] = pm1_var
                        pm2_var = torch.exp(-torch.abs(torch.norm(y_gpu[k][i]-y_gpu[k][j])))
                        pm2_tensor[k][i, j] = pm2_var
                        pm2_tensor[k].T[i,j] = pm2_var
                pm1_tensor[k].fill_diagonal_(0)
                pm2_tensor[k].fill_diagonal_(0)

            save_stdout = sys.stdout
            sys.stdout = open('trash', 'w')
            dist_xx = compute_dist(pm1_tensor,pm1_tensor)
            dist_yy = compute_dist(pm2_tensor,pm2_tensor)
            sys.stdout = save_stdout

            tilde_p1 = U_center_gpu(dist_xx)
            tilde_p2 = U_center_gpu(dist_yy)
            var1 = U_product_gpu(tilde_p1,tilde_p2)
            gcov.append(var1)
            g1 = U_product_gpu(tilde_p1,tilde_p1)
            g2 = U_product_gpu(tilde_p2,tilde_p2)
            var2 = var1/torch.sqrt(g1*g2)
            gcorr.append(var2)

        gcov_list[l,m] = np.mean(gcov)
        gcor_list[l,m] = np.mean(gcorr)

        print(l,m,gcov_list[l,m],gcor_list[l,m])
    
d1 = gcov_list 
d2 = gcor_list

df1 = pd.DataFrame(d1)
df2 = pd.DataFrame(d2)

df1.to_csv('.../tcov_diff_nodes.csv')
df2.to_csv('.../tcor_diff_nodes.csv')
    
    
    
    
    