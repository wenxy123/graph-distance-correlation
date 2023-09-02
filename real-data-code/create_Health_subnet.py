import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
#from stl import mesh


path = '.../Data/Health/CPAC200_space-MNI152NLin6_res-2x2x2.nii.gz/'
path_list = os.listdir(path)


list1 = [x.split('_')[0] for x in path_list]
list2 = list(set(list1))


list2.remove('sub-0025898')


sel_file = []
for i in list2:
    sel_name = i + '_ses-1_func_CPAC200_space-MNI152NLin6_res-2x2x2.nii.gz_abs_edgelist.csv'
    sel_file.append(sel_name)


sample = len(sel_file)
nodes = 200


pm1_tensor = torch.zeros(sample, nodes, nodes)

for k in range(len(sel_file)):
    path1 = path + sel_file[k]
    df = pd.read_csv(path1,header=None,sep=' ')
    df[0] = df[0] - 1
    df[1] = df[1] - 1
    i, j = torch.triu_indices(nodes,nodes,1)
    pm1_tensor[k][i,j] = torch.Tensor(list(df[2]))
    pm1_tensor[k].T[i,j] = torch.Tensor(list(df[2]))


group_df = pd.read_csv('.../Data/group_df1.csv')

f = lambda x:int(x.split('_')[1])-1
group_df['node1'] = group_df['node'].apply(f)

group_df1 = group_df[group_df['group']=='Sensory']
group_df2 = group_df[group_df['group']=='Limbic-Subcortical']
group_df3 = group_df[group_df['group']=='Cognitive Control']
group_df4 = group_df[group_df['group']=='Default Mode']
group_df5 = group_df[group_df['group']=='Uncertain']


nodes1 = len(group_df1)
nodes2 = len(group_df2)
nodes3 = len(group_df3)
nodes4 = len(group_df4)
nodes5 = len(group_df5)


def sample_graph_gpu(adj_A, sel_nodes):
    A11 = torch.index_select(adj_A, dim = 0, index = sel_nodes)
    A12 = torch.index_select(A11, dim = 1, index = sel_nodes)
    return A12


health_subnet1 = torch.zeros(sample, nodes1, nodes1)
health_subnet2 = torch.zeros(sample, nodes2, nodes2)
health_subnet3 = torch.zeros(sample, nodes3, nodes3)
health_subnet4 = torch.zeros(sample, nodes4, nodes4)
health_subnet5 = torch.zeros(sample, nodes5, nodes5)

for k in range(len(sel_file)):
    sel_nodes1 = torch.tensor(list(group_df1['node1']))
    sel_nodes2 = torch.tensor(list(group_df2['node1']))
    sel_nodes3 = torch.tensor(list(group_df3['node1']))
    sel_nodes4 = torch.tensor(list(group_df4['node1']))
    sel_nodes5 = torch.tensor(list(group_df5['node1']))
    adj_A = pm1_tensor[k]
    health_subnet1[k] = sample_graph_gpu(adj_A, sel_nodes1)
    health_subnet2[k] = sample_graph_gpu(adj_A, sel_nodes2)
    health_subnet3[k] = sample_graph_gpu(adj_A, sel_nodes3)
    health_subnet4[k] = sample_graph_gpu(adj_A, sel_nodes4)
    health_subnet5[k] = sample_graph_gpu(adj_A, sel_nodes5)


np.save('.../Data/health_subnet1.np.npy', np1)
np.save('.../Data/health_subnet2.np.npy', np2)
np.save('.../Data/health_subnet3.np.npy', np3)
np.save('.../Data/health_subnet4.np.npy', np4)
np.save('.../Data/health_subnet5.np.npy', np5)


