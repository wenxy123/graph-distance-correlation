import numpy as np
import pandas as pd
import torch
import os

path = '.../Data/Peking_1/'

group_df = pd.read_csv('.../Data/group_df1.csv')
phenotypic_df = pd.read_csv('.../Data/Peking_1_phenotypic.csv')
patient = list(phenotypic_df['ScanDir ID'])

file_list = []
for i in patient:
    name = '/sfnwmrda' + str(i) + '_session_1_rest_1_cc200_TCs.1D'
    file_name = path + str(i) + name
    file_list.append(file_name)

sample = len(file_list)


#f = lambda x:x.replace(" ","")
#group_df['node'] = group_df['node'].apply(f)

group = ['Sensory','Limbic-Subcortical','Cognitive Control','Default Mode','Uncertain']


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

node_list1 = list(group_df1['node'])
node_list2 = list(group_df2['node'])
node_list3 = list(group_df3['node'])
node_list4 = list(group_df4['node'])
node_list5 = list(group_df5['node'])

subnet_tensor1 = np.zeros((sample,nodes1, nodes1))
subnet_tensor2 = np.zeros((sample,nodes2, nodes2))
subnet_tensor3 = np.zeros((sample,nodes3, nodes3))
subnet_tensor4 = np.zeros((sample,nodes4, nodes4))
subnet_tensor5 = np.zeros((sample,nodes5, nodes5))

for i in range(len(file_list)):
    path = file_list[i]
    df = pd.read_csv(path,sep='\t')
    df = df.drop(['File','Sub-brick'],axis=1)
    colname = list(df.columns.values)
    colname1 = [x.replace(" ","") for x in colname]
    df.columns = colname1

    df1 = df[node_list1].T
    df2 = df[node_list2].T
    df3 = df[node_list3].T
    df4 = df[node_list4].T
    df5 = df[node_list5].T

    corr1 = np.corrcoef(df1)
    np.fill_diagonal(corr1, 0)
    corr1 = np.abs(corr1)
    subnet_tensor1[i] = corr1

    corr2 = np.corrcoef(df2)
    np.fill_diagonal(corr2, 0)
    corr2 = np.abs(corr2)
    subnet_tensor2[i] = corr2

    corr3 = np.corrcoef(df3)
    np.fill_diagonal(corr3, 0)
    corr3 = np.abs(corr3)
    subnet_tensor3[i] = corr3

    corr4 = np.corrcoef(df4)
    np.fill_diagonal(corr4, 0)
    corr4 = np.abs(corr4)
    subnet_tensor4[i] = corr4

    corr5 = np.corrcoef(df5)
    np.fill_diagonal(corr5, 0)
    corr5 = np.abs(corr5)
    subnet_tensor5[i] = corr5

#np_array1 = subnet_tensor1.numpy()
#np_array2 = subnet_tensor2.numpy()
#np_array3 = subnet_tensor3.numpy()
#np_array4 = subnet_tensor4.numpy()
#np_array5 = subnet_tensor5.numpy()

np.save('.../Data/sort_ADHD_subnet1.np', subnet_tensor1)
np.save('.../Data/sort_ADHD_subnet2.np', subnet_tensor2)
np.save('.../Data/sort_ADHD_subnet3.np', subnet_tensor3)
np.save('.../Data/sort_ADHD_subnet4.np', subnet_tensor4)
np.save('.../Data/sort_ADHD_subnet5.np', subnet_tensor5)
