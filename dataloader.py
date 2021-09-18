import pandas as pd
from torch.utils.data import IterableDataset, Dataset
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
import scipy.io as sci
import scipy as sp
import random
import numpy as np
import math
import os

class RecData(object):
    def __init__(self, dir, file_name):
        file_name = file_name + 'data.mat'
        # file_name = file_name + 'data.txt'
        self.file_name = os.path.join(dir, file_name)

    def get_data(self,ratio):
        mat = self.load_file(filename=self.file_name)
        train_mat, test_mat = self.split_matrix(mat, ratio)
        return train_mat, test_mat
    
    def load_file(self,filename=''):
        # if file_name.endswith('.mat'):
        #     return sci.loadmat(file_name)['data']
        # else:
        #     raise ValueError('not supported file type')
        if filename.endswith('.mat'):
            return sci.loadmat(filename)['data']
        elif filename.endswith('.txt') or filename.endswith('.tsv'):
            sep = '\t'
        elif filename.endswith('.csv'):
            sep = ','
        else:
            raise ValueError('not supported file type')
        max_user = -1
        max_item = -1
        row_idx = []
        col_idx = []
        data = []
        for line in open(filename):
            user, item, rating = line.strip().split(sep)
            user, item, rating = int(user) -1, int(item)-1, float(rating)
            row_idx.append(user)
            col_idx.append(item)
            data.append(rating)
            if user > max_user:
                max_user = user
            if item > max_item:
                max_item = item
        return sp.sparse.csc_matrix((data, (row_idx, col_idx)), (max_user+1, max_item+1))

    def split_matrix(self, mat, ratio=0.8):
        mat = mat.tocsr()  #按行读取，即每一行为一个用户
        m,n = mat.shape
        train_data_indices = []
        train_indptr = [0] * (m+1)
        test_data_indices = []
        test_indptr = [0] * (m+1)
        for i in range(m):
            row = [(mat.indices[j], mat.data[j]) for j in range(mat.indptr[i], mat.indptr[i+1])]
            train_idx = random.sample(range(len(row)), round(ratio * len(row)))
            train_binary_idx = np.full(len(row), False)
            train_binary_idx[train_idx] = True
            test_idx = (~train_binary_idx).nonzero()[0]
            for idx in train_idx:
                train_data_indices.append(row[idx]) 
            train_indptr[i+1] = len(train_data_indices)
            for idx in test_idx:
                test_data_indices.append(row[idx])
            test_indptr[i+1] = len(test_data_indices)

        [train_indices, train_data] = zip(*train_data_indices)
        [test_indices, test_data] = zip(*test_data_indices)

        train_mat = sp.sparse.csr_matrix((train_data, train_indices, train_indptr), (m,n))
        test_mat = sp.sparse.csr_matrix((test_data, test_indices, test_indptr), (m,n))
        return train_mat, test_mat


class UserItemData(Dataset):
    def __init__(self, train_mat, train_flag=True):
        super(UserItemData, self).__init__()
        self.train = train_mat
        if train_flag is True:
            self.users = np.random.permutation(self.train.shape[0])
        else:
            self.users = np.arange(self.train.shape[0])
    
    def __len__(self):
        return self.train.shape[0]
    
    def __getitem__(self, idx):
        # return self.user[idx], self.item[idx]
        pos_idx = self.train[self.users[idx]].nonzero()[1]
        return pos_idx