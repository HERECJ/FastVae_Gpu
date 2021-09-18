from sklearn import cluster
from sampler import softamx_ny
import torch
from sklearn.cluster import KMeans
import scipy as sp
import scipy.sparse
import scipy.special
import numpy as np

class MidxUniform:
    def __init__(self, mat:sp.sparse, num_neg:int, user_embs:torch.tensor, item_embs:np.ndarray, num_cluster:int, device, **kwargs):
        self.mat = mat.tocsr()
        self.num_users = mat.shape[0]
        self.num_items = item_embs.shape[0]
        self.num_neg = num_neg
        self.preprocess_time = 0.0
        self.sample_time = 0.0

        self.num_cluster = num_cluster

        latent_dim = user_embs.shape[1]
        self.user_embs = user_embs
        assert latent_dim%2 <1, ValueError('The dimension must be the even!') 
        cluster_dim = latent_dim // 2
        self.cluster_dim = cluster_dim

        cluster_kmeans_0 = KMeans(n_clusters=self.num_cluster, random_state=0).fit(item_embs[:,:cluster_dim])
        centers_0 = cluster_kmeans_0.cluster_centers_
        # self.centers_0 = centers_0
        self.centers_0 = torch.tensor(centers_0, device=device, dtype=torch.float32)
        codes_0 = cluster_kmeans_0.labels_

        cluster_kmeans_1 = KMeans(n_clusters=self.num_cluster, random_state=0).fit(item_embs[:,cluster_dim:])
        centers_1 =cluster_kmeans_1.cluster_centers_
        # self.centers_1 = centers_1
        self.centers_1 = torch.tensor(centers_1, device=device,dtype=torch.float32)
        codes_1 = cluster_kmeans_1.labels_

        self.union_idx = [codes_0[i] * num_cluster + codes_1[i] for i in range(self.num_items)]

        self.combine_cluster = sp.sparse.csc_matrix((np.ones_like(self.union_idx), (np.arange(self.num_items), self.union_idx)), shape=(self.num_items, self.num_cluster**2))
        combine_sum = np.sum(self.combine_cluster, axis=0).A
        self.idx_nonzero = combine_sum > 0

        w_kk = np.float32(self.combine_cluster.sum(axis=0).A)
        w_kk[self.idx_nonzero] = np.log(w_kk[self.idx_nonzero])
        w_kk[np.invert(self.idx_nonzero)] = -np.inf

        # self.kk_mtx = w_kk.reshape((self.num_cluster, self.num_cluster))
        kk_mtx = w_kk.reshape((self.num_cluster, self.num_cluster))
        self.kk_mtx = torch.tensor(kk_mtx, device=device)

        # cluster_emb_0 = centers_0[codes_0]
        # cluster_emb_1 = centers_1[codes_1]
        # cluster_emb = np.concatenate((cluster_emb_0, cluster_emb_1), axis=1)
    
    def sample(self, query_emb, pos_items=None):
        rat_centers_1 = torch.exp(torch.matmul(query_emb[self.cluster_dim:], self.centers_1.T))
        # rat_centers_1 = torch.matmul(self.user_embs[user_id][self.cluster_dim:], self.centers_1.T)
        
        tmp_weight_c0 = torch.matmul(self.kk_mtx, rat_centers_1)

        rat_centers_0 = torch.exp(torch.matmul(query_emb[:self.cluster_dim], self.centers_0.T))
        # rat_centers_0 = torch.matmul(self.user_embs[user_id][:self.cluster_dim], self.centers_0.T)

        # p_centers_0 = torch.softmax(tmp_weight_c0 * rat_centers_0, dim=-1)
        p_centers_0 = tmp_weight_c0 * rat_centers_0
        p_centers_0 = p_centers_0 / torch.sum(p_centers_0, dim=-1) 
        c0 = torch.multinomial(p_centers_0, self.num_neg, replacement=True)
        final_p0 = p_centers_0[c0]

        # p_centers_1 = torch.softmax(self.kk_mtx[c0] * rat_centers_1, dim=-1)
        p_centers_1 = self.kk_mtx[c0] * rat_centers_1
        p_centers_1 = p_centers_1 / torch.sum(p_centers_1, dim=-1, keepdim=True)
        c1 = torch.multinomial(p_centers_1, 1, replacement=True)
        final_p1 = torch.gather(p_centers_1, 1, c1).squeeze(-1)
        idx_final_cluster = c0 * self.num_cluster + c1.squeeze()
        idx_final_cluster = idx_final_cluster.cpu().data.numpy()

        item_cnt = np.float32(self.combine_cluster.indptr[idx_final_cluster + 1] - self.combine_cluster.indptr[idx_final_cluster])


        item_idx = np.int32(np.floor(item_cnt * np.random.rand(self.num_neg)))

        final_items = self.combine_cluster.indices[ item_idx  + self.combine_cluster.indptr[idx_final_cluster]]

        # final_probs = -np.log(item_cnt)
        # final_probs += np.log(final_p0.cpu().data.numpy())
        # final_probs += np.log(final_p1.cpu().data.numpy())
        final_probs = -torch.log(torch.tensor(item_cnt, device=c0.device))
        final_probs += torch.log(final_p0)
        final_probs += torch.log(final_p1)


        if pos_items is not None:
            clusters_idx = self.combine_cluster[pos_items, :].nonzero()[1]
            k_0 = torch.LongTensor(clusters_idx // self.num_cluster).to(c0.device)
            k_1 = torch.LongTensor(clusters_idx % self.num_cluster).to(c0.device)
            pos_prob0 = p_centers_0[k_0]
            p_centers_k1 = self.kk_mtx[k_0] * rat_centers_1

            p_centers_k1 = p_centers_k1 / torch.sum(p_centers_k1, dim=-1, keepdim=True)
            pos_prob1 = torch.gather(p_centers_k1, 1, k_1.unsqueeze(-1)).squeeze(-1)
            item_cnt = self.kk_mtx[k_0, k_1]
            final_pos_probs = -torch.log(item_cnt)
            final_pos_probs += 
            # raise NotImplementedError
        return final_items, final_probs



def setup_seed(seed):
    import os
    os.environ['PYTHONHASHSEED']=str(seed)

    import random
    random.seed(seed)
    np.random.seed(seed)

from dataloader import RecData, Sampler_Dataset
from torch.utils.data import DataLoader
if __name__ == "__main__":
    data = RecData('datasets', 'ml100k')
    train, test = data.get_data(0.8)
    dim = 32
    user_num, item_num = train.shape
    # user_emb = np.load('u.npy')
    # item_emb = np.load('v.npy')
    setup_seed(10)
    import torch
    
    # device = torch.device(2)
    device = torch.device('cuda')

    # user_emb, item_emb = np.random.randn(user_num, dim), np.random.randn(item_num, dim)
    user_emb = torch.rand((user_num, dim), device=device) * 0.01
    item_emb = np.random.randn(item_num, dim)
    sampler = MidxUniform(train, 1956, user_emb, item_emb, 9, device)
    query_emb = torch.rand(dim, device=device)
    # items, probs = sampler.sample(4)
    import time
    t0 = time.time()
    # for idx in range(10000):
    for idx in range(1):
        items, probs = sampler.sample(query_emb, [9,2])
    print(time.time()- t0)
        

    
    