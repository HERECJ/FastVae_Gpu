import numpy as np
from numpy.lib.function_base import append
import scipy as sp
import scipy.sparse
import scipy.special
from sklearn.cluster import KMeans
import bisect
import random
import math
from alias_method import AliasTable as alias
# import torch
import time


def softamx_ny(arr):
    max_v = arr.max(axis=-1, keepdims=True)
    return np.exp(arr - max_v - np.log(np.sum(np.exp(arr - max_v), axis=-1, keepdims=True)))



class SamplerUserModel:
    """
    For each user, sample negative items
    """
    def __init__(self, mat, num_neg, user_embs, item_embs, num_cluster):
        self.mat = mat.tocsr()
        self.num_users = mat.shape[0]
        self.num_items = item_embs.shape[0]
        print(self.num_items)
        self.num_neg = num_neg
        self.preprocess_time = 0.0
        self.sample_time = 0.0


    def preprocess(self, user_id):
        # self.exist = set(self.mat.indices[j] for j in range(self.mat.indptr[user_id], self.mat.indptr[user_id + 1]))
        pass

    def __sampler__(self, user_id):
        def sample():
            return np.random.randint(0, self.num_items, size=self.num_neg), -np.log(np.ones(self.num_neg))
        return sample

    def negative_sampler(self, user_id):
        t0 = time.time()
        self.preprocess(user_id)
        t1 = time.time()
        sample = self.__sampler__(user_id)
        neg_item, neg_prob = sample()
        t2 = time.time()
        self.preprocess_time += t1 - t0
        # print(t1 - t0)
        self.sample_time += t2 - t1
        # print(self.preprocess_time, self.sample_time)
        pos_idx = self.mat[user_id].nonzero()[1] # array
        pos_prob = self.compute_item_p(user_id, pos_idx)
        
        return pos_idx, pos_prob, neg_item, neg_prob, np.array([t1-t0])
    
    def compute_item_p(self, user_id, item_list):
        return -np.log(np.ones(len(item_list)))
    
    def get_times(self):
        return self.preprocess_time, self.sample_time


class PopularSamplerModel(SamplerUserModel):
    def __init__(self, mat, num_neg, user_embs, item_embs, num_cluster, mode=0):
        super(PopularSamplerModel, self).__init__(mat, num_neg, user_embs, item_embs, num_cluster)
        pop_count = np.squeeze(mat.sum(axis=0).A)
        if mode == 0:
            pop_count = np.log(pop_count + 1)
        elif mode == 1:
            pop_count = np.log(pop_count + 1) + 1e-6
        elif mode == 2:
            pop_count = pop_count**0.75
        self.pop_prob = pop_count / np.sum(pop_count)
        self.table = alias.alias_table_construct(self.pop_prob)
        self.pop_count = pop_count

    def __sampler__(self, user_id):
        def sample():
            seeds = np.random.rand(self.num_neg)
            neg_items = alias.sample_from_alias_table_1d(self.table, seeds)
            probs = np.log(self.pop_prob[neg_items])
            return neg_items, probs
        return sample
    
    def compute_item_p(self, user_id, item_list):
        return np.log(self.pop_prob[item_list])

class ExactSamplerModel(SamplerUserModel):
    def __init__(self, mat, num_neg, user_embs, item_embs, num_cluster):
        super(ExactSamplerModel, self).__init__(mat, num_neg, user_embs, item_embs, num_cluster)
        self.user_embs = user_embs #.cpu().detach().numpy()
        self.item_embs = item_embs #.cpu().detach().numpy()

    def preprocess(self, user_id):
        super(ExactSamplerModel, self).preprocess(user_id)
        self.pred = self.user_embs[user_id] @ self.item_embs.T
        self.score = softamx_ny(self.pred)
        # self.score_cum = self.score.cumsum()
        # self.score_cum[-1] = 1.0
        self.table = alias.alias_table_construct(self.score)

        
    def __sampler__(self, user_id):
        def sample():
            # neg_items = np.zeros(self.num_neg, dtype=np.int32)
            # probs = np.zeros(self.num_neg)
            seeds = np.random.rand(self.num_neg)
            # for idx, s in enumerate(seeds):
            #     k = bisect.bisect(self.score_cum,s)
            #     p = self.pred[k]
            #     neg_items[idx] = k
            #     probs[idx] = p
            neg_items = alias.sample_from_alias_table_1d(self.table, seeds)
            probs = np.log(self.score[neg_items])
            return neg_items, probs
        return sample

    def compute_item_p(self, user_id, item_list):
        return np.log(self.score[item_list])


class SoftmaxApprSampler(SamplerUserModel):
    """
    PQ methods, each item vector is splitted into three parts
    """
    def __init__(self, mat, num_neg, user_embs, item_embs, num_cluster):
        super(SoftmaxApprSampler, self).__init__(mat, num_neg, user_embs, item_embs, num_cluster)
        self.num_cluster = num_cluster
        
        _, latent_dim = user_embs.shape
        
        cluster_dim = latent_dim // 2
        

        # user_embs = user_embs.cpu().data.detach().numpy()
        # item_embs = item_embs.cpu().data.detach().numpy()
        self.user_embs = user_embs


        cluster_kmeans_0 = KMeans(n_clusters=self.num_cluster, random_state=0).fit(item_embs[:,:cluster_dim])
        centers_0 = cluster_kmeans_0.cluster_centers_
        codes_0 = cluster_kmeans_0.labels_
        self.center_scores_0 = np.matmul(user_embs[:,:cluster_dim] , centers_0.T)
        

        cluster_kmeans_1 = KMeans(n_clusters=self.num_cluster, random_state=0).fit(item_embs[:,cluster_dim:])
        centers_1 =cluster_kmeans_1.cluster_centers_
        codes_1 = cluster_kmeans_1.labels_
        self.center_scores_1 = np.matmul(user_embs[:,cluster_dim:] , centers_1.T)


        self.union_idx = [codes_0[i] * num_cluster + codes_1[i] for i in range(self.num_items)]

        self.combine_cluster = sp.sparse.csc_matrix((np.ones_like(self.union_idx), (np.arange(self.num_items), self.union_idx)), shape=(self.num_items, self.num_cluster**2))
        combine_sum = np.sum(self.combine_cluster, axis=0).A
        self.idx_nonzero = combine_sum > 0


        cluster_emb_0 = centers_0[codes_0]
        cluster_emb_1 = centers_1[codes_1]
        cluster_emb = np.concatenate((cluster_emb_0, cluster_emb_1), axis=1)
        self.item_emb_res = item_embs - cluster_emb

    def preprocess(self, user_id):
        
        delta_rui = np.matmul(self.user_embs[user_id], self.item_emb_res.T)
        combine_tmp = sp.sparse.csr_matrix((np.exp(delta_rui), (np.arange(self.num_items), np.arange(self.num_items))), shape=(self.num_items, self.num_items))
        combine_mat = combine_tmp * self.combine_cluster

        # combine_max = np.max(combine_mat, axis=0).A
        # combine_norm = 


        # w_kk : \sum_{i\in K K'} exp(rui)
        w_kk = combine_mat.sum(axis=0).A
        w_kk[self.idx_nonzero] = np.log(w_kk[self.idx_nonzero])
        w_kk[np.invert(self.idx_nonzero)] = -np.inf

        self.kk_mtx = w_kk.reshape((self.num_cluster, self.num_cluster))
        
        r_centers_1 = self.center_scores_1[user_id]
        phi_k_tmp = self.kk_mtx  +  r_centers_1
        # self.p_table_1 = sp.special.softmax(phi_k_tmp, 1)

        self.p_table_1 = softamx_ny(phi_k_tmp)
        self.c_table_1 = alias.alias_table_construct_list(self.p_table_1)
        phi_k = np.sum(np.exp(phi_k_tmp), axis=1)

        r_centers_0 = self.center_scores_0[user_id]
        self.p_table_0 = softamx_ny(r_centers_0 + np.log(phi_k))
        self.c_table_0 = alias.alias_table_construct(self.p_table_0)

    
    def compute_item_p(self, user_id, item_list):
        clusters_idx = self.combine_cluster[item_list, :].nonzero()[1] # find the combine cluster idx
        k_0 = clusters_idx // self.num_cluster
        k_1 = clusters_idx % self.num_cluster
        
        p_0 = np.array(self.p_table_0[k_0])
        p_1 = self.p_table_1[k_0, k_1]

        frac = np.matmul(self.user_embs[user_id], self.item_emb_res[item_list].T)
        deno = self.kk_mtx[k_0, k_1]
        # p_r = np.exp(frac - deno)
        return np.log(p_0) + np.log(p_1) + frac - deno
        
    

    def comput_p(self, sampled_cluster, p_table):
        p_list = p_table[sampled_cluster]
        p_list_former = p_table[sampled_cluster - 1]
        return [ x - y  if x> y else x for x,y in zip(p_list, p_list_former)]

    def __sampler__(self, user_id):
        def sample():
            seeds_values = np.random.rand(4, self.num_neg)
            sampled_cluster_0 = alias.sample_from_alias_table_1d(self.c_table_0, seeds_values[0])
            p_0 = np.array(self.p_table_0[sampled_cluster_0])


            rand_idx_arr = np.int32(np.floor(self.num_cluster * seeds_values[3]))
            sampled_cluster_1 = alias.sample_from_alias_table_l(self.c_table_1, sampled_cluster_0, seeds_values[1], rand_idx_arr)
            
            p_1 = self.p_table_1[sampled_cluster_0,sampled_cluster_1]

            idx_final_cluster = sampled_cluster_0 * self.num_cluster + sampled_cluster_1 
            
            final_items = np.zeros(self.num_neg, dtype=np.int32)
            final_probs = np.zeros(self.num_neg)
            for i,c in enumerate(idx_final_cluster):
                items = [self.combine_cluster.indices[j] for j in range(self.combine_cluster.indptr[c], self.combine_cluster.indptr[c+1])]
                rui_items =  np.matmul(self.user_embs[user_id], self.item_emb_res[items].T)
                item_sample_idx, p = self.sample_final_items(rui_items)
                final_items[i] = items[item_sample_idx]
                final_probs[i] = p
            
            final_probs = np.log(p_0) + np.log(p_1) +  np.log(final_probs)
            return final_items, final_probs
        return sample

    def sample_final_items(self, scores, eps=1e-8, mode=1):
        pred = softamx_ny(scores)
        # pred = sp.special.softmax(scores)
        if mode == 0:
            # Gumbel noise
            us = np.random.rand(len(scores))
            tmp = scores - np.log(- np.log(us + eps) + eps)
            k = np.argmax(tmp)
            return k, pred[k] 
        elif mode == 1:
            score_cum = pred.cumsum()
            score_cum[-1] = 1.0
            k = bisect.bisect(score_cum, np.random.rand())
            return k, pred[k]

class SoftmaxApprSamplerUniform(SoftmaxApprSampler):
    """
    Uniform sampling for the final items
    """
    def __init__(self, mat, num_neg, user_embs, item_embs, num_cluster):
        super(SoftmaxApprSamplerUniform, self).__init__(mat, num_neg, user_embs, item_embs, num_cluster)
        
        w_kk = np.float64(self.combine_cluster.sum(axis=0).A)
        w_kk[self.idx_nonzero] = np.log(w_kk[self.idx_nonzero])
        w_kk[np.invert(self.idx_nonzero)] = -np.inf

        self.kk_mtx = w_kk.reshape((self.num_cluster, self.num_cluster))
        

    def preprocess(self, user_id):
        r_centers_1 = self.center_scores_1[user_id]
        phi_k_tmp = self.kk_mtx  +  r_centers_1
        self.p_table_1 = softamx_ny(phi_k_tmp)
        # print(self.p_table_1.dtype)
        self.c_table_1 = alias.alias_table_construct_list(self.p_table_1)
        # print((self.p_table_1>0).astype(float).sum())
        phi_k = np.sum(np.exp(phi_k_tmp), axis=1)

        r_centers_0 = self.center_scores_0[user_id]
        self.p_table_0 = softamx_ny(r_centers_0 + np.log(phi_k))
        self.c_table_0 = alias.alias_table_construct(self.p_table_0)
    
        
    def __sampler__(self, user_id):
        def sample():
            seeds_values = np.random.rand(4, self.num_neg)

            sampled_cluster_0 = alias.sample_from_alias_table_1d(self.c_table_0, seeds_values[0])
            p_0 = np.array(self.p_table_0[sampled_cluster_0])


            # rand_idx_arr = np.random.randint(0, self.num_cluster, self.num_neg)
            rand_idx_arr = np.int32(np.floor(self.num_cluster * seeds_values[3]))
            sampled_cluster_1 = alias.sample_from_alias_table_l(self.c_table_1, sampled_cluster_0, seeds_values[1], rand_idx_arr)
            

            p_1 = self.p_table_1[sampled_cluster_0,sampled_cluster_1]
            

            idx_final_cluster = sampled_cluster_0 * self.num_cluster + sampled_cluster_1

            item_cnt = np.float32(self.combine_cluster.indptr[idx_final_cluster + 1] - self.combine_cluster.indptr[idx_final_cluster])


            item_idx = np.int32(np.floor(item_cnt * seeds_values[2]))

            final_items = self.combine_cluster.indices[ item_idx  + self.combine_cluster.indptr[idx_final_cluster]]
            final_probs = -np.log(item_cnt)
            
            final_probs += np.log(p_0) + np.log(p_1)
            return final_items, final_probs
        return sample
    
    def compute_item_p(self, user_id, item_list):
        clusters_idx = self.combine_cluster[item_list, :].nonzero()[1] # find the combine cluster idx
        k_0 = clusters_idx // self.num_cluster
        k_1 = clusters_idx % self.num_cluster
        
        p_0 = np.array(self.p_table_0[k_0])
        p_1 = self.p_table_1[k_0, k_1]


        deno = self.kk_mtx[k_0, k_1]
        # p_r = np.exp( - deno)
        return np.log(p_0) + np.log(p_1) - deno

class SoftmaxApprSamplerPop(SoftmaxApprSampler):
    """
    Popularity sampling for the final items
    """
    def __init__(self, mat, num_neg, user_embs, item_embs, num_cluster, mode=1):
        super(SoftmaxApprSamplerPop, self).__init__(mat, num_neg, user_embs, item_embs, num_cluster)
        pop_count = np.squeeze(mat.sum(axis=0).A)
        append_items = np.ones(self.num_items - len(pop_count))
        pop_count = np.r_[pop_count, append_items]
        if mode == 0:
            pop_count = np.log(pop_count + 1)
        elif mode == 1:
            pop_count = np.log(pop_count + 1) + 1e-6
        elif mode == 2:
            pop_count = pop_count**0.75
        
        self.pop_count = pop_count
        
        
        combine_tmp = sp.sparse.csr_matrix((self.pop_count, (np.arange(self.num_items), np.arange(self.num_items))), shape=(self.num_items, self.num_items))
        combine_mat = combine_tmp * self.combine_cluster

        w_kk = combine_mat.sum(axis=0).A

        w_k_tmp = sp.sparse.csr_matrix(( 1.0/(np.squeeze(w_kk) + np.finfo(float).eps), (np.arange(self.num_cluster**2), np.arange(self.num_cluster**2))), shape=(self.num_cluster**2, self.num_cluster**2))
        
        self.pop_probs_mat = (combine_mat * w_k_tmp).tocsc()
        
        # self.pop_cum_mat = self.pop_probs_mat.copy()
        self.pop_alias_dic = [[]] * (self.num_cluster**2)
        for c in range(self.num_cluster**2):
            idx = range(self.pop_probs_mat.indptr[c], self.pop_probs_mat.indptr[c+1])
            item_prob = self.pop_probs_mat.data[idx]
            # item_cum_prob = np.cumsum(np.array(item_prob))
            self.pop_alias_dic[c] = alias.alias_table_construct(item_prob)
      
        
        w_kk[self.idx_nonzero] = np.log(w_kk[self.idx_nonzero])
        w_kk[np.invert(self.idx_nonzero)] = -np.inf

        self.kk_mtx = w_kk.reshape((self.num_cluster, self.num_cluster))



    def preprocess(self, user_id):
        r_centers_1 = self.center_scores_1[user_id]
        phi_k_tmp = self.kk_mtx  +  r_centers_1
        self.p_table_1 = softamx_ny(phi_k_tmp)
        # print(self.p_table_1.dtype)
        self.c_table_1 = alias.alias_table_construct_list(self.p_table_1)
        # print(self.p_table_1, self.c_table_1)
        # print((self.p_table_1>0).astype(float).sum())
        phi_k = np.sum(np.exp(phi_k_tmp), axis=1)

        r_centers_0 = self.center_scores_0[user_id]
        self.p_table_0 = softamx_ny(r_centers_0 + np.log(phi_k))
        self.c_table_0 = alias.alias_table_construct(self.p_table_0)

    # def preprocess(self, user_id):
    #     test_time_list  = []
    #     test_time_list.append(time.time())
    #     r_centers_1 = self.center_scores_1[user_id]
    #     test_time_list.append(time.time())
    #     phi_k_tmp = self.kk_mtx  +  r_centers_1
    #     test_time_list.append(time.time())
    #     self.p_table_1 = softamx_ny(phi_k_tmp)
    #     test_time_list.append(time.time())
    #     self.c_table_1 = alias.alias_table_construct_list(self.p_table_1)
    #     test_time_list.append(time.time())
    #     phi_k = np.sum(np.exp(phi_k_tmp), axis=1)
    #     test_time_list.append(time.time())
    #     r_centers_0 = self.center_scores_0[user_id]
    #     test_time_list.append(time.time())
    #     self.p_table_0 = softamx_ny(r_centers_0 + np.log(phi_k))
    #     test_time_list.append(time.time())
    #     self.c_table_0 = alias.alias_table_construct(self.p_table_0)
    #     test_time_list.append(time.time())
    #     print([ float(a-b) for a,b in zip(test_time_list[1:], test_time_list[:-1])])
        

    def __sampler__(self, user_id):
        def sample():
            seeds_values = np.random.rand(5, self.num_neg)
            sampled_cluster_0 = alias.sample_from_alias_table_1d(self.c_table_0, seeds_values[0])
            p_0 = np.array(self.p_table_0[sampled_cluster_0])

            # rand_idx_arr = np.random.randint(0, self.num_cluster, self.num_neg)
            rand_idx_arr = np.int32(np.floor(self.num_cluster * seeds_values[3]))
            sampled_cluster_1 = alias.sample_from_alias_table_l(self.c_table_1, sampled_cluster_0, seeds_values[1], rand_idx_arr) 
            
            p_1 = self.p_table_1[sampled_cluster_0,sampled_cluster_1]
            

            idx_final_cluster = sampled_cluster_0 * self.num_cluster + sampled_cluster_1
            
            # idx_items_lst = [[ self.combine_cluster.indices[j] for j in range(self.combine_cluster.indptr[c], self.combine_cluster.indptr[c+1])] for c in idx_final_cluster]

            final_items, final_probs = [0] * self.num_neg , [0.0] * self.num_neg
            
            len_arr = np.zeros(self.num_neg) 
            for i in range(self.num_neg):
                len_arr[i] = len(self.pop_alias_dic[idx_final_cluster[i]])
            rand_idx_arr = np.int32(np.floor(len_arr * seeds_values[4]))
            item_idx = alias.sample_from_alias_table_l(self.pop_alias_dic, idx_final_cluster, seeds_values[2], rand_idx_arr)

                    # num = len(alias_table_l[0])
        # rand_idx_arr = np.random.randint(0, num, sample_num)

            final_items = self.pop_probs_mat.indices[ item_idx  + self.pop_probs_mat.indptr[idx_final_cluster]]
            final_probs = self.pop_probs_mat.data[ item_idx  + self.pop_probs_mat.indptr[idx_final_cluster]]

            final_probs = np.log(p_0) + np.log(p_1) +  np.log(final_probs)
            return  final_items, final_probs
        return sample
    
    def compute_item_p(self, user_id, item_list):
        clusters_idx = self.combine_cluster[item_list, :].nonzero()[1] # find the combine cluster idx
        p_r = self.pop_probs_mat[item_list, clusters_idx].A.squeeze()
        k_0 = clusters_idx // self.num_cluster
        k_1 = clusters_idx % self.num_cluster
        
        p_0 = np.array(self.p_table_0[k_0])
        p_1 = self.p_table_1[k_0, k_1]
        
        # p_r =
        # p_r = self.pop_probs_mat[item_list,:].data
        # p_r = np.array(self.pop_probs_mat[item_list,:].data)
        return np.log(p_0) + np.log(p_1) + np.log(p_r)

class UniformSoftmaxSampler(SoftmaxApprSamplerUniform):
    def __init__(self, mat, num_neg, user_embs, item_embs, num_cluster):
        super(UniformSoftmaxSampler, self).__init__(mat, num_neg, user_embs, item_embs, num_cluster)
    
    def preprocess(self, user_id):
        r_centers_1 = self.center_scores_1[user_id]
        phi_k_tmp = self.kk_mtx  +  r_centers_1
        self.p_table_1 = softamx_ny(phi_k_tmp)
        self.c_table_1 = alias.alias_table_construct_list(self.p_table_1)


        phi_k = np.sum(np.exp(phi_k_tmp), axis=1)

        r_centers_0 = self.center_scores_0[user_id]
        self.p_table_0 = softamx_ny(r_centers_0 + np.log(phi_k))
        self.c_table_0 = alias.alias_table_construct(self.p_table_0)

        delta_rui = np.matmul(self.user_embs[user_id], self.item_emb_res.T)
        combine_tmp = sp.sparse.csr_matrix((np.exp(delta_rui), (np.arange(self.num_items), np.arange(self.num_items))), shape=(self.num_items, self.num_items))
        combine_mat = combine_tmp * self.combine_cluster

        # combine_max = np.max(combine_mat, axis=0).A
        # combine_norm = 


        # w_kk : \sum_{i\in K K'} exp(rui)
        w_kk = combine_mat.sum(axis=0).A
        w_kk[self.idx_nonzero] = np.log(w_kk[self.idx_nonzero])
        w_kk[np.invert(self.idx_nonzero)] = -np.inf

        self.kk_mtx_res = w_kk.reshape((self.num_cluster, self.num_cluster))

    
    def __sampler__(self, user_id):
        def sample():
            seeds_values = np.random.rand(4, self.num_neg)
            sampled_cluster_0 = alias.sample_from_alias_table_1d(self.c_table_0, seeds_values[0])
            p_0 = np.array(self.p_table_0[sampled_cluster_0])


            rand_idx_arr = np.random.randint(0, self.num_cluster, self.num_neg)
            rand_idx_arr = np.int32(np.floor(self.num_cluster * seeds_values[3]))
            sampled_cluster_1 = alias.sample_from_alias_table_l(self.c_table_1, sampled_cluster_0, seeds_values[1], rand_idx_arr)
            
            p_1 = self.p_table_1[sampled_cluster_0,sampled_cluster_1]

            idx_final_cluster = sampled_cluster_0 * self.num_cluster + sampled_cluster_1

            final_items = np.zeros(self.num_neg, dtype=np.int32)
            final_probs = np.zeros(self.num_neg)
            for i,c in enumerate(idx_final_cluster):
                items = [self.combine_cluster.indices[j] for j in range(self.combine_cluster.indptr[c], self.combine_cluster.indptr[c+1])]
                rui_items =  np.matmul(self.user_embs[user_id], self.item_emb_res[items].T)
                item_sample_idx, p = self.sample_final_items(rui_items)
                final_items[i] = items[item_sample_idx]
                final_probs[i] = p
            
            final_probs = np.log(p_0) + np.log(p_1) +  np.log(final_probs)
            return final_items, final_probs
        return sample

    def compute_item_p(self, user_id, item_list):
        clusters_idx = self.combine_cluster[item_list, :].nonzero()[1] # find the combine cluster idx
        k_0 = clusters_idx // self.num_cluster
        k_1 = clusters_idx % self.num_cluster
        
        p_0 = np.array(self.p_table_0[k_0])
        p_1 = self.p_table_1[k_0, k_1]

        frac = np.matmul(self.user_embs[user_id], self.item_emb_res[item_list].T)
        deno = self.kk_mtx_res[k_0, k_1]
        # p_r = np.exp(frac - deno)
        return np.log(p_0) + np.log(p_1) + frac - deno

class DynamicNegativeSampling(SamplerUserModel):
    def __init__(self,  mat, num_neg, user_embs, item_embs, num_cluster):
        self.mat = mat.tocsr()
        self.num_users, self.num_items = mat.shape
        self.num_neg = num_neg
        self.user = user_embs
        self.item = item_embs
        self.num_sample_uni = num_cluster

    def __sampler__(self, user_id):
        def sample():
            uniform_item = np.random.randint(0, self.num_items, size=(self.num_neg, self.num_sample_uni))
            scores = np.matmul(self.item[uniform_item], self.user[user_id])
            max_idx = np.argmax(scores, axis=-1)
            idx = np.arange(self.num_neg)
            items = uniform_item[idx, max_idx]
            return items, scores[idx, max_idx]
        return sample

    def compute_item_p(self, user_id, item_list):
        return np.matmul(self.item[item_list], self.user[user_id])

class Tree:
    def __init__(self, mat):
        self.depth = math.ceil(math.log2(mat.shape[0]))
        self.tree = [None] * self.depth
        self.tree[0] = mat
        for d in range(1, self.depth):
            child = self.tree[d-1]
            if child.shape[0] % 2 != 0:
                child = np.r_[child, np.zeros([1, child.shape[1]])]
            self.tree[d] = child[::2] + child[1::2]

    def sampling(self, vector, neg):
        rand_num_arr = np.random.rand(neg, self.depth)
        samples = np.zeros(neg, dtype=np.int32)
        prob = np.zeros(neg, dtype=np.float32)
        for i in range(neg):
            selected = 0
            rand_num = rand_num_arr[i]
            for d in range(self.depth, 0, -1):
                if self.tree[d-1].shape[0] > selected * 2 + 1:
                    score = np.matmul(self.tree[d-1][[selected * 2, selected * 2 + 1]], vector)
                    idx_child = 0 if score[0]/(score[0]+score[1]) > rand_num[d-1] else 1
                    selected = selected * 2 + idx_child
                else:
                    selected = selected * 2
            prob[i] = score[idx_child]
            samples[i] = selected
        return samples, prob


class KernelBasedSampling(SamplerUserModel):
    @staticmethod
    def getkernel(mat, alpha):
        phi_t = np.matmul(np.expand_dims(mat, axis=2), np.expand_dims(mat, axis=1))
        phi_ = np.reshape(phi_t, (mat.shape[0], -1)) #* math.sqrt(alpha)
        phi = np.c_[phi_, np.ones(mat.shape[0])]
        return phi

    def __init__(self,  mat, num_neg, user_embs, item_embs, num_cluster):
        self.mat = mat.tocsr()
        self.num_users, self.num_items = mat.shape
        self.num_neg = num_neg
        self.user = user_embs
        self.item = item_embs
        self.alpha = num_cluster
        # construct balance binary tree
        self.item_phi = KernelBasedSampling.getkernel(self.item, self.alpha)
        print(self.item_phi.shape)
        self.user_phi = KernelBasedSampling.getkernel(self.user, self.alpha)
        self.tree = Tree(self.item_phi)

    def __sampler__(self, user_id):
        def sample():
            return self.tree.sampling(self.user_phi[user_id], self.num_neg)
        return sample

    def compute_item_p(self, user_id, item_list):
        return np.matmul(self.item_phi[item_list], self.user_phi[user_id])



def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start_user
    overall_end = dataset.end_user
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start_user = overall_start + worker_id * per_worker
    dataset.end_user = min(dataset.start_user + per_worker, overall_end)

def get_max_length(x):
    return max(x, key=lambda x: x.shape[0]).shape[0]

def pad_sequence_int(seq):
    def _pad(_it, _max_len):
        return np.concatenate(( _it, np.zeros(_max_len - len(_it), dtype=np.int32) ))
    return [_pad(it, get_max_length(seq)) for it in seq]

def pad_sequence(seq):
    def _pad(_it, _max_len):
        return np.concatenate(( _it, np.zeros(_max_len - len(_it)) ))
    return [_pad(it, get_max_length(seq)) for it in seq]

def custom_collate(batch):
    import time
    t0 = time.time()
    transposed = zip(*batch)
    lst = []
    for samples in transposed:
        if type(samples[0]) in [np.int, np.int32, np.int64]:
               lst.append(torch.LongTensor(samples))
        else:
            if type(samples[0][0]) in [np.int, np.int32, np.int64]:
                lst.append(torch.LongTensor(pad_sequence_int(samples)))
            else:
                lst.append(torch.tensor(pad_sequence(samples)))
    # print("padding time: ", time.time() - t0)
    return lst

def setup_seed(seed):
    import os
    os.environ['PYTHONHASHSEED']=str(seed)

    import random
    random.seed(seed)
    np.random.seed(seed)
    

# def main(dataloader):

#     for batch_idx, data in enumerate(dataloader):
#         ruis, prob_pos, neg_id, prob = data
#         if batch_idx > 2:
#             break


# def main(train_sample):
#     import time
#     t0 = time.time()
#     tt0 = t0
#     for i in range(user_num):
#         k = train_sample[i]
#         # print(k)
#         if i > 20:
#             break
#     print(time.time() - t0)



from dataloader import RecData, Sampler_Dataset
from torch.utils.data import DataLoader
if __name__ == "__main__":
    data = RecData('datasets', 'ml100k')
    train, test = data.get_data(0.8)
    user_num, item_num = train.shape
    # user_emb = np.load('u.npy')
    # item_emb = np.load('v.npy')
    setup_seed(10)
    import torch
    user_emb, item_emb = np.random.randn(user_num, 32), np.random.randn(item_num, 32)
    # user_emb , item_emb = torch.tensor(user_emb), torch.tensor(item_emb)
    # sampler = UniformSoftmaxSampler(train, 10000, user_emb, item_emb, 25)
    sampler = SoftmaxApprSamplerUniform(train, 2000, user_emb, item_emb, 9)
    # sampler = ExactSamplerModel(train[:50], 5000, user_emb, item_emb, 25)
    import time
    t0 = time.time()
    for idx in range(10000):
        sampler.negative_sampler(4)
    print(time.time()- t0)

    # train_sample = Sampler_Dataset(sampler)
    # train_dataloader = DataLoader(train_sample, batch_size=16, num_workers=4, collate_fn=custom_collate)

    # for idx, data in enumerate(train_dataloader):
    #     ruis, prob_pos, neg_id, prob_neg = data
    #     import pdb; pdb.set_trace()
    