# The cluster algorithmn(K-means) is implemented on the GPU
from operator import imod, neg
from numpy.core.numeric import indices
import scipy.sparse as sps
from sklearn import cluster
from sklearn.cluster import KMeans
import torch
import numpy as np
import torch.nn as nn
from torch._C import device, dtype


def kmeans(X, K_or_center, max_iter=300, verbose=False):
    N = X.size(0)
    if isinstance(K_or_center, int) is True:
        K = K_or_center
        C = X[torch.randperm(N, device=X.device)[:K]]
    else:
        K = K_or_center.size(0)
        C = K_or_center
    prev_loss = np.inf
    for iter in range(max_iter):
        dist = torch.sum(X * X, dim=-1, keepdim=True) - 2 * (X @ C.T) + torch.sum(C * C, dim=-1).unsqueeze(0)
        assign = dist.argmin(-1)
        assign_m = torch.zeros(N, K, device=X.device)
        assign_m[(range(N), assign)] = 1
        loss = torch.sum(torch.square(X - C[assign,:])).item()
        if verbose:
            print(f'step:{iter:<3d}, loss:{loss:.3f}')
        if (prev_loss - loss) < prev_loss * 1e-6:
            break
        prev_loss = loss
        cluster_count = assign_m.sum(0)
        C =  (assign_m.T @ X) / cluster_count.unsqueeze(-1)
        empty_idx = cluster_count<.5
        ndead = empty_idx.sum().item()
        C[empty_idx] = X[torch.randperm(N, device=X.device)[:ndead]]
    return C, assign, assign_m, loss

def construct_index(cd01, K):
    # Stable is availabel in PyTorch 1.9. Earlier version is not supported.
    cd01, indices = torch.sort(cd01, stable=True)
    # save the indices according to the cluster 
    cluster, count = torch.unique_consecutive(cd01, return_counts=True)
    count_all = torch.zeros(K**2 + 1, dtype=torch.long, device=cd01.device)
    count_all[cluster + 1] = count
    indptr = count_all.cumsum(dim=-1)
    return indices, indptr

class SamplerBase(nn.Module):
    """
        Uniformly Sample negative items for each query. 
    """
    def __init__(self, num_items, num_neg, device, **kwargs):
        super().__init__()
        self.num_items = num_items
        self.num_neg = num_neg
        self.device = device
    
    def forward(self, query, pos_items=None, padding=0):
        """
        Input
            query: torch.tensor
                Sequential models:
                query: (B,L,D), pos_items : (B, L)
                Normal models:
                query: (B,D), pos_items: (B,L)
        Output
            pos_prob(None if no pos_items), neg_items, neg_prob
            pos_items.shape == pos_prob.shape
            neg_items.shape == neg_prob.shape
            Sequential models:
            neg_items: (B,L,N)
            Normal
        """
        assert padding == 0
        num_queries = np.prod(query.shape[:-1]) # for sequential models the number of queries is the B x L
        neg_items = torch.randint(1, self.num_items + 1, size=(num_queries, self.num_neg), device=self.device)
        neg_items = neg_items.view(*query.shape[:-1], -1)
        neg_prob = -torch.log(self.num_items * torch.ones_like(neg_items, dtype=torch.float))
        if pos_items is not None:
            pos_prob = -torch.log(self.num_items * torch.ones_like(pos_items, dtype=torch.float))
            return pos_prob, neg_items, neg_prob
        return None, neg_items, neg_prob

class PopularSampler(SamplerBase):
    def __init__(self, pop_count, num_neg, device, mode=0, **kwargs):
        super().__init__(pop_count.shape[0], num_neg, device)
        pop_count = torch.from_numpy(pop_count).to(self.device)
        if mode == 0:
            pop_count = torch.log(pop_count + 1)
        elif mode == 1:
            pop_count = torch.log(pop_count) + 1e-16
        elif mode == 2:
            pop_count = pop_count**0.75
        
        pop_count = torch.cat([torch.zeros(1, device=self.device), pop_count])
        self.pop_prob = pop_count / pop_count.sum()
        self.table = torch.cumsum(self.pop_prob, -1)
        self.pop_prob[0] = torch.ones(1, device=self.device)
        

    def forward(self, query, pos_items=None, padding=0):
        assert padding == 0
        num_queris = np.prod(query.shape[:-1])
        seeds = torch.rand(num_queris, self.num_neg, device=self.device)
        neg_items = torch.searchsorted(self.table, seeds)
        neg_items = neg_items.view(*query.shape[:-1], -1)
        neg_prob = torch.log(self.pop_prob[neg_items])
        if pos_items is not None:
            pos_prob = torch.log(self.pop_prob[pos_items])
            return pos_prob, neg_items, neg_prob
        return None, neg_items, neg_prob

class MidxUniform(SamplerBase):
    """
        Midx Sampler with Uniform Variant
    """

    def __init__(self, item_embs:torch.tensor, num_neg, device, num_cluster, item_pop:torch.tensor = None, **kwargs):
        super().__init__(item_embs.shape[0], num_neg, device)
        if isinstance(num_cluster, int) is True:
            self.K = num_cluster
        else:
            self.K = num_cluster.size(0)
        
        embs1, embs2 = torch.chunk(item_embs, 2, dim=-1)
        self.c0, cd0, cd0m, _ = kmeans(embs1, num_cluster)
        self.c1, cd1, cd1m, _ = kmeans(embs2, num_cluster)

        self.c0_ = torch.cat([torch.zeros(1, self.c0.size(1), device=self.device), self.c0], dim=0) ## for retreival probability, considering padding
        self.c1_ = torch.cat([torch.zeros(1, self.c1.size(1), device=self.device), self.c1], dim=0) ## for retreival probability, considering padding

        self.cd0 = torch.cat([torch.tensor([-1]).to(self.device), cd0], dim=0) + 1 ## for retreival probability, considering padding
        self.cd1 = torch.cat([torch.tensor([-1]).to(self.device), cd1], dim=0) + 1 ## for retreival probability, considering padding

        cd01 = cd0 * self.K + cd1
        self.indices, self.indptr = construct_index(cd01, self.K)

        if item_pop is None:
            self.wkk = cd0m.T @ cd1m 
        else:
            self.wkk = cd0m.T @ (cd1m * item_pop.view(-1, 1))
    
    def forward(self, query, pos_items=None, padding=0):
        assert padding == 0
        q0, q1 = query.view(-1, query.size(-1)).chunk(2, dim=-1)
        r1 = q1 @ self.c1.T
        r1s = torch.softmax(r1, dim=-1) # num_q x K1
        r0 = q0 @ self.c0.T
        r0s = torch.softmax(r0, dim=-1) # num_q x K0
        s0 = (r1s @ self.wkk.T) * r0s # num_q x K0 | wkk: K0 x K1
        k0 = torch.multinomial(s0, self.num_neg, replacement=True) # num_q x neg
        p0 = torch.gather(r0, -1, k0)     # num_q * neg
            
        subwkk = self.wkk[k0, :]          # num_q x neg x K1
        s1 = subwkk * r1s.unsqueeze(1)     # num_q x neg x K1
        k1 = torch.multinomial(s1.view(-1, s1.size(-1)), 1).squeeze(-1).view(*s1.shape[:-1]) # num_q x neg
        p1 = torch.gather(r1, -1, k1) # num_q x neg
        k01 = k0 * self.K + k1  # num_q x neg
        p01 = p0 + p1
        neg_items, neg_prob = self.sample_item(k01, p01)
        if pos_items is not None:
            pos_prop = self.compute_item_p(query, pos_items)
            return pos_prop, neg_items.view(*query.shape[:-1], -1), neg_prob.view(*query.shape[:-1], -1)
        return None, neg_items.view(*query.shape[:-1], -1), neg_prob.view(*query.shape[:-1], -1)

    def sample_item(self, k01, p01):
        item_cnt = self.indptr[k01 + 1] - self.indptr[k01] # num_q x neg, the number of items
        item_idx = torch.floor(item_cnt * torch.rand_like(item_cnt, dtype=torch.float32, device=self.device)).long() # num_q x neg
        neg_items = self.indices[item_idx  + self.indptr[k01]] + 1
        neg_prob = p01
        return neg_items, neg_prob
    
    def compute_item_p(self, query, pos_items):
        # query: B x L x D, pos_items: B x L || query: B x D, pos_item: B x L1 || assume padding=0
        k0 = self.cd0[pos_items] # B x L || B x L1
        k1 = self.cd1[pos_items] # B x L || B x L1
        c0 = self.c0_[k0, :] # B x L x D || B x L1 x D
        c1 = self.c1_[k1, :] # B x L x D || B x L1 x D
        q0, q1 = query.chunk(2, dim=-1) # B x L x D || B x D
        if query.dim() == pos_items.dim():
            r = (torch.bmm(c0, q0.unsqueeze(-1)) + torch.bmm(c1, q1.unsqueeze(-1))).squeeze(-1) # B x L1
        else:
            r = torch.sum(c0 * q0, dim=-1) + torch.sum(c1 * q1, dim=-1) # B x L
        return r



class MidxUniPop(MidxUniform):
    """
    Popularity sampling for the final items
    """
    def __init__(self, item_embs: np.ndarray, num_neg, device, num_cluster, pop_count, mode=1, **kwargs):
        if mode == 0:
            pop_count = np.log(pop_count + 1)
        elif mode == 1:
            pop_count = np.log(pop_count + 1) + 1e-6
        elif mode == 2:
            pop_count = pop_count**0.75
        pop_count = torch.tensor(pop_count, dtype=torch.float32, device=device)
        super(MidxUniPop, self).__init__(item_embs, num_neg, device, num_cluster, pop_count)
        
        self.p = torch.cat([torch.ones(1, device=self.device), pop_count], dim=0)  # this is similar, to avoid log 0 !!! in case of zero padding 
        self.cp = pop_count[self.indices]
        for c in range(self.K**2):
            start, end = self.indptr[c], self.indptr[c+1]
            if end > start:
                cumsum = self.cp[start:end].cumsum(-1)
                self.cp[start:end] = cumsum / cumsum[-1]
        

    def forward(self, query, pos_items=None, padding=0):
        return super().forward(query, pos_items=pos_items, padding=padding)

    def sample_item(self, k01, p01):
        # k01 num_q x neg, p01 num_q x neg
        start = self.indptr[k01]
        last = self.indptr[k01 + 1] - 1
        count = last - start + 1
        maxlen = count.max()
        # print(maxlen)
        fullrange = start.unsqueeze(-1) + torch.arange(maxlen, device=self.device).reshape(1, 1, maxlen) # num_q x neg x maxlen
        fullrange = torch.minimum(fullrange, last.unsqueeze(-1))
        item_idx = torch.searchsorted(self.cp[fullrange], torch.rand_like(start, dtype=torch.float32, device=self.device).unsqueeze(-1)).squeeze(-1) ## num_q x neg
        item_idx = torch.minimum(item_idx, last)
        neg_items = self.indices[item_idx + self.indptr[k01]] + 1
        # neg_probs = self.p[item_idx + self.indptr[k01] + 1] # plus 1 due to considering padding, since p include num_items + 1 entries
        neg_probs = self.p[neg_items]
        return  neg_items, p01 + torch.log(neg_probs)
    
    def compute_item_p(self, query, pos_items):
        r = super().compute_item_p(query, pos_items)
        p_r = self.p[pos_items]
        return r + torch.log(p_r)

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "5"
    device = 'cuda'
    from dataloader import RecData
    from utils import setup_seed

    setup_seed(10)
    data = RecData('datasets', 'amazoni')
    train, test = data.get_data(0.8)
    dim = 200
    user_num, num_items = train.shape
    num_neg = 2000
    num_cluster = 32
    max_iter = 200
    # item_embs = np.random.randn(num_items, dim)
    item_embs = torch.randn(num_items,dim, device=device) * 0.1
    pop_count = np.squeeze(train.sum(axis=0).A)
    device = torch.device(device)
    # sampler0 = SamplerBase(num_items, num_neg, device)
    # sampler1 = PopularSampler(pop_count, num_neg, device)
    sampler2 = MidxUniform(item_embs, num_neg, device, num_cluster)
    sampler3 = MidxUniPop(item_embs, num_neg, device, num_cluster, pop_count)
    batch_size = 1
    query = torch.randn(batch_size, dim, device=device) * 0.1
    
    # pop_item = torch.randint(0, num_items+1, size=(batch_size))
    # sampler0(query, pop_item)
    # sampler1(query, pop_item)
    # sampler2(query, pop_item)
    # sampler3(query, pop_item)
    count_tensor = torch.zeros(num_items, dtype=torch.long, device=device)
    for i in range(max_iter):
        _, neg_items, _ = sampler2(query)
        # _, neg_items, _ = sampler3(query)
        ids, counts = torch.unique(neg_items -1, return_counts=True)
        count_tensor[ids] += counts
        # print(count_tensor.max(), counts.max())
    count_t = count_tensor / count_tensor.sum(-1)

    exact_prob = torch.softmax( torch.matmul(query, item_embs.T), dim=-1).squeeze()



    # =========================================
    # Plot prob
    item_ids = pop_count.argsort()

    exact_prob = exact_prob.cpu().data.numpy()
    count_prob = count_t.cpu().data.numpy()
    import matplotlib.pyplot as plt
    plt.plot(exact_prob[item_ids].cumsum(), label='Softmax', linewidth=4.0)
    plt.plot(count_prob[item_ids].cumsum(), label='Midx_Uni')
    plt.legend()
    plt.savefig('amazoni_check.jpg')





    




