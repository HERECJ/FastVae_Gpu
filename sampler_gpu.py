from operator import imod, neg
import scipy.sparse as sps
from sklearn.cluster import KMeans
import torch
import numpy as np
import torch.nn as nn

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

    def __init__(self, item_embs:np.ndarray, num_neg, device, num_cluster, **kwargs):
        super().__init__(item_embs.shape[0], num_neg, device)
        self.K = num_cluster
        item_embs = item_embs.cpu().data.numpy()
        embs1, embs2 = np.array_split(item_embs, 2, axis=-1)
        cluster_kmeans_0 = KMeans(n_clusters=self.K, random_state=0).fit(embs1)
        self.c0 = torch.tensor(cluster_kmeans_0.cluster_centers_.T, dtype=torch.float32, device=self.device)
        cd0 = cluster_kmeans_0.labels_
        cluster_kmeans_1 = KMeans(n_clusters=self.K, random_state=0).fit(embs2)
        self.c1 = torch.tensor(cluster_kmeans_1.cluster_centers_.T, dtype=torch.float32, device=self.device)
        cd1 = cluster_kmeans_1.labels_
        self.c0_ = torch.cat([torch.zeros(self.c0.size(0), 1, device=self.device), self.c0], dim=1) ## for retreival probability, considering padding
        self.c1_ = torch.cat([torch.zeros(self.c1.size(0), 1, device=self.device), self.c1], dim=1) ## for retreival probability, considering padding
        self.cd0 = torch.tensor(np.insert(cd0, 0, -1), dtype=torch.long, device=self.device) ## for retreival probability, considering padding
        self.cd1 = torch.tensor(np.insert(cd1, 0, -1), dtype=torch.long, device=self.device) ## for retreival probability, considering padding
        cd01 = cd0 * self.K + cd1
        self.member = sps.csc_matrix((np.ones_like(cd01), (np.arange(self.num_items), cd01)), \
            shape=(self.num_items, self.K**2))
        self.indices = torch.tensor(self.member.indices, dtype=torch.long, device=self.device)
        self.indptr = torch.from_numpy(self.member.indptr).to(self.device)
        self.wkk = torch.tensor(np.sum(self.member, axis=0).A, dtype=torch.float32, device=self.device).reshape(self.K, self.K)
    
    def forward(self, query, pos_items=None, padding=0):
        assert padding == 0
        q0, q1 = query.view(-1, query.size(-1)).chunk(2, dim=-1)
        r1 = q1 @ self.c1
        r1s = torch.softmax(r1, dim=-1) # num_q x K1
        r0 = q0 @ self.c0
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
        c0 = self.c0_.T[k0, :] # B x L x D || B x L1 x D
        c1 = self.c1_.T[k1, :] # B x L x D || B x L1 x D
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
        super().__init__(item_embs, num_neg, device, num_cluster)
        if mode == 0:
            pop_count = np.log(pop_count + 1)
        elif mode == 1:
            pop_count = np.log(pop_count + 1) + 1e-6
        elif mode == 2:
            pop_count = pop_count**0.75
        
        pop_diag = sps.csr_matrix((pop_count, (np.arange(self.num_items), np.arange(self.num_items))), shape=(self.num_items, self.num_items))
        member_pop = pop_diag * self.member
        w_kk = member_pop.sum(axis=0).A
        member_pop_norm = sps.csr_matrix(( 1.0/(np.squeeze(w_kk) + np.finfo(float).eps), \
            (np.arange(self.K**2), np.arange(self.K**2))), shape=(self.K**2, self.K**2))
        pop_probs_mat = (member_pop * member_pop_norm).tocsc()
        self.indptr = torch.from_numpy(pop_probs_mat.indptr).to(self.device)
        self.indices = torch.tensor(pop_probs_mat.indices, dtype=torch.long, device=self.device)
        self.p = torch.from_numpy(np.insert(pop_count, 0, 1.0)).to(self.device)
        cp = np.zeros_like(pop_probs_mat.data)
        for c in range(self.K**2):
            start, end = pop_probs_mat.indptr[c], pop_probs_mat.indptr[c+1]
            if end > start:
                cp[start:end] = pop_probs_mat.data[start:end].cumsum()
        self.cp = torch.from_numpy(cp).to(self.device)
        self.wkk = torch.tensor(w_kk, dtype=torch.float32, device=self.device).reshape(self.K, self.K)

    def forward(self, query, pos_items=None, padding=0):
        return super().forward(query, pos_items=pos_items, padding=padding)

    def sample_item(self, k01, p01):
        # k01 num_q x neg, p01 num_q x neg
        start = self.indptr[k01]
        last = self.indptr[k01 + 1] - 1
        count = last - start + 1
        maxlen = count.max()
        print(maxlen)
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
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    device = 'cuda'
    from dataloader import RecData
    data = RecData('datasets', 'ml10M')
    train, test = data.get_data(0.8)
    dim = 28
    user_num, num_items = train.shape
    num_neg = 31
    num_cluster = 41
    item_embs = np.random.randn(num_items, dim)
    pop_count = np.squeeze(train.sum(axis=0).A)
    
    sampler0 = SamplerBase(num_items, num_neg, device)
    sampler1 = PopularSampler(pop_count, num_neg, device)
    sampler2 = MidxUniform(item_embs, num_neg, device, num_cluster)
    sampler3 = MidxUniPop(item_embs, num_neg, device, num_cluster, pop_count)
    batch_size = 23
    query = torch.randn(batch_size, 201, dim, device=torch.device(device))
    
    pop_item = torch.randint(0, num_items+1, size=(batch_size, 201))
    sampler0(query, pop_item)
    sampler1(query, pop_item)
    sampler2(query, pop_item)
    sampler3(query, pop_item)


    




