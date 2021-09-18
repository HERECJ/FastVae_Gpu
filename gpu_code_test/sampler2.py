import scipy.sparse as sps
#import scipy.special
from sklearn.cluster import KMeans
import torch
import numpy as np


class SamplerBase(torch.Module):
    """
    For each user, sample negative items
    """
    def __init__(self, num_items):
        self.num_items = num_items

    def forward(self, query, num_neg, pos_items=None, padding=0):
        assert padding == 0
        # query: B x L x D, pos_items: B x L || query: B x D, pos_item: B x L || assume padding=0
        # not deal with reject sampling
        with torch.no_grad():
            num_queries = np.prod(query.shape[:-1])
            neg_items = torch.randint(1, self.num_items + 1, size=(num_queries, num_neg)) # padding with zero
            neg_items = neg_items.reshape(query.shape[:-1], -1) # B x L x Neg || B x Neg
            neg_prob = self.compute_item_p(query, neg_items)
            pos_prob = self.compute_item_p(query, pos_items)
        return pos_prob, neg_items, neg_prob

    def compute_item_p(self, query, pos_items):
        return - torch.log(torch.ones_like(pos_items))
    


class PopularSamplerModel(SamplerBase):
    def __init__(self, pop_count, mode=0):
        super(PopularSamplerModel, self).__init__(pop_count.shape[0])
        with torch.no_grad():
            pop_count = torch.tensor(pop_count, dtype=torch.float)
            if mode == 0:
                pop_count = torch.log(pop_count + 1)
            elif mode == 1:
                pop_count = torch.log(pop_count + 1) + 1e-6
            elif mode == 2:
                pop_count = pop_count**0.75

            pop_count = torch.cat([torch.ones(1), pop_count]) ## adding a padding value
            self.pop_prob = pop_count / pop_count.sum()
            self.table = torch.cumsum(self.pop_prob)
            self.pop_count = pop_count

    def forward(self, query, num_neg, pos_items=None, padding=0):
        assert padding == 0
        with torch.no_grad():
            num_queries = np.prod(query.shape[:-1])
            seeds = torch.rand(num_queries, num_neg)
            neg_items = torch.searchsorted(self.table, seeds)
            neg_items = neg_items.reshape(query.shape[:-1], -1) # B x L x Neg || B x Neg
            neg_prob = self.compute_item_p(query, neg_items)
            pos_prob = self.compute_item_p(query, pos_items)
        return pos_prob, neg_items, neg_prob
    
    def compute_item_p(self, query, pos_items):
        return torch.log(self.pop_prob[pos_items])  # padding value with log(0)



class SoftmaxApprSamplerUniform(SamplerBase):
    """
    Uniform sampling for the final items
    """

    def __init__(self, item_embs, num_cluster):
        super(SoftmaxApprSamplerUniform, self).__init__(item_embs.shape[0])
        self.K = num_cluster
        embs1, embs2 = np.array_split(item_embs, 2, axis=-1)
        cluster_kmeans_0 = KMeans(n_clusters=self.K, random_state=0).fit(embs1)
        self.c0 = torch.from_numpy(cluster_kmeans_0.cluster_centers_.T)
        cd0 = cluster_kmeans_0.labels_
        cluster_kmeans_1 = KMeans(n_clusters=self.K, random_state=0).fit(embs2)
        self.c1 = torch.from_numpy(cluster_kmeans_1.cluster_centers_.T)
        cd1 = cluster_kmeans_1.labels_
        self.c0_ = torch.cat([torch.zeros(self.c0.size(0), 1), self.c0], dim=1) ## for retreival probability, considering padding
        self.c1_ = torch.cat([torch.zeros(self.c1.size(0), 1), self.c1], dim=1) ## for retreival probability, considering padding
        self.cd0 = torch.from_numpy(np.insert(cd0, 0, -1)) + 1 ## for retreival probability, considering padding
        self.cd1 = torch.from_numpy(np.insert(cd1, 0, -1)) + 1 ## for retreival probability, considering padding
        cd01 = cd0 * self.K + cd1
        self.member = sps.csc_matrix((np.ones_like(cd01), (np.arange(self.num_items), cd01)), \
            shape=(self.num_items, self.K**2))
        self.indices = torch.from_numpy(self.member.indices)
        self.indptr = torch.from_numpy(self.member.indptr)
        self.wkk = torch.from_numpy(np.sum(self.member, axis=0).A).reshape(self.K, self.K)
        
    def forward(self, query, num_neg, pos_items=None, padding=0):
        assert padding ==0
        with torch.no_grad:
            q0, q1 = query.view(-1, query.size(-1)).chunk(2, dim=-1)
            r1 = q1 @ self.c1
            r1s = torch.softmax(r1, dim=-1) # num_q x K1
            r0 = q0 @ self.c0
            r0s = torch.softmax(r0, dim=-1) # num_q x K0
            s0 = (r1s @ self.wkk.T) * r0s # num_q x K0 | wkk: K0 x K1
            k0 = torch.multinomial(s0, num_neg, replacement=True) # num_q x neg
            p0 = torch.gather(r0, -1, k0)     # num_q * neg
            subwkk = self.wkk[k0, :]          # num_q x neg x K1
            s1 = subwkk * r1s.unsqueeze(1)     # num_q x neg x K1
            k1 = torch.multinomial(s1.view(-1, s1.size(-1)), 1).squeeze(-1).view(*s1.shape[:-1]) # num_q x neg
            p1 = torch.gather(r1, -1, k1) # num_q x neg
            k01 = k0 * self.K + k1  # num_q x neg
            p01 = p0 + p1
            neg_items, neg_prob = self.sample_item(query, pos_items, k01, p01)
            pos_prop = self.compute_item_p(query, pos_items)
            return pos_prop, neg_items.view(*query.shape[:-1], -1), neg_prob.view(*query.shape[:-1], -1)

    def sample_item(self, k01, p01):
        item_cnt = self.indptr[k01 + 1] - self.indptr[k01] # num_q x neg, the number of items
        item_idx = torch.int32(torch.floor(item_cnt * torch.rand_like(item_cnt))) # num_q x neg
        neg_items = self.indices[item_idx  + self.indptr[k01]] + 1
        neg_prob = p01
        return neg_items, neg_prob
    
    def compute_item_p(self, query, pos_items):
        # query: B x L x D, pos_items: B x L || query: B x D, pos_item: B x L1 || assume padding=0
        k0 = self.cd0[pos_items] # B x L || B x L1
        k1 = self.cd1[pos_items] # B x L || B x L1
        c0 = self.c0_[:, k0] # B x L x D || B x L1 x D
        c1 = self.c1_[:, k1] # B x L x D || B x L1 x D
        q0, q1 = query.chunk(2, dim=-1) # B x L x D || B x D
        if query.dim() == pos_items.dim():
            r = torch.bmm(c0, q0.unsqueeze(-1)) + torch.bmm(c1, q1.unsqueeze(-1)).squeeze(-1) # B x L1
        else:
            r = torch.sum(c0 * q0, dim=-1) + torch.sum(c1 * q1, dim=-1) # B x L
        return r

class SoftmaxApprSamplerPop(SoftmaxApprSamplerUniform):
    """
    Popularity sampling for the final items
    """
    def __init__(self, item_embs, pop_count, num_cluster, mode=1):
        super(SoftmaxApprSamplerPop, self).__init__(item_embs, num_cluster)
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
        self.indptr = torch.from_numpy(pop_probs_mat.indptr)
        self.indices = torch.from_numpy(pop_probs_mat.indices)
        self.p = torch.from_numpy(np.insert(pop_count, 0, .0))
        cp = np.zeros_like(pop_probs_mat.data)
        for c in range(self.K**2):
            start, end = pop_probs_mat.indptr[c], pop_probs_mat.indptr[c+1]
            if end > start:
                cp[start:end] = pop_probs_mat.data[start:end].cumsum()
        self.cp = torch.from_numpy(cp)
        self.wkk = torch.from_numpy(w_kk).reshape(self.K, self.K)


    def sample_item(self, k01, p01):
        # k01 num_q x neg, p01 num_q x neg
        start = self.indptr[k01]
        last = self.indptr[k01 + 1] - 1
        count = last - start + 1
        maxlen = count.max()
        fullrange = start.unsqueeze(-1) + torch.arange(maxlen).reshape(1, 1, maxlen) # num_q x neg x maxlen
        fullrange = torch.minimum(fullrange, last.unsqueeze(-1))
        item_idx = torch.searchsorted(self.cp[fullrange], torch.rand_like(start).unsqueeze(-1)).squeeze(-1) ## num_q x neg
        item_idx = torch.minimum(item_idx, last)
        neg_items = self.indices[item_idx + self.indptr[k01]]
        neg_probs = self.p[item_idx + self.indptr[k01] + 1] # plus 1 due to considering padding, since p include num_items + 1 entries
        return  neg_items, p01 + np.log(neg_probs)
    
    def compute_item_p(self, query, pos_items):
        r = super().compute_item_p(query, pos_items)
        p_r = self.p[pos_items]
        return r + np.log(p_r)
