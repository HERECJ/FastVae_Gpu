import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class BaseVAE(nn.Module):
    def __init__(self, num_item, dims, active='relu', dropout=0.5):
        """
        dims is a list for latent dims
        """
        super(BaseVAE, self).__init__()
        self.num_item = num_item
        
        self.dims = dims
        assert len(dims) == 2, 'Not supported dims'
        self.encode_layer_0 = nn.Embedding(self.num_item + 1, dims[0], padding_idx=0)
        self.encode_layer_1 = nn.Linear(dims[0], dims[1] * 2)

        self.decode_layer_0 = nn.Linear(dims[1], dims[0])
        # self._Item_Embeddings = nn.Embedding(self.num_item + 1, dims[0], padding_idx=0)
        self._Item_Embeddings = nn.Linear(dims[0], self.num_item + 1)

        self.dropout = nn.Dropout(dropout)

        if active == 'relu':
            self.act = F.relu
        elif active == 'tanh':
            self.act == F.tanh
        elif active == 'sigmoid':
            self.act == F.sigmoid
        else:
            raise ValueError('Not supported active function')

    
    def encode(self, item_id):
        # item_id is padded
        count_nonzero = item_id.count_nonzero(dim=1).unsqueeze(-1) # batch_user * 1
        user_embs = self.encode_layer_0(item_id) # batch_user * dims
        user_embs = torch.sum(user_embs, dim=1) / count_nonzero.pow(0.5)
        user_embs = self.dropout(user_embs)
        
        h = self.act(user_embs)
        h = self.encode_layer_1(h)
        mu, logvar = h[:, :self.dims[1]], h[:, self.dims[1]:]
        return mu, logvar
    
    def decode(self, user_emb_encode, items):
        user_emb = self.decode_layer_0(user_emb_encode)
        # item_embs = self._Item_Embeddings(items)
        # item_embs = F.normalize(item_embs)
        return self._Item_Embeddings(user_emb)
    

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    
    def forward(self, pos_items, sampler=None):
        mu, logvar = self.encode(pos_items)
        z = self.reparameterize(mu, logvar)

        items = torch.arange(self.num_item + 1, device=z.device)
        self.pos_items = pos_items
        part_rats = self.decode(z, items)
        loss = self.loss_function(part_rats)

        return mu, logvar, loss
    
    def kl_loss(self, mu, log_var, anneal=1.0, reduction=False):
        if reduction is True:
            return -anneal * 0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim = 1), dim = 0)
        else:
            return -anneal * 0.5 * torch.sum(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim = 1), dim = 0)
    
    def loss_function(self, part_rats, prob_neg=None, pos_rats=None, prob_pos=None, reduction=False):
        # max_v, _ = torch.max(part_rats, dim=-1)
        # chuli = part_rats - max_v.unsqueeze(-1)
        # logits =  chuli  - torch.log(torch.sum(torch.exp(chuli), dim=-1)).unsqueeze(-1)

        logits = F.log_softmax(part_rats, dim=-1)
        idx_mtx = (self.pos_items > 0).double()
        if reduction is True:
            return -torch.sum(torch.gather(logits, 1, self.pos_items) * idx_mtx, dim=-1).mean()
        else:
            return -torch.sum(torch.gather(logits, 1, self.pos_items)* idx_mtx, dim=-1).sum()
 

    def _get_user_emb(self, user_his):
        user_emb, _ = self.encode(user_his)
        return self.decode_layer_0(user_emb)

    def _get_item_emb(self):
        return self._Item_Embeddings.weight[1:]
    
        
class VAE_Sampler(BaseVAE):
    def __init__(self, num_item, dims, active='relu', dropout=0.5):
        super(VAE_Sampler, self).__init__(num_item, dims, active=active, dropout=dropout)
        self._Item_Embeddings = nn.Embedding(self.num_item + 1, dims[0], padding_idx=0)
    
    def decode(self, user_emb_encode, items):
        user_emb = self.decode_layer_0(user_emb_encode)
        item_embs = self._Item_Embeddings(items)
        # return torch.matmul(user_emb.view(user_emb.shape[0], 1, -1), item_embs.transpose(1,2)).squeeze(1)
        # return user_emb.unsqueeze(1).bmm(item_embs.transpose(1,2)).squeeze(1)
        return (user_emb.unsqueeze(1) * item_embs).sum(-1)
        # return (user_emb.view(user_emb.shape[0], 1, -1) * item_embs).sum(-1)
        # return torch.einsum('ijk,ik->ij', item_embs, user_emb)


    def forward(self, pos_items, sampler):
        
        mu, logvar = self.encode(pos_items)
        z = self.reparameterize(mu, logvar)

        user_emb = self.decode_layer_0(z)
        with torch.no_grad():
            pos_prob, neg_items, neg_prob = sampler(user_emb, pos_items)

        pos_items_emb = self._Item_Embeddings(pos_items)
        neg_items_emb = self._Item_Embeddings(neg_items)
        
        pos_rat = (user_emb.unsqueeze(1) * pos_items_emb).sum(-1)
        neg_rat = (user_emb.unsqueeze(1) * neg_items_emb).sum(-1)
        loss = self.loss_function(neg_rat, neg_prob, pos_rat, pos_prob)
        return mu, logvar, loss
    
    def loss_function(self, part_rats, log_prob_neg=None, pos_rats=None, log_prob_pos=None, reduction=False):
        idx_mtx = (pos_rats != 0).double()
        new_pos = pos_rats - log_prob_pos.detach()
        new_neg = part_rats - log_prob_neg.detach()

        # parts_log_sum_exp = torch.logsumexp(new_neg, dim=-1).unsqueeze(-1)
        # final = torch.log( torch.exp(new_pos) + torch.exp(parts_log_sum_exp))
        parts_sum_exp = torch.sum(torch.exp(new_neg), dim=-1).unsqueeze(-1)
        final = torch.log(torch.exp(new_pos) + parts_sum_exp)
        
        if reduction is True:
            return torch.sum((- new_pos + final) * idx_mtx, dim=-1 ).mean()
        else:
            return torch.sum((- new_pos + final) * idx_mtx, dim=-1 ).sum()
        
        # idx_mtx = (pos_rats != 0).double()
        # new_pos = pos_rats - log_prob_pos.detach()
        # new_neg = part_rats - log_prob_neg.detach()
        # # new_pos[pos_rats==0] = -np.inf
        # logits = torch.log_softmax(torch.cat([new_pos, new_neg], dim=-1), dim=-1)
        
        # num_pos_item = pos_rats.shape[1]
        
        # if reduction is True:
        #     return -torch.sum( logits[:, :num_pos_item] * idx_mtx, dim=-1).mean()
        # else:
        #     return -torch.sum( logits * idx_mtx, dim=-1).sum()

