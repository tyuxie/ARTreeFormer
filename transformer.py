import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from model_zoo import TimeEmbedding, MeanStdPooling
import time
import math
    

class Transformer(nn.Module):
    def __init__(self, ntips, cfg):
        super().__init__()
        self.ntips = ntips
        self.hidden_dim, self.n_head = cfg.hidden_dim, cfg.n_head
        self.MHA = nn.MultiheadAttention(self.hidden_dim, self.n_head, batch_first=True)
        # self.MHAnorm= nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(ntips, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        self.readout = nn.Sequential(
            nn.Linear(self.hidden_dim*2, self.hidden_dim*2),
            nn.LayerNorm(self.hidden_dim*2),
            nn.ELU(),
            nn.Linear(self.hidden_dim*2, self.hidden_dim*2),
            nn.LayerNorm(self.hidden_dim*2),
            nn.ELU(),
            nn.Linear(self.hidden_dim*2,1)
        )
        # self.readout = nn.Sequential(
        #     nn.Linear(self.hidden_dim*2, self.hidden_dim),
        #     nn.LayerNorm(self.hidden_dim),
        #     nn.ELU(),
        #     nn.Linear(self.hidden_dim,1)
        # )
        self.time_embedding = TimeEmbedding(2 * self.hidden_dim)
        self.register_buffer('query', torch.eye(ntips-3, self.hidden_dim))
    
    def forward(self, node_features, edge_index, t):
        temb = self.time_embedding(t)
        batch_size, nnodes = edge_index.shape[0], edge_index.shape[1]
        # node_features = self.MHAnorm(node_features)
        node_features = self.mlp(node_features)
        # new_feature, _ = self.MHA(self.query[None, None, t-3].repeat(batch_size,1,1), node_features + temb[:,None,:self.hidden_dim], node_features+temb[:,None,self.hidden_dim:]) 
        new_feature, _ = self.MHA(self.query[None, None, t-3].repeat(batch_size,1,1), node_features, node_features) 

        child_info = node_features[:,:-1]
        parent_info = torch.gather(node_features,1, edge_index[:,:-1,0].unsqueeze(-1).expand(-1,-1,node_features.shape[-1]))
        edge_info = torch.max(child_info, parent_info)
        edge_info = torch.concat([new_feature.expand(-1,nnodes-1,-1), edge_info], dim=-1)
        logits = self.readout(edge_info+temb.unsqueeze(1)).squeeze(-1)
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        return logits


class AttnBlock(nn.Module):
    def __init__(self, hidden_dim, n_head):
        super().__init__()
        self.hidden_dim, self.n_head = hidden_dim, n_head
        self.MHA = nn.MultiheadAttention(self.hidden_dim, self.n_head, batch_first=True)
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), 
                                 nn.GELU(approximate='tanh'), 
                                 nn.Linear(self.hidden_dim, self.hidden_dim))
    
    def forward(self, x):
        x_norm1 = self.norm1(x)
        x_attn, _ = self.MHA(x_norm1, x_norm1, x_norm1)
        x = x + x_attn

        x_norm2 = self.norm2(x)
        x = x + self.mlp(x_norm2)
        return x