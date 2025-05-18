import torch
import torch.nn as nn
import math
from model_zoo import GNNStack, GatedGraphConv, IDConv, MeanStdPooling
import time

class GNN_BranchModel(nn.Module):          
    def __init__(self, ntips, cfg):
        super().__init__()
        self.ntips = ntips
        self.register_buffer('leaf_features', torch.eye(self.ntips))
        
        if cfg.gnn_type == 'identity':
            self.gnn = IDConv()
        elif cfg.gnn_type != 'ggnn':
            self.gnn = GNNStack(self.ntips, cfg.hidden_dim, num_layers=cfg.num_layers, bias=cfg.bias, gnn_type=cfg.gnn_type, aggr=cfg.aggr, project=cfg.project)
        else:
            self.gnn = GatedGraphConv(cfg.hidden_dim, num_layers=cfg.num_layers, bias=cfg.bias)
            
        if cfg.gnn_type == 'identity':
            self.mean_std_net = MeanStdPooling(self.ntips, cfg.hidden_dim, bias=cfg.bias)
        else:
            self.mean_std_net = MeanStdPooling(cfg.hidden_dim, cfg.hidden_dim, bias=cfg.bias)
        
    
    def fixpoint(self, edge_index: torch.LongTensor, tol=1e-5, max_iters=10000):
        '''
        edge_index describes the index of the adjacency nodes of a node. Its shape is (number of trees, number of nodes, 3).
        X is the initialization of the features of the nodes.
        (each row of X is initialized as [1/feature_dim]*feature_dim)
        '''
        bs = edge_index.shape[0]
        dim, nf = self.ntips-2, self.ntips
        assert dim + nf == edge_index.shape[1]
        X = torch.ones((bs, dim, nf), device=edge_index.device) / nf
        
        identity = self.leaf_features.unsqueeze(0).repeat(bs, 1, 1)

        t = time.time()
        for it in range(1, max_iters + 1):
            X_old = X.clone()
            neigh = torch.gather(torch.concat([identity, X], dim=1), dim=1, index=edge_index[:,nf:].reshape(bs, -1).unsqueeze(-1).repeat(1,1,nf))
            neigh = neigh.reshape(bs, dim, 3, nf)
            X = torch.mean(neigh, dim=2)
            Lnorm = torch.mean(torch.abs(X-X_old), dim=(1,2))

            if not torch.any(Lnorm > tol):
                break
        output = torch.concat([identity, X], dim=1)
        return output

    def _edge_index(self, tree):
        node_idx_list, edge_index = [], []           
        for node in tree.traverse('preorder'):
            neigh_idx_list = []
            if not node.is_root():
                neigh_idx_list.append(node.up.name)
                if not node.is_leaf():
                    neigh_idx_list.extend([child.name for child in node.children])
                else:
                    neigh_idx_list.extend([-1, -1])              
            else:
                neigh_idx_list.extend([child.name for child in node.children])
            
            edge_index.append(neigh_idx_list)
            node_idx_list.append(node.name)
        
        node_idx_list, edge_index = torch.tensor(node_idx_list).long().to(self.leaf_features.device), torch.tensor(edge_index).long().to(self.leaf_features.device)
        branch_idx_map = torch.sort(node_idx_list, dim=0, descending=False)[1]        

        return edge_index[branch_idx_map]
    
    
    def mean_std(self, tree, **kwargs):
        node_features, edge_index = self.node_embedding(tree)
        node_features = self.gnn(node_features, edge_index)

        return self.mean_std_net(node_features, edge_index[:-1, 0])
            
    
    def sample_branch_base(self, n_particles):
        samp_log_branch = torch.randn(size=(n_particles, 2*self.ntips-3), device=self.leaf_features.device)
        return samp_log_branch, torch.sum(-0.5*math.log(2*math.pi) - 0.5*samp_log_branch**2, -1)
    
    
    def forward(self, edge_index):
        bs, nnodes, _ = edge_index.shape
        node_features = self.fixpoint(edge_index)

        compact_node_features = node_features.view(-1, node_features.shape[-1])
        compact_edge_index = torch.where(edge_index>-1, edge_index + torch.arange(0, bs, device=self.leaf_features.device)[:,None,None]*nnodes, -1)
        compact_parent_index = compact_edge_index[:,:-1,0].contiguous().view(-1)
        compact_edge_index = compact_edge_index.view(-1, compact_edge_index.shape[-1])

        compact_node_features = self.gnn(compact_node_features, compact_edge_index)
        node_features = compact_node_features.view(bs, nnodes, compact_node_features.shape[-1])
        mean, std = self.mean_std_net(node_features, compact_parent_index)
        mean, std = mean.view(bs, nnodes-1), std.view(bs, nnodes-1)
        samp_log_branch, logq_branch = self.sample_branch_base(bs)
        samp_log_branch, logq_branch = samp_log_branch * std.exp() + mean - 2.0, logq_branch - torch.sum(std, -1)

        return samp_log_branch, logq_branch 