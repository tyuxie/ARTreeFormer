import torch
import torch.nn as nn
import numpy as np
import pdb
from ete3 import TreeNode

def decompJC(symm=False):
    # pA = pG = pC = pT = .25
    pden = np.array([.25, .25, .25, .25])
    rate_matrix_JC = 1.0/3 * np.ones((4,4))
    for i in range(4):
        rate_matrix_JC[i,i] = -1.0
    
    if not symm:
        D_JC, U_JC = np.linalg.eig(rate_matrix_JC)
        U_JC_inv = np.linalg.inv(U_JC)
    else:
        D_JC, W_JC = np.linalg.eigh(np.dot(np.dot(np.diag(np.sqrt(pden)), rate_matrix_JC), np.diag(np.sqrt(1.0/pden))))
        U_JC = np.dot(np.diag(np.sqrt(1.0/pden)), W_JC)
        U_JC_inv = np.dot(W_JC.T, np.diag(np.sqrt(pden)))
    
    return D_JC, U_JC, U_JC_inv, rate_matrix_JC


def decompHKY(pden, kappa, symm=False):
    pA, pG, pC, pT = pden
    beta = 1.0/(2*(pA+pG)*(pC+pT) + 2*kappa*(pA*pG+pC*pT))
    rate_matrix_HKY = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            if j!=i:
                rate_matrix_HKY[i,j] = pden[j]
            if i+j == 1 or i+j == 5:
                rate_matrix_HKY[i,j] *= kappa
    
    for i in range(4):
        rate_matrix_HKY[i,i] = - sum(rate_matrix_HKY[i,])
    
    rate_matrix_HKY = beta * rate_matrix_HKY
    
    if not symm:
        D_HKY, U_HKY = np.linalg.eig(rate_matrix_HKY)
        U_HKY_inv = np.linalg.inv(U_HKY)
    else:
        D_HKY, W_HKY = np.linalg.eigh(np.dot(np.dot(np.diag(np.sqrt(pden)), rate_matrix_HKY), np.diag(np.sqrt(1.0/pden))))
        U_HKY = np.dot(np.diag(np.sqrt(1.0/pden)), W_HKY)
        U_HKY_inv = np.dot(W_HKY.T, np.diag(np.sqrt(pden)))
       
    return D_HKY, U_HKY, U_HKY_inv, rate_matrix_HKY


def decompGTR(pden, AG, AC, AT, GC, GT, CT, symm=False):
    pA, pG, pC, pT = pden
    beta = 1.0/(2*(AG*pA*pG+AC*pA*pC+AT*pA*pT+GC*pG*pC+GT*pG*pT+CT*pC*pT))
    rate_matrix_GTR = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            if j!=i:
                rate_matrix_GTR[i,j] = pden[j]
                if i+j == 1:
                    rate_matrix_GTR[i,j] *= AG
                if i+j == 2:
                    rate_matrix_GTR[i,j] *= AC
                if i+j == 3 and abs(i-j) > 1:
                    rate_matrix_GTR[i,j] *= AT
                if i+j == 3 and abs(i-j) == 1:
                    rate_matrix_GTR[i,j] *= GC
                if i+j == 4:
                    rate_matrix_GTR[i,j] *= GT
                if i+j == 5:
                    rate_matrix_GTR[i,j] *= CT
    
    for i in range(4):
        rate_matrix_GTR[i,i] = - sum(rate_matrix_GTR[i,])
    
    rate_matrix_GTR = beta * rate_matrix_GTR
    
    if not symm:
        D_GTR, U_GTR = np.linalg.eig(rate_matrix_GTR)
        U_GTR_inv = np.linalg.inv(U_GTR)
    else:
        D_GTR, W_GTR = np.linalg.eigh(np.dot(np.dot(np.diag(np.sqrt(pden)), rate_matrix_GTR), np.diag(np.sqrt(1.0/pden))))
        U_GTR = np.dot(np.diag(np.sqrt(1.0/pden)), W_GTR)
        U_GTR_inv = np.dot(W_GTR.T, np.diag(np.sqrt(pden)))        
    
    return D_GTR, U_GTR, U_GTR_inv, rate_matrix_GTR

class PHY(nn.Module):
    nuc2vec = {'A':[1.,0.,0.,0.], 'G':[0.,1.,0.,0.], 'C':[0.,0.,1.,0.], 'T':[0.,0.,0.,1.],
           '-':[1.,1.,1.,1.], '?':[1.,1.,1.,1.], 'N':[1.,1.,1.,1.], 'R':[1.,1.,0.,0.],
           'Y':[0.,0.,1.,1.], 'S':[0.,1.,1.,0.], 'W':[1.,0.,0.,1.], 'K':[0.,1.,0.,1.],
           'M':[1.,0.,1.,0.], 'B':[0.,1.,1.,1.], 'D':[1.,1.,0.,1.], 'H':[1.,0.,1.,1.],
           'V':[1.,1.,1.,0.], '.':[1.,1.,1.,1.], 'U':[0.,0.,0.,1.]}
    
    def __init__(self, data, taxa, pden, subModel, scale=0.1, unique_site=True):
        super().__init__()
        self.ntips = len(data)
        self.nsites = len(data[0])
        self.taxa = taxa
        Qmodel, Qpara = subModel
        if Qmodel == "JC":
            D, U, U_inv, rateM = decompJC()  ##We use JC model in VBPI
        if Qmodel == "HKY":
            D, U, U_inv, rateM = decompHKY(pden, Qpara)
        if Qmodel == "GTR":
            AG, AC, AT, GC, GT, CT = Qpara
            D, U, U_inv, rateM = decompGTR(pden, AG, AC, AT, GC, GT, CT)
        
        self.register_buffer('pden', torch.from_numpy(pden).float())
        self.register_buffer('D', torch.from_numpy(D).float())
        self.register_buffer('U', torch.from_numpy(U).float())
        self.register_buffer('U_inv', torch.from_numpy(U_inv).float())
        
        if unique_site:
            L, site_counts = map(torch.FloatTensor, self.initialCLV(data, unique_site=True))
            # L = self.L.to(device=device)
            # self.site_counts = self.site_counts.to(device=device)
        else:
            L, site_counts = torch.FloatTensor(self.initialCLV(data)), torch.FloatTensor([1.0])
        self.register_buffer('L', L)
        self.register_buffer('site_counts', site_counts)
        self.scale = scale

    def initialCLV(self, data, unique_site=False):
        if unique_site:
            data_arr = np.array(list(zip(*data)))
            unique_sites, counts = np.unique(data_arr, return_counts=True, axis=0)
            unique_data = unique_sites.T
            
            return [np.transpose([self.nuc2vec[c] for c in unique_data[i]]) for i in range(self.ntips)], counts
        else:
            return [np.transpose([self.nuc2vec[c] for c in data[i]]) for i in range(self.ntips)]

    def logprior(self, log_branch):
        return -torch.sum(torch.exp(log_branch)/self.scale + np.log(self.scale) - log_branch, -1)
    
    def loglikelihood(self, log_branch, tree):
        branch_D = torch.einsum("i,j->ij", (log_branch.exp(), self.D))
        transition_matrix = torch.matmul(torch.einsum("ij,kj->kij", (self.U, torch.exp(branch_D))), self.U_inv).clamp(0.0)
        scaler_list = []
        for node in tree.traverse("postorder"):
            if node.is_leaf():
                node.state = self.L[node.name].detach()
            else:
                node.state = 1.0
                for child in node.children:
                    node.state *= transition_matrix[child.name].mm(child.state)
                scaler = torch.sum(node.state, 0)
                node.state /= scaler
                scaler_list.append(scaler)

        scaler_list.append(torch.mm(self.pden.view(-1,4), tree.state).squeeze())
        logll = torch.sum(torch.log(torch.stack(scaler_list)) * self.site_counts)
        return logll
    
    def _loglikelihood(self, log_branch, trees):
        bs = len(trees)
        seqlen = self.L.shape[-1]
        child_parent_pairs = [[[] for b in range(bs)] for i in range(self.ntips-2)]
        for b, tree in enumerate(trees):
            i = 0
            for node in tree.traverse('postorder'):
                if not node.is_leaf():
                    child_parent_pairs[i][b] = [child.name for child in node.children]
                    child_parent_pairs[i][b].append(node.name)
                    i += 1
        child_parent_pairs = [torch.LongTensor(child_parent_pairs[i]).to(self.L.device) for i in range(self.ntips-2)]

        ll = torch.concat([self.L, torch.zeros((self.ntips-2, 4, seqlen), device=self.L.device)], dim=0).unsqueeze(0).repeat(bs, 1, 1, 1)
        branch_D = log_branch.exp().unsqueeze(-1) * self.D
        transition_matrix = ((self.U * torch.exp(branch_D).unsqueeze(2)) @ self.U_inv).clamp(0.0) ## (bs, nbranch, 4, 4)
        
        scalar_list = []

        for child_parent_pair in child_parent_pairs:
            child, parent = child_parent_pair[:,:-1], child_parent_pair[:,-1]
            child_ll = torch.gather(ll, 1, child[:,:,None,None].repeat(1,1,4,seqlen)) ## (bs, 2, 4, seqlen)
            child_transition_matrix = torch.gather(transition_matrix, 1, child[:,:,None,None].repeat(1,1,4,4))
            # child_ll = ll[range(bs), child]
            # child_transition_matrix = transition_matrix[range(bs), child]
            up_ll = torch.prod(torch.matmul(child_transition_matrix, child_ll), dim=1) ## (bs, 2, 4, seqlen) -> (bs, 4, seqlen)
            scaler = torch.sum(up_ll, dim=1)
            scalar_list.append(scaler)
            parent_ll = up_ll / scaler.unsqueeze(1)
            
            addition = torch.zeros_like(ll)
            addition[range(bs), parent] = parent_ll
            ll = ll + addition
            # ll[range(bs), parent] = parent_ll
        
        root_scaler = torch.sum(self.pden[:,None] * parent_ll, dim=1)
        scalar_list.append(root_scaler) 
        logll = torch.sum(torch.log(torch.stack(scalar_list, dim=1)) * self.site_counts, dim=(1,2)) ## (bs, nactions, seqlen)
        return logll
            
    def logp_joint(self, log_branch, tree):
        return self.logprior(log_branch) + self.loglikelihood(log_branch, tree)
    

class Parsimony(nn.Module):
    nuc2vec = {'A':[1,0,0,0,0], 'G':[0,1,0,0,0], 'C':[0,0,1,0,0], 'T':[0,0,0,1,0],
           '-':[0,0,0,0,1], '?':[1,1,1,1,0], 'N':[1,1,1,1,0]}
    
    def __init__(self, data, taxa):
        super().__init__()
        self.ntips = len(data)
        self.nsites = len(data[0])
        self.taxa = taxa

        L, site_counts = map(torch.LongTensor, self.initialCLV(data))
        
        self.register_buffer('L', L)
        self.register_buffer('site_counts', site_counts)

    def initialCLV(self, data):
        data_arr = np.array(list(zip(*data)))
        unique_sites, counts = np.unique(data_arr, return_counts=True, axis=0)
        unique_data = unique_sites.T
        
        return [np.transpose([self.nuc2vec[c] for c in unique_data[i]]) for i in range(self.ntips)], counts

    def score(self, tree: TreeNode):
        # init by mapping terminal clades and states in column_i
        score = torch.zeros(size=(self.L.shape[2],), dtype=torch.long, device=self.L.device)
        for node in tree.traverse('postorder'):
            if node.is_leaf():
                node.state = self.L[node.name]
            else:
                left_state = node.children[0].state
                right_state = node.children[1].state
                intersection = left_state * right_state
                union = ((left_state + right_state) > 0.5).long()
                intersection_empty = (intersection.sum(0, keepdim=True) < 0.5)
                next_state = intersection_empty.long() * union + (1-intersection_empty.long()) * intersection
                node.state = next_state
                score[intersection_empty.squeeze()] += 1

        left_state = tree.state
        right_state = tree.children[2].state
        intersection = left_state * right_state
        intersection_empty = (intersection.sum(0, keepdim=True) < 0.5)
        score[intersection_empty.squeeze()] += 1
        return torch.sum(score * self.site_counts)
    
    def _score(self, trees):
        bs = len(trees)
        seqlen = self.L.shape[-1]
        child_parent_pairs = [[[] for b in range(bs)] for i in range(self.ntips-1)]  ## add one because we need to treat the root node.

        for b, tree in enumerate(trees):
            i = 0
            for node in tree.traverse('postorder'):
                if node.is_root():
                    c1, c2, c3 = node.children
                    child_parent_pairs[i][b] = [c1.name, c2.name, node.name]
                    i += 1
                    child_parent_pairs[i][b] = [node.name, c3.name, -1]
                elif not node.is_leaf():
                    child_parent_pairs[i][b] = [child.name for child in node.children]
                    child_parent_pairs[i][b].append(node.name)
                    i += 1
                
        child_parent_pairs = [torch.LongTensor(child_parent_pairs[i]).to(self.L.device) for i in range(self.ntips-1)]

        state = torch.concat([self.L, torch.zeros((self.ntips-1, 5, seqlen),dtype=torch.long, device=self.L.device)], dim=0).unsqueeze(0).repeat(bs, 1, 1, 1)

        score = torch.zeros(size=(bs, self.L.shape[2]), dtype=torch.long, device=self.L.device)

        for child_parent_pair in child_parent_pairs:
            child, parent = child_parent_pair[:,:-1], child_parent_pair[:,-1]
            child_state = torch.gather(state, 1, child[:,:,None,None].repeat(1,1,5,seqlen)) ## (bs, 2, 5, seqlen)
            intersection = torch.prod(child_state, dim=1)
            union = (torch.sum(child_state, dim=1) > 0.5).long()
            intersection_empty = (intersection.sum(1, keepdim=True) < 0.5)  ## (bs, 1, seqlen)
            next_state = intersection_empty.long() * union + (1-intersection_empty.long()) * intersection
            state[range(bs), parent] = next_state
            score[intersection_empty.squeeze(1)] += 1
        return torch.sum(score * self.site_counts, dim=-1)