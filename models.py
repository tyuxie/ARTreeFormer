import torch
import torch.nn as nn
import time
import math
import tqdm
import numpy as np
from ete3 import TreeNode
from Bio.Phylo.TreeConstruction import *
import psutil
import os
from phyloModel import PHY, Parsimony
from gnn_Model import GNN_BranchModel
from transformer import Transformer
import gc
from collections import defaultdict
from torch.utils.data import DataLoader
from ema_pytorch import EMA
from copy import deepcopy
    
class VBPIbase(nn.Module):
    EPS = np.finfo(float).eps
    def __init__(self, ntips, emp_tree_freq=None, cfg=None):
        super().__init__()
        self.emp_tree_freq = emp_tree_freq
        self.ntips = ntips
        self.register_buffer('log_p_tau', torch.FloatTensor([float(-np.sum(np.log(np.arange(3, 2*self.ntips-3, 2))))]))
        self.tree_model = Transformer(self.ntips, cfg.transformer)

    def squaretrick(self, edge_index: torch.LongTensor, tol=1e-5, max_iters=10000):
        bs, current_ntips = edge_index.shape[0], (edge_index.shape[1] + 2) // 2
        dim, nf = current_ntips - 2, current_ntips
        assert dim + nf == edge_index.shape[1]

        X = torch.concat([torch.eye(nf, device=edge_index.device), torch.ones(dim, nf, device=edge_index.device) / nf], dim=0).unsqueeze(0).repeat(bs,1,1)
        A = torch.zeros((bs, dim+nf, dim+nf), device=edge_index.device)
        A[:,nf:].scatter_(dim=-1, index=edge_index[:, nf:], value=1/3)
        A[:, :nf, :nf] = torch.eye(nf, device=edge_index.device).unsqueeze(0).repeat(bs, 1, 1)

        for it in range(1, max_iters + 1):
            X_old = X.clone()
            A = torch.matmul(A, A)
            X = A @ X_old

            Lnorm = torch.mean(torch.abs(X-X_old), dim=(1,2))
            if not torch.any(Lnorm > tol):
                break
        if not current_ntips == self.ntips:
            X = torch.concat([X, torch.zeros((bs, dim+nf, self.ntips-nf), device=X.device)], dim=-1)
        return X

    def init_tree(self):
        name_dict = {}
        tree = TreeNode(name=3)
        name_dict[3] = tree
        for i in [0,1,2]:
            node = TreeNode(name=i)
            tree.add_child(node)
            name_dict[i] = node
        edge_index = [
            [3,-1,-1],
            [3,-1,-1],
            [3,-1,-1],
            [0,1,2]
        ]
        return tree, name_dict, edge_index


    def _add_node(self, trees, name_dicts, pos, taxon, edge_index):

        for i in range(len(trees)):
            
            tree, name_dict, position = trees[i], name_dicts[i], pos[i]
            new_leaf_node = TreeNode()
            new_pendant_node = TreeNode()
            anchor_node = name_dict[position]
            parent_node = anchor_node.up

            parent_node.remove_child(anchor_node)
            new_pendant_node.add_child(anchor_node)
            new_pendant_node.add_child(new_leaf_node)
            parent_node.add_child(new_pendant_node)
            
            current_taxon_node = name_dict[taxon]
            root_node = tree
            new_leaf_node.name, new_pendant_node.name, current_taxon_node.name, root_node.name = taxon, 2*taxon-2, root_node.name, 2*taxon-1
            name_dict[new_leaf_node.name] = new_leaf_node
            name_dict[new_pendant_node.name] = new_pendant_node
            name_dict[current_taxon_node.name] = current_taxon_node
            name_dict[root_node.name] = root_node
            
            edge_index[i].append([0,0,0])
            edge_index[i].append([0,0,0])
            this_edge_index = edge_index[i]
            this_edge_index = self._update_edge_index(new_leaf_node, this_edge_index)
            this_edge_index = self._update_edge_index(new_pendant_node, this_edge_index)
            this_edge_index = self._update_edge_index(anchor_node, this_edge_index)
            this_edge_index = self._update_edge_index(parent_node, this_edge_index)
            this_edge_index = self._update_edge_index(root_node, this_edge_index)
            for node in root_node.get_children():
                this_edge_index = self._update_edge_index(node, this_edge_index)
            this_edge_index = self._update_edge_index(current_taxon_node, this_edge_index)
            if not current_taxon_node.is_root():
                this_edge_index = self._update_edge_index(current_taxon_node.up, this_edge_index)
            for node in current_taxon_node.get_children():
                this_edge_index = self._update_edge_index(node, this_edge_index)
            edge_index[i] = this_edge_index
        return trees, name_dicts, edge_index
    
    def _update_edge_index(self, node, edge_index):
        if node.is_root():
            p1, p2, p3 = [child.name for child in node.get_children()]
        elif node.is_leaf():
            p1, p2, p3 = node.up.name, -1, -1
        else:
            p1 = node.up.name
            p2, p3 = [child.name for child in node.get_children()]
        edge_index[node.name] = [p1,p2,p3]
        return edge_index

    def sample_trees(self, n_particles, eps=0.0):
        logq_tree = 0.0
        trees, name_dicts, edge_index = zip(*[self.init_tree() for _ in range(n_particles)])
        trees, name_dicts, edge_index = list(trees), list(name_dicts), list(edge_index)


        for taxon in range(3, self.ntips):
            edge_index_tensor = torch.LongTensor(edge_index).to(self.log_p_tau.device)
            node_features = self.squaretrick(edge_index_tensor)
            logits = self.tree_model.forward(node_features, edge_index_tensor, taxon)
            log_prob = torch.log(logits.exp() * (1-eps) + torch.ones_like(logits) / logits.shape[0] * eps)
            pos = torch.multinomial(input=log_prob.exp(), num_samples=1).squeeze(-1)
            logq_tree += torch.gather(log_prob, dim=1, index=pos.unsqueeze(-1)).squeeze(-1)
            trees, name_dicts, edge_index = self._add_node(trees, name_dicts, pos.tolist(), taxon, edge_index)

        return list(trees), logq_tree, torch.LongTensor(edge_index).to(self.log_p_tau.device)
    

    def tree_prob(self, decisions):
        n_particles = decisions.shape[0]
        trees, name_dicts, edge_index = zip(*[self.init_tree() for _ in range(n_particles)])
        trees, name_dicts, edge_index = list(trees), list(name_dicts), list(edge_index)
        logq_tree = 0.0

        for taxon in range(3, self.ntips):
            edge_index_tensor = torch.LongTensor(edge_index).to(self.log_p_tau.device)
            node_features = self.squaretrick(edge_index_tensor)
            logits = self.tree_model.forward(node_features, edge_index_tensor, taxon)
            pos = decisions[:, taxon-3]
            logq_tree += torch.gather(logits, dim=1, index=pos[:,None]).squeeze(-1)
            trees, name_dicts, edge_index = self._add_node(trees, name_dicts, pos.tolist(), taxon, edge_index)
        return logq_tree



    @torch.no_grad()
    def kl_div(self):
        kl_div = 0.0
        probs = []
        negDataEnt = np.sum(self.emp_tree_freq.dataset.wts * np.log(np.maximum(self.emp_tree_freq.dataset.wts, self.EPS)))
        for i, decisions in enumerate(iter(self.emp_tree_freq)):
            tree_prob = self.tree_prob(decisions.to(self.log_p_tau.device)).exp().tolist()
            if isinstance(tree_prob, list):
                probs.extend(tree_prob)
            elif isinstance(tree_prob, float):
                probs.append(tree_prob)
            else:
                raise TypeError
        kl_div = negDataEnt - np.sum(self.emp_tree_freq.dataset.wts * np.log(np.maximum(probs, self.EPS)))
        return kl_div, probs



class TDE(VBPIbase):
    def __init__(self, dataloader, ntips, emp_tree_freq, model_cfg):
        super(TDE, self).__init__(ntips=ntips, emp_tree_freq=emp_tree_freq, cfg=model_cfg)
        self.dataloader = dataloader
    
    def nll(self, decisions, wts):
        '''Calculate the negative log-likelihood.'''
        loss = torch.sum(self.tree_prob(decisions) * wts)
        return -loss
    
    def emp_ll(self):
        probs = []
        dataloader = DataLoader(self.dataloader.dataset, batch_size=500, shuffle=False)
        for i, decisions in enumerate(iter(dataloader)):
            tree_prob = self.tree_prob(decisions.to(self.log_p_tau.device)).exp().tolist()
            if isinstance(tree_prob, list):
                probs.extend(tree_prob)
            elif isinstance(tree_prob, float):
                probs.append(tree_prob)
            else:
                raise TypeError
        emp_ll = np.sum(self.dataloader.dataset.wts * np.log(np.maximum(probs, self.EPS)))
        return emp_ll

    def learn(self, cfg, logger=None):
        if cfg.optimizer.type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=cfg.optimizer.stepsz)
        elif cfg.optimizer.type == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=cfg.optimizer.stepsz)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=list(range(cfg.optimizer.anneal_freq, cfg.optimizer.maxiter+cfg.optimizer.anneal_freq, cfg.optimizer.anneal_freq)), gamma=cfg.optimizer.anneal_rate)
        run_time = -time.time()
        nlls_track, kldivs, nlls, gradnorms, emakldivs, ells, emaells = [], [], [], [], [], [], []
        self.ema = EMA(self.tree_model, beta=cfg.optimizer.ema_beta, update_every=cfg.optimizer.ema_update_every, update_after_step=cfg.optimizer.ema_update_after_step)

        wts = torch.ones(size=(cfg.optimizer.batch_size,)) / cfg.optimizer.batch_size
        self.tree_model.train()
        iterator = iter(self.dataloader)
        for it in tqdm.trange(1, cfg.optimizer.maxiter+1):
            decisions = next(iterator)
            loss = self.nll(decisions.to(self.log_p_tau.device), wts.to(self.log_p_tau.device))
            nlls.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            gradnorm = torch.nn.utils.clip_grad.clip_grad_norm_(self.parameters(), max_norm=cfg.optimizer.clip_value if cfg.optimizer.clip_grad else float('inf'), error_if_nonfinite=True)
            gradnorms.append(gradnorm.item())
            optimizer.step()
            self.ema.update()
            scheduler.step()

            if it % cfg.optimizer.test_freq == 0:
                run_time += time.time()
                logger.info('{} Iter {}:({:.1f}s) NLL Loss {:.4f} | GradNorm: Mean: {:.4f} Max: {:.4f} | Memory: {:.4f} MB'.format(time.asctime(time.localtime(time.time())), it, run_time, np.mean(nlls), np.mean(gradnorms), np.max(gradnorms), psutil.Process(os.getpid()).memory_info().rss/1024/1024))               
                nlls_track.append(np.mean(nlls))
                run_time = -time.time()
                nlls, gradnorms = [], []             
            if self.emp_tree_freq and it % cfg.optimizer.kl_freq == 0:
                self.tree_model.eval()
                kldiv, pred_prob = self.kl_div()
                ell = self.emp_ll()
                ells.append(ell)
                kldivs.append(kldiv)
                current = deepcopy(self.tree_model.state_dict())
                with torch.no_grad():
                    self.tree_model.load_state_dict(self.ema.ema_model.state_dict())
                kldiv, pred_prob = self.kl_div()
                ell = self.emp_ll()
                emaells.append(ell)
                emakldivs.append(kldiv)
                with torch.no_grad():
                    self.tree_model.load_state_dict(current)
                self.tree_model.train()
                del current
                run_time += time.time()
                logger.info('>>> Iter {}:({:.1f}s) KL {:.4f} LL {:.4f} | EMA KL {:.04f} LL {:.4f}'.format(it, run_time, kldivs[-1], ells[-1], emakldivs[-1], emaells[-1]))
                run_time = -time.time()

            if it % cfg.optimizer.save_freq == 0:
                state = {'model': self.tree_model.state_dict(), 'ema': self.ema.ema_model.state_dict()}
                torch.save(state, cfg.base.save_to_path.replace('final', str(it)))

        if cfg.base.save_to_path is not None:
            state = {'model': self.tree_model.state_dict(), 'ema': self.ema.ema_model.state_dict()}
            torch.save(state, cfg.base.save_to_path)


        if self.emp_tree_freq:
            return nlls_track, kldivs, emakldivs, ells, emaells
        else:
            return nlls_track
    
class VBPI(VBPIbase):
    def __init__(self, taxa, data, pden, subModel, emp_tree_freq=None,
                 scale=0.1, cfg=None):
        super(VBPI, self).__init__(ntips=len(data), emp_tree_freq=emp_tree_freq, cfg=cfg.tree)
        self.phylo_model = PHY(data, taxa, pden, subModel, scale=scale)  ## the unnormalized posterior density.
        self.branch_model = GNN_BranchModel(self.ntips, cfg.branch.gnn)
        self.scale = scale
        self.taxa = taxa
        self.step = 0

    @torch.no_grad()
    def lower_bound(self, n_particles=1, n_runs=1000):
        lower_bounds = []
        for run in range(n_runs):
            samp_trees, logq_tree, edge_index = self.sample_trees(n_particles)
            samp_log_branch, logq_branch = self.branch_model(edge_index)
            logll = torch.stack([self.phylo_model.loglikelihood(log_branch, tree) for log_branch, tree in zip(*[samp_log_branch, samp_trees])])
            logp_prior = self.phylo_model.logprior(samp_log_branch)   
            lower_bounds.append(torch.logsumexp(logll + logp_prior - logq_tree - logq_branch + self.log_p_tau - math.log(n_particles), 0))            
        lower_bound = torch.stack(lower_bounds).mean()
            
        return lower_bound.item()

    @torch.no_grad()
    def lower_bound(self, n_particles=1, n_runs=1000):
        lower_bounds = []
        for run in range(n_runs):
            samp_trees, logq_tree, edge_index = self.sample_trees(n_particles)
            samp_log_branch, logq_branch = self.branch_model(edge_index)
            logll = torch.stack([self.phylo_model.loglikelihood(log_branch, tree) for log_branch, tree in zip(*[samp_log_branch, samp_trees])])
            logp_prior = self.phylo_model.logprior(samp_log_branch)   
            lower_bounds.append(torch.logsumexp(logll + logp_prior - logq_tree - logq_branch + self.log_p_tau - math.log(n_particles), 0))            
        lower_bound = torch.stack(lower_bounds).mean()
            
        return lower_bound.item()
    
    @torch.no_grad()
    def batched_elbo(self, n_runs=1000):
        samp_trees, logq_tree, edge_index = self.sample_trees(n_runs, eps=0.0)
        samp_log_branch, logq_branch = self.branch_model(edge_index)
        logll = self.phylo_model._loglikelihood(samp_log_branch, samp_trees)
        logp_prior = self.phylo_model.logprior(samp_log_branch)
        elbos = logll + logp_prior - logq_tree - logq_branch + self.log_p_tau
        return elbos

    def rws_lower_bound(self, inverse_temp=1.0, n_particles=10, eps=0.0):
        samp_trees, logq_tree, edge_index = self.sample_trees(n_particles, eps=eps)
        samp_log_branch, logq_branch = self.branch_model(edge_index)
        logll = torch.stack([self.phylo_model.loglikelihood(log_branch, tree) for log_branch, tree in zip(*[samp_log_branch, samp_trees])])
        logp_prior = self.phylo_model.logprior(samp_log_branch)
        logp_joint = inverse_temp * logll + logp_prior
        lower_bound = torch.logsumexp(logll + logp_prior - logq_tree - logq_branch + self.log_p_tau - math.log(n_particles), 0)
        
        l_signal = logp_joint - logq_tree.detach() - logq_branch
        temp_lower_bound = torch.logsumexp(l_signal - math.log(n_particles), dim=0)
        snis_wts = torch.softmax(l_signal, dim=0)
        rws_fake_term = torch.sum(snis_wts.detach() * logq_tree, dim=0)

        return temp_lower_bound, rws_fake_term, lower_bound, torch.max(logll)

    def vimco_lower_bound(self, inverse_temp=1.0, n_particles=10, eps=0.0):
        samp_trees, logq_tree, edge_index = self.sample_trees(n_particles, eps=eps)
        samp_log_branch, logq_branch = self.branch_model(edge_index)

        logll = self.phylo_model._loglikelihood(samp_log_branch, samp_trees)
        logp_prior = self.phylo_model.logprior(samp_log_branch)
        logp_joint = inverse_temp * logll + logp_prior
        lower_bound = torch.logsumexp(logll + logp_prior - logq_tree - logq_branch + self.log_p_tau - math.log(n_particles), 0)
        
        l_signal = logp_joint - logq_tree - logq_branch
        mean_exclude_signal = (torch.sum(l_signal) - l_signal) / (n_particles-1.)
        control_variates = torch.logsumexp(l_signal.view(-1,1).repeat(1, n_particles) - l_signal.diag() + mean_exclude_signal.diag() - math.log(n_particles), dim=0)
        temp_lower_bound = torch.logsumexp(l_signal - math.log(n_particles), dim=0)
        vimco_fake_term = torch.sum((temp_lower_bound - control_variates).detach() * logq_tree, dim=0)

        return temp_lower_bound, vimco_fake_term, lower_bound, torch.max(logll)

    def learn(self, cfg, logger=None):
        lbs, lls = [], []
        test_kl_div, test_lb = [], []
        grad_norms = []
        optimizer_tree = torch.optim.Adam(params=self.tree_model.parameters(), lr=cfg.optimizer.tree.stepsz)
        optimizer_branch = torch.optim.Adam(params=self.branch_model.parameters(), lr=cfg.optimizer.branch.stepsz)
        scheduler_tree = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_tree, milestones=list(range(cfg.optimizer.tree.anneal_freq_warm, cfg.optimizer.tree.anneal_freq_warm+cfg.optimizer.warm_start_interval, cfg.optimizer.tree.anneal_freq_warm)) + list(range(cfg.optimizer.warm_start_interval+cfg.optimizer.tree.anneal_freq, cfg.optimizer.maxiter+cfg.optimizer.tree.anneal_freq, cfg.optimizer.tree.anneal_freq)), gamma=cfg.optimizer.tree.anneal_rate)
        scheduler_branch = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_branch, milestones=list(range(cfg.optimizer.branch.anneal_freq_warm, cfg.optimizer.branch.anneal_freq_warm+cfg.optimizer.warm_start_interval, cfg.optimizer.branch.anneal_freq_warm)) + list(range(cfg.optimizer.warm_start_interval+cfg.optimizer.branch.anneal_freq, cfg.optimizer.maxiter+cfg.optimizer.branch.anneal_freq, cfg.optimizer.branch.anneal_freq)), gamma=cfg.optimizer.branch.anneal_rate)

        run_time = -time.time()
        self.train()

        for _ in range(self.step):
            scheduler_tree.step()
            scheduler_branch.step()

        for it in tqdm.trange(self.step + 1, cfg.optimizer.maxiter+1):
            overall_time = time.time()
            inverse_temp = min(1., cfg.optimizer.init_inverse_temp + it * 1.0/cfg.optimizer.warm_start_interval)
            eps = cfg.optimizer.eps_max * max(1. - it / cfg.optimizer.eps_period, 0.0) 
            if cfg.objective.loss_type == 'vimco':
                temp_lower_bound, vimco_fake_term, lower_bound, logll  = self.vimco_lower_bound(inverse_temp, cfg.objective.n_particles, eps)
                loss = - temp_lower_bound - vimco_fake_term
            elif cfg.objective.loss_type == 'rws':
                temp_lower_bound, rws_fake_term, lower_bound, logll = self.rws_lower_bound(inverse_temp, cfg.objective.n_particles, eps)
                loss = - temp_lower_bound - rws_fake_term
            else:
                raise NotImplementedError

            lbs.append(lower_bound.item())
            lls.append(logll.item())
            
            optimizer_tree.zero_grad()
            optimizer_branch.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad.clip_grad_norm_(parameters=self.parameters(), max_norm=cfg.optimizer.clip_value if cfg.optimizer.clip_grad else float('inf'), error_if_nonfinite=True)
            grad_norms.append(grad_norm.item())
            optimizer_tree.step()
            scheduler_tree.step()
            optimizer_branch.step()
            scheduler_branch.step()


            gc.collect()
            if it % cfg.optimizer.test_freq == 0:
                run_time += time.time()
                logger.info('{} Iter {}:({:.3f}s) Lower Bound: {:.4f} | Loglikelihood: {:.4f} | GradNorm: Mean: {:.4f} Max: {:.4f} | Memory: {:.04f} MB'.format(time.asctime(time.localtime(time.time())), it, run_time, np.mean(lbs), np.max(lls), np.mean(grad_norms), np.max(grad_norms), psutil.Process(os.getpid()).memory_info().rss/1024/1024))
                if it % cfg.optimizer.lb_test_freq == 0:
                    self.tree_model.eval()
                    run_time = -time.time()
                    test_lb.append(self.lower_bound(n_particles=1))
                    run_time += time.time()
                    if self.emp_tree_freq:
                        kldiv, pred_probs = self.kl_div()
                        test_kl_div.append(kldiv)
                        logger.info('>>> Iter {}:({:.1f}s) Test Lower Bound: {:.4f} Test KL: {:.4f}'.format(it, run_time, test_lb[-1], test_kl_div[-1]))
                    else:
                        logger.info('>>> Iter {}:({:.1f}s) Test Lower Bound: {:.4f}'.format(it, run_time, test_lb[-1]))
                    self.tree_model.train()
                    gc.collect()
                run_time = -time.time()
                lbs, lls = [], []
                grad_norms = []
            if it % cfg.optimizer.save_freq == 0:
                torch.save(self.state_dict(), cfg.base.save_to_path.replace('final', str(it)))
        torch.save(self.state_dict(), cfg.base.save_to_path)

        return test_lb, test_kl_div


class ParsiVBPI(VBPIbase):
    def __init__(self, taxa, data, emp_tree_freq=None, cfg=None):
        super(ParsiVBPI, self).__init__(ntips=len(data), emp_tree_freq=emp_tree_freq, cfg=cfg.tree)
        self.parsi_model = Parsimony(data, taxa)  ## the unnormalized posterior density.
        self.taxa = taxa
        self.step = 0

    @torch.no_grad()
    def lower_bound(self, n_particles=1, n_runs=1000):
        lower_bounds = []
        for run in range(n_runs):
            samp_trees, logq_tree, edge_index = self.sample_trees(n_particles)
            logll = - torch.stack([self.parsi_model.score(tree) for tree in samp_trees])
            lower_bounds.append(torch.logsumexp(logll - logq_tree - math.log(n_particles), 0))            
        lower_bound = torch.stack(lower_bounds).mean()
            
        return lower_bound.item()
    
    @torch.no_grad()
    def batched_elbo(self, n_runs=1000):
        samp_trees, logq_tree, edge_index = self.sample_trees(n_runs, eps=0.0)
        logll = - self.parsi_model._score(samp_trees)
        elbos = logll - logq_tree
        return elbos

    def rws_lower_bound(self, inverse_temp=1.0, n_particles=10, eps=0.0):
        samp_trees, logq_tree, edge_index = self.sample_trees(n_particles, eps=eps)
        logll = - torch.stack([self.parsi_model.score(tree) for tree in samp_trees])
        logp_joint = inverse_temp * logll
        lower_bound = torch.logsumexp(logll - logq_tree - math.log(n_particles), 0)
        
        l_signal = logp_joint - logq_tree.detach()
        temp_lower_bound = torch.logsumexp(l_signal - math.log(n_particles), dim=0)
        snis_wts = torch.softmax(l_signal, dim=0)
        rws_fake_term = torch.sum(snis_wts.detach() * logq_tree, dim=0)

        return temp_lower_bound, rws_fake_term, lower_bound, torch.max(logll)

    def vimco_lower_bound(self, inverse_temp=1.0, n_particles=10, eps=0.0):
        samp_trees, logq_tree, edge_index = self.sample_trees(n_particles, eps=eps)
        
        logll = - self.parsi_model._score(samp_trees)
        logp_joint = inverse_temp * logll
        lower_bound = torch.logsumexp(logll - logq_tree - math.log(n_particles), 0)
        
        l_signal = logp_joint - logq_tree
        mean_exclude_signal = (torch.sum(l_signal) - l_signal) / (n_particles-1.)
        control_variates = torch.logsumexp(l_signal.view(-1,1).repeat(1, n_particles) - l_signal.diag() + mean_exclude_signal.diag() - math.log(n_particles), dim=0)
        temp_lower_bound = torch.logsumexp(l_signal - math.log(n_particles), dim=0)
        vimco_fake_term = torch.sum((temp_lower_bound - control_variates).detach() * logq_tree, dim=0)

        return temp_lower_bound, vimco_fake_term, lower_bound, torch.max(logll)

    def learn(self, cfg, logger=None):
        lbs, lls = [], []
        test_kl_div, test_lb = [], []
        grad_norms = []
        optimizer_tree = torch.optim.Adam(params=self.tree_model.parameters(), lr=cfg.optimizer.tree.stepsz)
        scheduler_tree = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_tree, milestones=list(range(cfg.optimizer.tree.anneal_freq_warm, cfg.optimizer.tree.anneal_freq_warm+cfg.optimizer.warm_start_interval, cfg.optimizer.tree.anneal_freq_warm)) + list(range(cfg.optimizer.warm_start_interval+cfg.optimizer.tree.anneal_freq, cfg.optimizer.maxiter+cfg.optimizer.tree.anneal_freq, cfg.optimizer.tree.anneal_freq)), gamma=cfg.optimizer.tree.anneal_rate)

        run_time = -time.time()
        self.train()

        for _ in range(self.step):
            scheduler_tree.step()

        for it in tqdm.trange(self.step + 1, cfg.optimizer.maxiter+1):
            overall_time = time.time()
            inverse_temp = min(1., cfg.optimizer.init_inverse_temp + it * 1.0/cfg.optimizer.warm_start_interval)
            eps = cfg.optimizer.eps_max * max(1. - it / cfg.optimizer.eps_period, 0.0) 
            if cfg.objective.loss_type == 'vimco':
                temp_lower_bound, vimco_fake_term, lower_bound, logll  = self.vimco_lower_bound(inverse_temp, cfg.objective.n_particles, eps)
                loss = - temp_lower_bound - vimco_fake_term
            elif cfg.objective.loss_type == 'rws':
                temp_lower_bound, rws_fake_term, lower_bound, logll = self.rws_lower_bound(inverse_temp, cfg.objective.n_particles, eps)
                loss = - temp_lower_bound - rws_fake_term
            else:
                raise NotImplementedError

            lbs.append(lower_bound.item())
            lls.append(logll.item())
            
            optimizer_tree.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad.clip_grad_norm_(parameters=self.parameters(), max_norm=cfg.optimizer.clip_value if cfg.optimizer.clip_grad else float('inf'), error_if_nonfinite=True)
            grad_norms.append(grad_norm.item())
            optimizer_tree.step()
            scheduler_tree.step()
            
            gc.collect()
            if it % cfg.optimizer.test_freq == 0:
                run_time += time.time()
                logger.info('{} Iter {}:({:.3f}s) Lower Bound: {:.4f} | Loglikelihood: {:.4f} | GradNorm: Mean: {:.4f} Max: {:.4f} | Memory: {:.04f} MB'.format(time.asctime(time.localtime(time.time())), it, run_time, np.mean(lbs), np.max(lls), np.mean(grad_norms), np.max(grad_norms), psutil.Process(os.getpid()).memory_info().rss/1024/1024))
                if it % cfg.optimizer.lb_test_freq == 0:
                    self.tree_model.eval()
                    run_time = -time.time()
                    test_lb.append(self.lower_bound(n_particles=1))
                    run_time += time.time()
                    if self.emp_tree_freq:
                        kldiv, pred_probs = self.kl_div()
                        test_kl_div.append(kldiv)
                        logger.info('>>> Iter {}:({:.1f}s) Test Lower Bound: {:.4f} Test KL: {:.4f}'.format(it, run_time, test_lb[-1], test_kl_div[-1]))
                    else:
                        logger.info('>>> Iter {}:({:.1f}s) Test Lower Bound: {:.4f}'.format(it, run_time, test_lb[-1]))
                    self.tree_model.train()
                    gc.collect()
                run_time = -time.time()
                lbs, lls = [], []
                grad_norms = []
            if it % cfg.optimizer.save_freq == 0:
                torch.save(self.state_dict(), cfg.base.save_to_path.replace('final', str(it)))
        torch.save(self.state_dict(), cfg.base.save_to_path)
        
        return test_lb, test_kl_div