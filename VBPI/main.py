import argparse
from copy import deepcopy
from multiprocessing import Pool
import numpy as np
import logging
import os
import sys
# from dataManipulation import *
sys.path.append("..")
from utils import mcmc_treeprob,loadData, summary, namenum
from omegaconf import OmegaConf
from datasets import DecisionData
from torch.utils.data import DataLoader
from models import VBPI
import tqdm
import torch
import random

def main():
    cfg_file = OmegaConf.load('config.yaml')
    cfg = OmegaConf.merge(cfg_file, OmegaConf.from_cli())

    torch.manual_seed(cfg.base.seed)
    np.random.seed(cfg.base.seed)
    random.seed(cfg.base.seed)

    ###### Load Data
    unorderdata, unordertaxa = loadData(cfg.data.data_path + cfg.data.dataset + '.fasta', 'fasta')
    taxa = sorted(unordertaxa)
    indexs = [unordertaxa.index(taxa[i]) for i in range(len(taxa))]
    data = [unorderdata[indexs[i]] for i in range(len(indexs))]
    del unorderdata, unordertaxa

    cfg.base.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    name =  cfg.objective.loss_type + '_' + cfg.tree.tree_type + ('_transformer' if cfg.branch.branch_type == 'transformer' else '') + '_hL_' + str(cfg.branch.gnn.num_layers) + '_aggr' + cfg.branch.gnn.aggr
    if cfg.branch.gnn.project:
        name = name + '_proj'
    name = name + '_' + cfg.base.date
    cfg.base.folder = os.path.join(cfg.base.workdir, cfg.data.dataset, name)
    os.makedirs(cfg.base.folder, exist_ok= (not cfg.base.mode == 'train'))

    cfg.base.save_to_path = os.path.join(cfg.base.folder, 'final.pt')
    cfg.base.logpath = os.path.join(cfg.base.folder, 'final.log')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    filehandler = logging.FileHandler(cfg.base.logpath)
    filehandler.setLevel(logging.INFO)
    logger.addHandler(filehandler)

    logger.info('Training with the following settings:')
    logger.info(OmegaConf.to_yaml(cfg))

    if cfg.data.empFreq:
        decision_data = DecisionData(cfg.data.dataset, 'emp')
        emp_tree_freq = DataLoader(decision_data, batch_size=500, shuffle=False)
        # emp_tree_freq = get_empdataloader(cfg.data.dataset,  batch_size=200)
        logger.info('Empirical estimates from MrBayes loaded')
    else:
        emp_tree_freq = None
    
    model = VBPI(taxa, data, pden=np.ones(4)/4., subModel=('JC', 1.0), emp_tree_freq=emp_tree_freq, cfg=cfg).to(cfg.base.device)

    logger.info(f'Running on device: {cfg.base.device}')  
    logger.info('Parameter Info:')
    for param in model.parameters():
        logger.info(param.dtype)
        logger.info(param.size())

    if cfg.base.mode == 'train':
        logger.info('\nVBPI running, results will be saved to: {}\n'.format(cfg.base.save_to_path))
        test_lb, test_kl_div = model.learn(cfg=cfg, logger=logger)
                    
        np.save(cfg.base.save_to_path.replace('.pt', '_test_lb.npy'), test_lb)
        if cfg.data.empFreq:
            np.save(cfg.base.save_to_path.replace('.pt', '_kl_div.npy'), test_kl_div)
    elif cfg.base.mode == 'resume':
        cfg.base.resume_path = cfg.base.save_to_path.replace('final.pt', f'{cfg.base.resume_from_iter}.pt')
        logger.info('\nVBPI running, results will be saved to: {}\n'.format(cfg.base.resume_path))
        model.load_state_dict(torch.load(cfg.base.resume_path))
        model.step = cfg.base.resume_from_iter
        test_lb, test_kl_div = model.learn(cfg=cfg, logger=logger)
    elif cfg.base.mode == 'test':
        logger.info('\nVBPI testing')
        model.load_state_dict(torch.load(cfg.base.save_to_path, map_location=cfg.base.device))
        logger.info('\nVBPI running, results will be saved to: {}\n'.format(cfg.base.save_to_path))
        if cfg.data.empFreq:
            kl, pred_probs = model.kl_div()
            np.save(cfg.base.save_to_path.replace('final.pt', 'last_kl_div.npy'), [kl])
            np.save(cfg.base.save_to_path.replace('final.pt', 'last_pred_probs.npy'), pred_probs)
            logger.info(f'The KL divergence is {kl:.4f}')
        else:
            ELBOs = []
            for i in tqdm.trange(1, 200+1):
                ELBOs.append(model.batched_elbo(500))
                logger.info(f'{i}-th batch evaluated')
            ELBOs = torch.concat(ELBOs).cpu()
            torch.save(ELBOs, cfg.base.save_to_path.replace('final.pt', f'final_elbos_{np.prod(ELBOs.shape) // 1000}Kitems.pt'))
            lb1000 = torch.logsumexp(ELBOs.reshape(-1, 1000) - np.log(1000), dim=-1)
            logger.info(f'LB1000 {lb1000.mean().item():.3f}({lb1000.std().item():.3f})')
            lb10 = torch.logsumexp(ELBOs.reshape(-1, 100, 10) - np.log(10), dim=-1).mean(-1)
            logger.info(f'LB10 {lb10.mean().item():.3f}({lb10.std().item():.3f})')
            lb1 = ELBOs.reshape(-1, 1000).mean(-1)
            logger.info(f'LB1 {lb1.mean().item():.3f}({lb1.std().item():.3f})')

if __name__ == '__main__':
    main()
    sys.exit()