import argparse
from copy import deepcopy
from multiprocessing import Pool
import os
import sys
import numpy as np
import dill
from omegaconf import OmegaConf
import datetime
import logging
import torch
import random
import time
import pickle
sys.path.append("..")
from torch.utils.data import DataLoader, WeightedRandomSampler
from datasets import DecisionData
from models import TDE

def main():
    cfg_file = OmegaConf.load('config.yaml')
    cfg_cmd = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg_file, cfg_cmd)

    name = cfg.model.tree_type + '_' + cfg.base.date
    cfg.base.folder = f'{cfg.base.workdir}/{cfg.data.dataset}/repo{cfg.data.repo}/{name}'
    os.makedirs(cfg.base.folder, exist_ok=True if cfg.base.mode == 'test' else False)
    cfg.base.save_to_path = cfg.base.folder + '/final.pt'
    cfg.base.logpath = cfg.base.folder + '/final.log'
    cfg.base.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    filehandler = logging.FileHandler(cfg.base.logpath)
    filehandler.setLevel(logging.INFO)
    logger.addHandler(filehandler)

    logger.info('Training with the following settings:')
    logger.info(OmegaConf.to_yaml(cfg))


    dataset = DecisionData(cfg.data.dataset, cfg.data.repo)
    sampler = WeightedRandomSampler(weights=dataset.wts, num_samples=cfg.optimizer.batch_size*cfg.optimizer.maxiter, replacement=True)
    dataloader = DataLoader(dataset, batch_size=cfg.optimizer.batch_size, sampler=sampler, num_workers=1)

    wts = np.load(os.path.join(dataloader.dataset.path, 'wts.npy'))
    taxa = np.load(os.path.join(dataloader.dataset.path, 'taxa.npy'))
    wts = np.array(wts) / np.sum(wts)

    if cfg.data.empFreq:
        decision_data = DecisionData(cfg.data.dataset, 'emp')
        emp_tree_freq = DataLoader(decision_data, batch_size=500, shuffle=False)
        logger.info('Empirical estimates from MrBayes loaded')
    else:
        emp_tree_freq = None


    model = TDE(dataloader, ntips=len(taxa), emp_tree_freq=emp_tree_freq, model_cfg=cfg.model).to(cfg.base.device)
    logger.info('Optimal NLL: {}'.format(-np.sum(wts * np.log(wts))))
    logger.info('Parameter Info:')
    for param in model.parameters():
        logger.info(param.dtype)
        logger.info(param.size())

    if cfg.base.mode == 'train':
        logger.info('\nThis version assumes learnable embeddings for the first three taxa. \ Use time embedding')
        logger.info('\nTree Generation Model training, results will be saved to: {}\n'.format(cfg.base.save_to_path))
        nlls = model.learn(cfg, logger=logger)

        if cfg.data.empFreq:
            np.save(cfg.base.logpath.replace('final.log', 'nll.npy'), nlls[0])
            np.save(cfg.base.logpath.replace('final.log', 'kldivs.npy'), nlls[1])
            np.save(cfg.base.logpath.replace('final.log', 'emakldivs.npy'), nlls[2])
            np.save(cfg.base.logpath.replace('final.log', 'ells.npy'), nlls[3])
            np.save(cfg.base.logpath.replace('final.log', 'emaells.npy'), nlls[4])
        else:
            np.save(cfg.base.logpath.replace('final.log', 'nll.npy'), nlls[0])
    elif cfg.base.mode == 'test':
        logger.info('Calculating the KL divergence to the ground truth')
        with torch.no_grad():
            model.tree_model.load_state_dict(torch.load(cfg.base.save_to_path)['model'])
            model.eval()
            kl, pred_probs = model.kl_div()
            np.save(cfg.base.logpath.replace('final.log', 'kldiv.npy'), [kl])
            np.save(cfg.base.logpath.replace('final.log', 'pred_probs.npy'), pred_probs)

            model.tree_model.load_state_dict(torch.load(cfg.base.save_to_path)['ema'])
            model.eval()
            kl, pred_probs = model.kl_div()
            np.save(cfg.base.logpath.replace('final.log', 'emakldiv.npy'), [kl])
            np.save(cfg.base.logpath.replace('final.log', 'ema_pred_probs.npy'), pred_probs)


if __name__ == '__main__':
    main()