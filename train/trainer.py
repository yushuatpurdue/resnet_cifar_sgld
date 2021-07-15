"""
Note that in Bayesian settings, the lr 2e-6 and weight decay 25 are equivalent to lr 0.1 and weight decay 5e-4 in standard setups.
"""

#!/usr/bin/python
import math
import copy
import sys
import os
import time
import csv
import dill
import argparse
import random
import collections
from random import shuffle
import pickle

from tqdm import tqdm ## better progressbar
from math import exp
from sys import getsizeof
import numpy as np

## import pytorch modules
import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.utils.data as data
import torchvision.datasets as datasets

## Import helper functions
from BayesEval import BayesEval
from train.sgmcmc import Sampler

CUDA_EXISTS = torch.cuda.is_available()


def trainer(nets, train_loader, test_loader, pars):
    criterion = nn.CrossEntropyLoss()
    init_T, init_lr = pars.T, pars.lr
    samplers, BMAS, myVars, cooling_time, lr_set = {}, [], [], [], []

    for idx in range(pars.nchains-1, -1, -1):
        print('Chain {} Initial learning rate {:.2e} temperature {:.2e}'.format(idx, init_lr, init_T))
        sampler = Sampler(nets, criterion, lr=init_lr, wdecay=pars.wdecay, T=init_T, total=pars.total)
        lr_set.insert(0, init_lr)
        init_T /= pars.Tgap
        init_lr /= pars.lrgap
        samplers[idx] = sampler

        BMAS.append(BayesEval())
        myVars.append(sys.float_info.max)
    counter, warm_up, adjusted_corrections = 1., 10, 0
    start = time.time()

    """ Initialization for variance reduction """
    last_full_losses, last_VRnets = [0] * pars.nchains, nn.ModuleList()
    for idx in range(pars.nchains):
        last_VRnets.extend([nets])

    for epoch in range(pars.epochs):
        """ update adaptive variance and variance reduction every [period] epochs """
        if pars.period > 0 and epoch % pars.period == 0 and epoch > warm_up:
            cur_full_losses = [0] * pars.nchains
            for idx in range(pars.nchains):
                stage_losses = []
                nets.eval()
                for i, (images, labels) in enumerate(train_loader):
                    images = Variable(images).cuda() if CUDA_EXISTS else Variable(images)
                    labels = Variable(labels).cuda() if CUDA_EXISTS else Variable(labels)
                    nets.zero_grad()
                    avg_loss = criterion(nets(images), labels).item()
                    cur_full_losses[idx] += avg_loss * pars.batch
                    scaled_full_loss = avg_loss * pars.total
                    
                    stage_losses.append(scaled_full_loss)
                std_epoch = np.std(stage_losses, ddof=1)
                myVars[idx] = 0.5 * std_epoch**2 if myVars[idx] == sys.float_info.max else ((1 - pars.alpha) * myVars[idx] + pars.alpha * 0.5 * std_epoch ** 2)
                print('Epoch {} Chain {} loss std {:.2e} variance {:.2e} smooth variance {:.2e}'.format(epoch, idx, std_epoch, 0.5 * std_epoch**2, myVars[idx]))
                last_VRnets[idx] = nets
                last_full_losses[idx] = cur_full_losses[idx]
        
        for idx in range(pars.nchains):
            nets.train()


        if pars.cycle >= 2:
            sub_sn = pars.epochs / pars.cycle
            cur_beta = (epoch % sub_sn) * 1.0 / sub_sn
            for idx in range(pars.nchains):
                samplers[idx].set_eta(lr_set[idx] / 2 * (np.cos(np.pi * cur_beta) + 1))
                if (epoch % sub_sn) * 1.0 / sub_sn == 0:
                    print('Chain {} Cooling down for optimization'.format(idx))
                    samplers[idx].set_T(1e10)
                elif epoch % sub_sn == int(pars.burn * sub_sn):
                    print('Chain {} Heating up for sampling'.format(idx))
                    samplers[idx].set_T(1e-10)

        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images).cuda() if CUDA_EXISTS else Variable(images)
            labels = Variable(labels).cuda() if CUDA_EXISTS else Variable(labels)
            counter += 1.
            loss_chains = []
            for idx in range(pars.nchains):
                loss = samplers[idx].step(images, labels)
                """ variance-reduced negative log posterior """
                
                loss_chains.append(loss)

            
        """ Anneaing """
        if epoch < pars.burn * pars.epochs:
            pars.bias_F *= pars.Tanneal
        for idx in range(pars.chains):
            if pars.cycle == 1:
                if epoch > 0.4 * pars.epochs and pars.LRanneal <= 1.:
                    samplers[idx].eta *= pars.LRanneal
                if epoch < pars.burn * pars.sn:
                    samplers[idx].set_T(pars.Tanneal)
                
            if (pars.cycle == 1 and epoch >= int(pars.burn * pars.epochs)) or (pars.cycle > 1 and cur_beta >= pars.burn):
                BMAS[idx].eval(nets, test_loader, bma=True)
            elif (pars.cycle == 1 and epoch < int(pars.burn * pars.epochs)) or (pars.cycle > 1 and cur_beta < pars.burn):
                BMAS[idx].eval(nets, test_loader, bma=False)
            print('Epoch {} Chain {} Acc: {:0.2f} BMA: {:0.2f} Best Acc: {:0.2f} Best BMA: {:0.2f} T: {:.2E}  Loss: {:0.2f} Corrections: {:0.2f}'\
                    .format(epoch, idx, BMAS[idx].cur_acc, BMAS[idx].bma_acc, BMAS[idx].best_cur_acc, BMAS[idx].best_bma_acc, \
                    samplers[idx].T, np.array(loss_chains[idx]).sum(), abs(adjusted_corrections)))
        print('')
    end = time.time()
    print('Time used {:.2f}s'.format(end - start))
