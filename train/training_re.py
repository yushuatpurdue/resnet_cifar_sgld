
'''
The files are created by Yushu Huang for the EMC-SGLD algorithm. YEAR 2021. 
They aim to apply the algorithm to the training of resnet under CIFAR dataset. 

In this file: function for one-step replica exchange training. 
use 'from train.training_re import training_re' to import function in this file. 
'''

import math
import copy
import sys
import os
import timeit
import csv
import argparse
from tqdm import tqdm ## better progressbar
from math import exp
from sys import getsizeof
import numpy as np
import random
import pickle
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
import torch.optim as optim

## import local

from BayesEval import BayesEval
from optimizers.SGLD import SGLD
from optimizers.SGHMC import SGHMC


def training_re(pars, model, train_dataset, criterion, mutation_method):
    '''
    Input: 
        pars            defined in main() by argparser. 
        model           a ModuleList of neural networks to train. 
        train_dataset    
        criterion       loss function. should be nn.CrossEntropyLoss() to obtain the potential. 
    '''

    if pars.nchains != 2:
        raise ValueError("cannot apply replica exchange: # chains != 2. ")

    ''' one step optimization. '''
    
        
        # print("nnet = ", nnet) # 要注释掉
    swap = 0

    for i, (images, labels) in enumerate(train_dataset):

        # first change the data type from 'tensor' to 'Variables'. 
        images = Variable(images).cuda() if torch.cuda.is_available() else Variable(images)
        labels = Variable(labels).cuda() if torch.cuda.is_available() else Variable(labels)
        Loss = []        # record loss for all chains. 

        for idx in range(pars.nchains):
            
            net = model[idx]
            ''' training net '''
            net.train()      # first set net to train mode. 
            # print("net", nnet, "parameter head", list(net.parameters())[0][0])
            eta         = pars.eta / (pars.etagap ** idx)  # different chains occupy different eta
            temperature = pars.T / (pars.Tgap ** idx)    # different chains occupy different temperature
            
            # take a net from the ModuleList: multichain_model, and put it into the optimizer. 
            # optimizer = optim.SGD(net.parameters(), lr=eta, momentum=0.9)        # no ladder here. need to modify later. 
            if mutation_method == 'SGHMC':
                optimizer = SGHMC(net.parameters(), eta=eta, temperature=temperature, wdecay=pars.wdecay)
            elif mutation_method == 'SGLD':
                optimizer = SGLD(net.parameters(), lr=eta, temperature=temperature)
            # optimizer = SGHMC(net.parameters(), eta=eta, temperature=temperature, wdecay=pars.wdecay)

            # then evaluate optimizer, calculate loss and back propagate. 
            optimizer.zero_grad()
            output = net(images)
            loss   = criterion(output, labels) * pars.total
            Loss   +=  [loss]
            
            # print(loss.item())       # 要注释掉
            # print("data = ", i)       # 要注释掉

            loss.backward()
            optimizer.step()
            ''' print results '''
            if i == 1:
                print('Chain {} Loss: {:0.2f} T: {:.2E} eta: {:.2E}'\
                            .format(idx, \
                            loss.item(), temperature, eta))

        ''' swap '''
        temperature1 = pars.T
        temperature2 = pars.T/pars.Tgap
        diff_inv_temperature = 1/temperature1 - 1/temperature2         # 1/t1 - 1/t2
        diff_loglikelihood = Loss[0].item() - Loss[1].item()                           # -logp1 - (-logp2)
        log_swap_rate = diff_inv_temperature*(diff_loglikelihood-diff_inv_temperature/pars.bias_F)

        if np.log(np.random.uniform(0, 1)) < log_swap_rate:
            swap += True
            net_temp = copy.deepcopy(model[0])
            model[0] = copy.deepcopy(model[1])
            model[1] = copy.deepcopy(net_temp)
        else:
            swap += False



    output = dict()
    output["Loss"] = Loss
    output["swap"] = swap
    return output
        
        
        
        


