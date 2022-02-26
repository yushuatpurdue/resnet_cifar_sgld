'''
The files are created by Yushu Huang for the EMC-SGLD algorithm. YEAR 2021. 
They aim to apply the algorithm to the training of resnet under CIFAR dataset. 

In this file: function for one-step EMC-SGLD training. 
use 'from train.training_emc import training_emc' to import this file. 
'''

import math
import copy
import sys
import os
import timeit
import csv
import argparse
from numpy.core.numeric import NaN
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
import resnet            ## all resnet models. 
from LoadData import LoadData
from optimizers.SGHMC import SGHMC
from optimizers.snooker import snooker
from train.training_parallel import training_parallel

def training_emc(pars, modulelist, train_dataset, crossoverRate, crossoverGap, criterion):
    '''
    one epoch emc optimizing
    Input: 
        pars            defined in main() by argparser. 
        modulelist      a ModuleList of neural networks to train. 
        train_dataset      
        criterion       loss function. should be nn.CrossEntropyLoss() to obtain the potential. 
    '''

    ''' snooker'''
    for model in modulelist:
        model.train()

    Boltzmann_pd = [0]*len(modulelist)                # unnormalized Boltzmann pd for sampling anchor chain

    # crossoverRate = copy.deepcopy(pars.crossoverRate)
    for i, (images, labels) in enumerate(train_dataset):

        if i % crossoverGap != 0:
            pars.crossoverRate = 0
        else:
            pars.crossoverRate = crossoverRate

        # first change the data type from 'tensor' to 'Variables'. 
        images = Variable(images).cuda() if torch.cuda.is_available() else Variable(images)
        labels = Variable(labels).cuda() if torch.cuda.is_available() else Variable(labels)
        Loss         = []                                 # record loss for all model. 
        for j in range(len(modulelist)):
            model = modulelist[j]
            # then evaluate optimizer, calculate loss and back propagate. 
            output = model(images)
            temp   = criterion(output, labels) * pars.total
            # if temp == NaN:
            #     print("model ",j, "loss = NaN")
            # Loss   += [temp]                                               # minibatch loss for every model. 
            if pars.crossoverRate != 0:
                Boltzmann_pd[j] = np.exp(-temp.item()/pars.selectionT)         # unnormalized Boltzmann pd for every model. 
            
        
        Snooker_test = snooker(train_dataset, criterion = nn.CrossEntropyLoss(), selection_t = pars.selectionT)   # selection_t = 100000 is a good choice for next to burn-in period. 
        crossover_idx = Snooker_test.step(pars, modulelist, Boltzmann_pd, train_dataset)

        ''' index of chains to go through SGHMC '''
        # SGHMC_idx = list(range(pars.nchains))
        # for i in crossover_idx:
        #     SGHMC_idx.remove(i)


        ''' the other part go through SGHMC '''
        # for idx in SGHMC_idx:
        #     eta         = pars.eta / (pars.etagap ** idx)  # different chains occupy different eta
        #     temperature = pars.T / (pars.Tgap ** idx)    # different chains occupy different temperature
        #     model = modulelist[idx]
        #     loss = Loss[idx]
        
        # # take a net from the ModuleList: multichain_model, and put it into the optimizer. 
        # # optimizer = optim.SGD(net.parameters(), lr=eta, momentum=0.9)        # no ladder here. need to modify later. 

        #     optimizer = SGHMC(model.parameters(), eta=eta, temperature=temperature, wdecay=pars.wdecay)
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
    




