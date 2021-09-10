
'''
The files are created by Yushu Huang for the EMC-SGLD algorithm. YEAR 2021. 
They aim to apply the algorithm to the training of resnet under CIFAR dataset. 

In this file: function for one-step SGLD training. 
use 'from train.training_parallel import training_parallel' to import this file. 
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


def training_parallel(pars, model, train_dataset, test_dataset, criterion):
    '''
    Input: 
        pars            defined in main() by argparser. 
        model           a ModuleList of neural networks to train. 
        train_dataset   
        test_dataset    
        criterion       loss function. should be nn.CrossEntropyLoss() to obtain the potential. 
    '''

    nnet        = 0         # counter of nets. 
    eta         = pars.eta  # lr^2/M. lr = learning rate, M = mass. see paper Chen et al SGHMC. 
    temperature = pars.T
    accuracy    = [BayesEval()] * pars.nchains    # a list recording the accuracy of model. 

    for net in model:

        ''' training net '''
        net.train()      # first set net to train mode. 
        # print("net", nnet, "parameter head", list(net.parameters())[0][0])

        # take a net from the ModuleList: multichain_model, and put it into the optimizer. 
        # optimizer = optim.SGD(net.parameters(), lr=pars.lr, momentum=0.9)        # no ladder here. need to modify later. 
        optimizer = SGHMC(net.parameters(), eta=eta, temperature=temperature, wdecay=pars.wdecay)
        
        # print("nnet = ", nnet) # 要注释掉

        for i, (images, labels) in enumerate(train_dataset):

            # first change the data type from 'tensor' to 'Variables'. 
            images = Variable(images).cuda() if torch.cuda.is_available() else Variable(images)
            labels = Variable(labels).cuda() if torch.cuda.is_available() else Variable(labels)

            # then evaluate optimizer, calculate loss and back propagate. 
            optimizer.zero_grad()
            output = net(images)
            loss   = criterion(output, labels) * pars.total
            
            # print(loss.item())       # 要注释掉
            # print("data = ", i)       # 要注释掉

            loss.backward()
            optimizer.step()

        ''' evaluating performance '''
        accuracy[nnet].eval(net, test_dataset, cumulate = True)

        ''' print results '''
        print('Chain {} Acc: {:0.2f} Loss: {:0.2f} T: {:.2E} eta: {:.2E}'\
                    .format(nnet, accuracy[nnet].OneEpoch_acc, \
                    loss.item(), temperature, eta))
        

        '''prepare for next net'''
        nnet        += 1            # counting # of nets. 
        eta         /= pars.etagap  # different chains occupy different eta
        temperature /= pars.Tgap    # temperature ladder between chains
        


