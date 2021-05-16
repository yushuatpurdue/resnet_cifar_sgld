#!/usr/bin/python

'''
The files are created by Yushu Huang for the EMC-SGLD algorithm. YEAR 2021. 
They aim to apply the algorithm to the training of resnet under CIFAR dataset. 

In this file: main function for training a bayes neural network
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
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.utils.data as data
import torchvision.datasets as datasets

## import local
import resnet            ## all resnet models. 
from LoadData import LoadData
'''
define the main function
'''
def main():
    ''' first define addon arguments for running this file. '''
    ## define an object "parser" to add attributes to it. 
    parser = argparse.ArgumentParser()    

    ## mcmc properties:
    parser.add_argument('-nchains', default=1, type=int, help='Total number of chains')     ## how many chains are going to run. 
    parser.add_argument('-epochs', default=1000, type=int, help='Sampling Epochs')       ## how many epochs (something like iterations) are going to run. 
    parser.add_argument('-T', default=0.05, type=float, help='Temperature for highest temperature chain')      ## highest temperature of chains. 
    parser.add_argument('-Tgap', default=0.01, type=float, help='Temperature gap between chains')       ## the rest of the chains have temperatures equispaced with this gap. 
    parser.add_argument('-lr', default=3e-6, type=float, help='Sampling learning rate')     ## learning rate for highest temperatur chain. 
    parser.add_argument('-lrgap', default=0.66, type=float, help='Learning rate gap between chains')    ## learning rate of the second high temperature chain = lrgap * learning rate for highest temperatur chain

    ## dataset properties: 
    parser.add_argument('-data', default='cifar100', help='CIFAR10/ CIFAR100')  ## data to be used (adds an attribute '-data', and names it as 'data'
    parser.add_argument('-batch', default=256, type=int, help='Batch size')   ## batch size for both training and testing

    ## neural network properties:
    parser.add_argument('-model', default='resnet20', help='which resnet model to use. In local package "resnet.py." ')

    ## add the arguments. 
    pars = parser.parse_args()

    ## print. 
    print(pars)
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)


    ''' set GPU and seed '''

    '''
    try:
        torch.cuda.set_device(pars.gpu)
    except: # in case the device has only one GPU
        torch.cuda.set_device(0) 
    '''
 

    torch.manual_seed(pars.seed)           # set seed for torch
    torch.cuda.manual_seed_all(pars.seed)      # set seed for all GPUs. 
    np.random.seed(pars.seed)              # set seed for numpy. !!It returns the random numbers with exactly the same order.          
    random.seed(pars.seed)                 # set seed in general. 
    torch.backends.cudnn.deterministic=True # make reproducible. 


    ''' set model '''

    if pars.data == 'cifar100':
        num_classes = 100
    elif pars.data == 'cifar10':
        num_classes = 10
    
    ## define model as an instance of resnet. 
    model = nn.DataParallel(resnet.__dict__[pars.model](num_classes = num_classes))
    model.cuda()

    ## define model for each chain, and append them into a list. 
    multichain_model = nn.ModuleList([model])
    multichain_model.extend([model for i in range(1, pars.nchains)])
    

    ''' set data '''
    train_loader, test_loader = LoadData(pars.batch, pars.batch, pars)

    ''' set optimizer '''
    optimizer = SGLD(model.parameters(), lr=0.1, temperature=1)

    ''' set loss '''
    loss = nn.CrossEntropyLoss()

    ''' training loop '''




