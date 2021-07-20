#!/usr/bin/python

'''
The files are created by Yushu Huang for the EMC-SGLD algorithm. YEAR 2021. 
They aim to apply the algorithm to the training of resnet under CIFAR dataset. 

In this file: main function for training a bayes neural network
'''

from BayesEval import BayesEval
import math
import copy
import sys
import os
import timeit
import csv
import argparse
from torch.optim import optimizer
from tqdm import tqdm ## better progressbar
from math import exp
from sys import getsizeof
import numpy as np
import random
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

## import local
import resnet            ## all resnet models. 
from LoadData import LoadData
from optimizers.SGLD import SGLD
from train.training_parallel import training_parallel
from train.trainer import trainer

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
    parser.add_argument('-T', default=0.1, type=float, help='Temperature for highest temperature chain')      ## highest temperature of chains. 
    parser.add_argument('-Tgap', default=0.01, type=float, help='Temperature gap between chains')       ## the rest of the chains have temperatures equispaced with this gap. 
    parser.add_argument('-eta', default=2e-6, type=float, help='squared Sampling learning rate')     ## learning_rate^2/mass for highest temperatur chain. see SGHMC paper page 6
    parser.add_argument('-lrgap', default=0.66, type=float, help='Learning rate gap between chains')    ## learning rate of the second high temperature chain = lrgap * learning rate for highest temperatur chain
    parser.add_argument('-period', default=2, type=int, help='estimate adaptive variance every [period] epochs')
    parser.add_argument('-cycle', default=1, type=int, help='Number of cycles')
    parser.add_argument('-burn', default=0.6, type=float, help='burn in iterations for sampling (sn * burn)')
    parser.add_argument('-LRanneal', default=0.984, type=float, help='lr annealing factor')
    parser.add_argument('-Tanneal', default=1.02, type=float, help='temperature annealing factor')
    parser.add_argument('-wdecay', default=25, type=float, help='Sampling weight decay')

    ## dataset properties: 
    parser.add_argument('-data', default='cifar100', help='CIFAR10/ CIFAR100')  ## data to be used (adds an attribute '-data', and names it as 'data'
    parser.add_argument('-batch', default=256, type=int, help='Batch size')   ## batch size for both training and testing
    parser.add_argument('-total', default=50000, type=int, help='Total data points')

    ## neural network properties:
    parser.add_argument('-model', default='resnet20', help='which resnet model to use. In local package "resnet.py." ')

    ## environmental properties: 
    parser.add_argument('-seed', default=random.randint(1, 1e6), type=int, help='Random Seed')

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
    # model.cpu()

    ## define model for each chain, and append them into a list. 
    multichain_model = nn.ModuleList([model])
    multichain_model.extend([model for i in range(1, pars.nchains)])
    

    ''' set data '''
    train_loader, test_loader = LoadData(pars.batch, pars.batch, pars)

    ''' training loop '''

    # below is the experimental area, eventually needs to be wrapped in training.py. 

    # no epoch issue is incorporated here. Use for loop to train by epoches in the training loop section in main. 

    criterion = nn.CrossEntropyLoss()            # loss function for training. CrossEntropyLoss generates the potential. 
    # accuracy = [BayesEval()] * pars.nchains
            

    for t in range(pars.epochs):

        print("epochs = ",t)      # 要注释掉
        training_parallel(pars, multichain_model, train_loader, test_loader, criterion)
        # for t in range(pars.epochs):

        # # training_parallel(pars, multichain_model, train_loader, test_loader, criterion)
        #     # nnet = 0                  # 要注释掉

        #     for net in multichain_model:
        #         trainer(net, train_loader, test_loader, pars)
        
        # nnet = 0

        # for net in multichain_model:

        #     ''' training nets '''
        #     # take a net from the ModuleList: multichain_model, and put it into the optimizer. 
        #     optimizer = SGLD(net.parameters(), lr=pars.lr, temperature=pars.T)        # no ladder here. need to modify later. 
            
        #     # print("nnet = ", nnet) # 要注释掉

        #     for i, (images, labels) in enumerate(train_loader):

        #         # first change the data type from 'tensor' to 'Variables'. 
        #         images = Variable(images).cuda() if torch.cuda.is_available() else Variable(images)
        #         labels = Variable(labels).cuda() if torch.cuda.is_available() else Variable(labels)

        #         # then define optimizer, evaluate, calculate loss and back propagate. 
        #         optimizer.zero_grad()
        #         output = net(images)
        #         loss = criterion(output, labels)
                
        #         # print(loss.item())       # 要注释掉
        #         # print("data = ", i)       # 要注释掉

        #         loss.backward()
        #         optimizer.step()

        #     ''' evaluating performance '''
        #     accuracy[nnet].eval(net, test_loader, cumulate = True)
        #     print(accuracy[nnet].OneEpoch_acc)
        #     nnet += 1               # 要注释掉


main()

