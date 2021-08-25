'''
The files are created by Yushu Huang for the EMC-SGLD algorithm. YEAR 2021. 
They aim to apply the algorithm to the training of resnet under CIFAR dataset. 
algorithm reference: Jun Liu: Monte Carlo Strategies in Scientific Computing, p232. 
In this file: a class -- SGHMC is created as the optimizer for the network. 
use 'from optimizers.SHMC import SGLD' to import this file. 
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
from optimizers.SGHMC import SGHMC

## for optimizer. 
from numpy.lib.function_base import gradient



def Boltzmann(selection_t, modulelist, train_loader):
    '''
    Boltzman calculates the Boltzmann density of a model: exp(-loss(model)/selection_t), where loss is the cross entropy loss. 

    It takes each module in modulelist, calculate it's full loss, then calculate the Boltzmann density. 
    '''
    criterion = nn.CrossEntropyLoss()           # cross entropy loss generates the potential. 
    Boltzmann_pd = [0]*len(modulelist)          # Boltzmann_pd records the Boltzmann probability density for each model. It is a list of length = nchains. 
    loss = 0                                    # loss is a temporary record of the loss. 
    count = 0
    for net in modulelist:
        net.train()
        for i, (images, labels) in enumerate(train_loader):
            ''' calculate the full loss of the net. '''

            # first change the data type from 'tensor' to 'Variables'. 
            images = Variable(images).cuda() if torch.cuda.is_available() else Variable(images)
            labels = Variable(labels).cuda() if torch.cuda.is_available() else Variable(labels)

            # then evaluate optimizer, calculate loss and back propagate. 
            output = net(images)
            temp   = criterion(output, labels) * len(images)
            print("batch loss = ", temp)
            # loss   += temp

        # Boltzmann_pd[count] = np.exp(-loss/selection_t)
        count += 1
    return(Boltzmann_pd)

    

''' main '''
## define an object "parser" to add attributes to it. 
parser = argparse.ArgumentParser()    

## mcmc properties:
parser.add_argument('-nchains', default=1, type=int, help='Total number of chains')     ## how many chains are going to run. 
parser.add_argument('-epochs', default=1000, type=int, help='Sampling Epochs')       ## how many epochs (something like iterations) are going to run. 
parser.add_argument('-T', default=0.1, type=float, help='Temperature for highest temperature chain')      ## highest temperature of chains. 
parser.add_argument('-Tgap', default=0.01, type=float, help='Temperature gap between chains')       ## the rest of the chains have temperatures equispaced with this gap. 
parser.add_argument('-eta', default=2e-6, type=float, help='squared Sampling learning rate')     ## learning_rate^2/mass for highest temperatur chain. see SGHMC paper page 6
parser.add_argument('-lr', default=2e-6, type=float, help='Sampling learning rate')     ## learning rate for highest temperatur chain. 
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




if pars.data == 'cifar100':
    num_classes = 100
elif pars.data == 'cifar10':
    num_classes = 10

model = nn.DataParallel(resnet.__dict__[pars.model](num_classes = num_classes))
# print(model.__dict__)


# model.cpu()
model.cuda()

## define model for each chain, and append them into a list. 
multichain_model = nn.ModuleList([model])
multichain_model.extend([model for i in range(1, 3)])

train_loader, test_loader = LoadData(pars.batch, pars.batch, pars)
        
''' test snooker'''
# Snooker_test = snooker(train_loader)
# boltzmann_pd = Snooker_test.step(multichain_model)
boltzmann_pd = Boltzmann(0.1, multichain_model, train_loader)
print('output:', boltzmann_pd)
print('output data type:', type(boltzmann_pd))

    