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

def traing_emc(pars, modulelist, train_dataset, criterion):
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
    loss         = []                                 # record loss for every model. 

    for i, (images, labels) in enumerate(train_dataset):

        # first change the data type from 'tensor' to 'Variables'. 
        images = Variable(images).cuda() if torch.cuda.is_available() else Variable(images)
        labels = Variable(labels).cuda() if torch.cuda.is_available() else Variable(labels)
        for j in range(len(modulelist)):
            model = modulelist[j]
            # then evaluate optimizer, calculate loss and back propagate. 
            output = model(images)
            temp   = criterion(output, labels) * pars.total
            loss   += [temp]                                               # minibatch loss for every model. 
            Boltzmann_pd[j] = np.exp(-temp.item()/pars.selectionT)         # nnormalized Boltzmann pd for every model. 
        
        Snooker_test = snooker(train_dataset, criterion = nn.CrossEntropyLoss(), selection_t = pars.selectionT)   # selection_t = 100000 is a good choice for next to burn-in period. 
        crossover_idx = Snooker_test.step(pars, modulelist, Boltzmann_pd, train_dataset)

        ''' index of chains to go through SGHMC '''
        SGHMC_idx = list(range(pars.nchains))
        for i in crossover_idx:
            SGHMC_idx.remove(i)


        ''' SGHMC '''
        for idx in SGHMC_idx:
            eta         = pars.eta / (pars.etagap ** idx)  # different chains occupy different eta
            temperature = pars.T / (pars.Tgap ** idx)    # different chains occupy different temperature
            model = multichain_model[idx]
        
        # take a net from the ModuleList: multichain_model, and put it into the optimizer. 
        # optimizer = optim.SGD(net.parameters(), lr=eta, momentum=0.9)        # no ladder here. need to modify later. 

            optimizer = SGHMC(model.parameters(), eta=eta, temperature=temperature, wdecay=pars.wdecay)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    




''' run '''
parser = argparse.ArgumentParser()    

## mcmc properties:
parser.add_argument('-nchains', default=5, type=int, help='Total number of chains')     ## how many chains are going to run. 
parser.add_argument('-epochs', default=1000, type=int, help='Sampling Epochs')       ## how many epochs (something like iterations) are going to run. 
parser.add_argument('-selectionT', default=0.1, type=float, help='Selection temperature for snooker crossover')      ## used in calculating Boltzmann distribution in snooker crossover. 
parser.add_argument('-crossoverRate', default=0.3, type=float, help='Rate of the chains for doing crossover')      ## how many chains in the population are going to go through snooker crossover. 
parser.add_argument('-T', default=0.005, type=float, help='Temperature for highest temperature chain')      ## lowest temperature of chains. 
parser.add_argument('-Tgap', default=0.5, type=float, help='Temperature gap between chains')       ## the next chain has temperatures = T/Tgap. 
parser.add_argument('-eta', default=2e-6, type=float, help='squared Sampling learning rate')     ## learning_rate^2/mass for highest temperatur chain. see SGHMC paper page 6
parser.add_argument('-etagap', default=0.66, type=float, help='gap between different chains etas. ')    ## eta of the second lowest temperature chain = eta/etagap for lowest temperature chain
parser.add_argument('-period', default=2, type=int, help='estimate adaptive variance every [period] epochs')
parser.add_argument('-cycle', default=1, type=int, help='Number of cycles')
parser.add_argument('-burn', default=0.6, type=float, help='burn in iterations for sampling (sn * burn)')
parser.add_argument('-etaanneal', default=0.984, type=float, help='lr annealing factor')
parser.add_argument('-Tanneal', default=1.02, type=float, help='temperature annealing factor')
parser.add_argument('-wdecay', default=25, type=float, help='Sampling weight decay')
parser.add_argument('-iterationR', default=30, type=int, help='number of iteration for SGLD in snooker scaler r')      ## how many iterations are going to run for SGLD sampling r in snooker. 
parser.add_argument('-lrR', default=5e-5, type=float, help='learning rate of SGLD in snooker scaler r')      ## max learning rate for SGLD sampling r in snooker. If r is close to 0 then choose a smaller learning rate.  

## dataset properties: 
parser.add_argument('-data', default='cifar100', help='CIFAR10/ CIFAR100')  ## data to be used (adds an attribute '-data', and names it as 'data'
parser.add_argument('-batch', default=256, type=int, help='Batch size')   ## batch size for both training and testing
parser.add_argument('-total', default=50000, type=int, help='Total data points')

## neural network properties:
parser.add_argument('-model', default='resnet20', help='which resnet model to use. In local package "resnet.py." ')

## environmental properties: 
# parser.add_argument('-seed', default=random.randint(1, 1e6), type=int, help='Random Seed')
parser.add_argument('-seed', default=42, type=int, help='Random Seed')

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
for i in range(pars.nchains-1):
    model_add = copy.deepcopy(model)
    multichain_model.extend([model_add])


''' set data '''
train_loader, test_loader = LoadData(pars.batch, pars.batch, pars)

''' training loop '''
criterion = nn.CrossEntropyLoss()            # loss function for training. CrossEntropyLoss generates the potential. 
# accuracy = [BayesEval()] * pars.nchains
traing_emc(pars, multichain_model, train_loader, criterion)