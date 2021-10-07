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
    Input: 
        pars            defined in main() by argparser. 
        modulelist      a ModuleList of neural networks to train. 
        train_dataset      
        criterion       loss function. should be nn.CrossEntropyLoss() to obtain the potential. 
    '''

    ''' snooker'''
    for model in modulelist:
        model.train()
    Snooker_test = snooker(train_dataset, criterion = nn.CrossEntropyLoss(), selection_t = pars.selectionT)   # selection_t = 100000 is a good choice for next to burn-in period. 
    crossover_idx = Snooker_test.step(pars, modulelist, train_dataset)

    ''' index of chains to go through SGHMC '''
    SGHMC_idx = list(range(pars.nchains))
    for i in crossover_idx:
        SGHMC_idx.remove(i)


    ''' SGHMC '''
    training_parallel(pars, modulelist, SGHMC_idx, train_dataset, criterion)
    