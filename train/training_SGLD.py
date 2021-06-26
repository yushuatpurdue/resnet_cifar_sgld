
'''
The files are created by Yushu Huang for the EMC-SGLD algorithm. YEAR 2021. 
They aim to apply the algorithm to the training of resnet under CIFAR dataset. 

In this file: training function. 
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

## import local

from BayesEval import BayesEval
from optimizers.SGLD import SGLD


def training_SGLD(pars, model, train_dataset, test_dataset, criterion):

    nnet = 0
    accuracy = [BayesEval()] * pars.nchains

    for net in model:

        ''' training nets '''
        # take a net from the ModuleList: multichain_model, and put it into the optimizer. 
        optimizer = SGLD(net.parameters(), lr=pars.lr, temperature=pars.T)        # no ladder here. need to modify later. 
        
        # print("nnet = ", nnet) # 要注释掉

        for i, (images, labels) in enumerate(train_dataset):

            # first change the data type from 'tensor' to 'Variables'. 
            images = Variable(images).cuda() if torch.cuda.is_available() else Variable(images)
            labels = Variable(labels).cuda() if torch.cuda.is_available() else Variable(labels)

            # then define optimizer, evaluate, calculate loss and back propagate. 
            optimizer.zero_grad()
            output = net(images)
            loss = criterion(output, labels)
            
            # print(loss.item())       # 要注释掉
            # print("data = ", i)       # 要注释掉

            loss.backward()
            optimizer.step()

        ''' evaluating performance '''
        accuracy[nnet].eval(net, test_dataset, cumulate = True)
        print(accuracy[nnet].OneEpoch_acc)
        
        nnet += 1


