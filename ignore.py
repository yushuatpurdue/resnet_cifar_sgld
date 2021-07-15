import math
import copy
import sys
import os
import timeit
import csv
import argparse
from tqdm import tqdm  # better progressbar
from math import exp
from sys import getsizeof
import numpy as np
import random
import pickle
# import pytorch modules
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
from torch.optim import Optimizer
import torchvision.models as models

## import local
import resnet  # all resnet models.
from LoadData import LoadData
from optimizers.SGLD import SGLD as optimizing_class
from BayesEval import BayesEval
from train.training_parallel import training_parallel
from optimizers.SGLD import SGLD
from optimizers.SGHMC import SGHMC
from train.trainer import trainer


## define an object "parser" to add attributes to it. 
parser = argparse.ArgumentParser()    

## mcmc properties:
parser.add_argument('-nchains', default=1, type=int, help='Total number of chains')     ## how many chains are going to run. 
parser.add_argument('-epochs', default=1000, type=int, help='Sampling Epochs')       ## how many epochs (something like iterations) are going to run. 
parser.add_argument('-T', default=0.1, type=float, help='Temperature for highest temperature chain')      ## highest temperature of chains. 
parser.add_argument('-Tgap', default=0.01, type=float, help='Temperature gap between chains')       ## the rest of the chains have temperatures equispaced with this gap. 
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


model.cpu()

## define model for each chain, and append them into a list. 
multichain_model = nn.ModuleList([model])
multichain_model.extend([model for i in range(1, 2)])

# for net in multichain_model:
#     print(len(list(net.parameters())))
# print(len(list(multichain_model.parameters())))          # 结论：如果不用for loop 访问每一个model，直接访问multichain_model 的parameters 的结果就是只访问了第一个model
# multichain_model.extend([model for i in range(1, pars.nchains)])
# print(multichain_model.parameters)
# print((list(multichain_model))[0])
# print(model.parameters)




# print(len((SGLD(model.parameters(), lr=0.1,temperature=1).param_groups[0])[1]))           # incorrect code. 

# print(type(torch.optim.SGD(model.parameters(), lr=0.1).param_groups[0]))
# groups = SGLD(model.parameters(), lr=0.1).param_groups[0]
# print(len(list(groups.values())[0][0][0]))

# print(list(groups.values())[0][0][0])
# print(torch.optim.SGD(model.parameters(), lr=0.1).state)

# print(type(torch.optim.SGD(multichain_model.parameters(), lr=0.1).param_groups[0]))
# groups = torch.optim.SGD(multichain_model.parameters(), lr=0.1).param_groups[1]
# print(len(list(groups.values())[0][0][0]))
# print(groups)
'''
for group in torch.optim.SGD(model.parameters(), lr=0.1).param_groups:
    # print(len(group))
    # break
    for parameters in group["params"]:
        
        # print(parameters.__dict__)
        
        state = torch.optim.SGD(model.parameters(), lr=0.1).state[parameters]
        # state["iteration"] = 0
        print(state)
        
        break
'''


train_loader, test_loader = LoadData(256, 256, pars)

# print(len(test_loader))


# for train in enumerate(train_loader):
#     print(len(train[1][0][255]))
#     break

'''
optimizer = SGLD(model.parameters(), lr=0.1, temperature=0)     # 做了个实验发现temperature = 1 is like 太大了
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()


for i, (images, labels) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(images)
    loss = criterion(output, labels)
    print(loss.item())
    loss.backward()
    optimizer.step()
'''
'''
    optimizer.zero_grad()
    output = model(images)
    loss = criterion(output, labels)
    print(loss.item())
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    output = model(images)
    loss = criterion(output, labels)
    print(loss.item())
    '''

criterion = nn.CrossEntropyLoss()
# accuracy = [BayesEval()] * pars.nchains


         

for t in range(pars.epochs):

    print("epochs = ",t)      # 要注释掉
    # training_parallel(pars, multichain_model, train_loader, test_loader, criterion)
    nnet = 0                  # 要注释掉

    for net in multichain_model:
        trainer(net, train_loader, test_loader, pars)
        
    #     # take a net from the ModuleList: multichain_model, and put it into the optimizer. 
    #     optimizer = SGHMC(net.parameters(), lr=0.001, temperature=0.05, friction=0.05)
        
    #     print("nnet = ", nnet) # 要注释掉

    #     for i, (images, labels) in enumerate(train_loader):

    #         # first change the data type from 'tensor' to 'Variables'. 
    #         images = Variable(images).cuda() if torch.cuda.is_available() else Variable(images)
    #         labels = Variable(labels).cuda() if torch.cuda.is_available() else Variable(labels)

    #         # then define optimizer, evaluate, calculate loss and back propagate. 
    #         optimizer.zero_grad()
    #         output = net(images)
    #         # print(output)
    #         loss = criterion(output, labels)
            
    #         print(loss.item())
    #         # print("data = ", i)

    #         loss.backward()
    #         optimizer.step()




    #     # accuracy[nnet].eval(net, test_loader, cumulate = True)
    #     # print(accuracy[nnet].best_OneEpoch_acc)
    #     nnet += 1               # 要注释掉

'''
for net in multichain_model:
    print(net.parameters)
       
    # take a net from the ModuleList: multichain_model, and put it into the optimizer. 
    optimizer = SGLD(net.parameters(), lr=0.001, temperature=0.05)

    for i, (images, labels) in enumerate(train_loader):

        # first change the data type from 'tensor' to 'Variables'. 
        images = Variable(images).cuda() if torch.cuda.is_available() else Variable(images)
        labels = Variable(labels).cuda() if torch.cuda.is_available() else Variable(labels)

        # then define optimizer, evaluate, calculate loss and back propagate. 
        optimizer.zero_grad()
        # output = net(images)
        output = net.forward(images).data
        print(sum(output[0]))
        # loss = criterion(output, labels)
        
        # print(loss.item())
        # print("data = ", i)

        # loss.backward()
        # optimizer.step()

        break
    
    break

'''