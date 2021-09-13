import math
import copy
import sys
import os
import timeit
import csv
import copy
import argparse
from torch.nn import parameter
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


## define an object "parser" to add attributes to it. 
parser = argparse.ArgumentParser()    

## mcmc properties:
parser.add_argument('-nchains', default=5, type=int, help='Total number of chains')     ## how many chains are going to run. 
parser.add_argument('-epochs', default=1000, type=int, help='Sampling Epochs')       ## how many epochs (something like iterations) are going to run. 
parser.add_argument('-selectionT', default=1e5, type=float, help='Selection temperature for snooker crossover')      ## used in calculating Boltzmann distribution in snooker crossover. 
parser.add_argument('-crossoverRate', default=0.3, type=float, help='Rate of the chains for doing crossover')      ## how many chains in the population are going to go through snooker crossover. 
parser.add_argument('-etagap', default=0.66, type=float, help='gap between different chains etas. ')    ## learning rate of the second high temperature chain = lrgap * learning rate for highest temperatur chain
parser.add_argument('-T', default=0.05, type=float, help='Temperature for highest temperature chain')      ## highest temperature of chains. 
parser.add_argument('-Tgap', default=0.5, type=float, help='Temperature gap between chains')       ## the rest of the chains have temperatures equispaced with this gap. 
parser.add_argument('-eta', default=2e-6, type=float, help='squared Sampling learning rate')     ## learning_rate^2/mass for highest temperatur chain. see SGHMC paper page 6
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
# model.cuda()

## define model for each chain, and append them into a list. 
multichain_model = nn.ModuleList([model])
for i in range(pars.nchains-1):
    model_add = copy.deepcopy(model)
    multichain_model.extend([model_add])



# print(len(multichain_model))
# print(type(model))

# for net in multichain_model:
#     print(len(list(net.parameters())))
# print(len(list(multichain_model.parameters())))          # 结论：如果不用for loop 访问每一个model，直接访问multichain_model 的parameters 的结果就是只访问了第一个model
# multichain_model.extend([model for i in range(1, pars.nchains)])
# print(multichain_model.parameters)
# print((list(multichain_model))[0])
# print(model.parameters)




# print(len((SGLD(model.parameters(), lr=0.1,temperature=1).param_groups[0])[1]))           # incorrect code. 

# print(type(torch.optim.SGD(model.parameters(), lr=0.1).param_groups[0]))    # dict
# print(SGHMC(model.parameters(), eta=pars.eta, temperature=pars.T, wdecay=pars.wdecay).param_groups[0].keys())   # dict_keys(['params', 'eta', 'alpha', 'temperature', 'wdecay'])
# print(type(torch.optim.SGD(model.parameters(), lr=0.1).param_groups[0]["params"]))    # list
# print(len(torch.optim.SGD(model.parameters(), lr=0.1).param_groups[0]["params"][0] - torch.optim.SGD(model.parameters(), lr=0.1).param_groups[0]["params"][0]))
# print(torch.optim.SGD(model.parameters(), lr=0.1).param_groups[0]["params"])
# print(type(torch.optim.SGD(model.parameters(), lr=0.1).param_groups[0]["params"][0]))    # torch.nn.parameter.Parameter
# print(torch.norm(torch.optim.SGD(model.parameters(), lr=0.1).param_groups[0]["params"][0]))   # tensor(2.4477, grad_fn=<CopyBackwards>), 能算，还能backward
# print(torch.norm(torch.optim.SGD(model.parameters(), lr=0.1).param_groups[0]["params"][0] - torch.optim.SGD(model.parameters(), lr=0.1).param_groups[0]["params"][0])) # 也能算。不过现在是0
# print(torch.norm(model.parameters))
# print(type(torch.optim.SGD(model.parameters(), lr=0.1).param_groups[0]["params"][0]))

# count = 0
# for name, parameter in model.named_parameters():
#     # print(parameter)
#     # break
#     print(name)
#     print(torch.norm(parameter))
#     # print(parameter.grad)
#     count+=1
# print(count)

train_loader, test_loader = LoadData(pars.batch, pars.batch, pars)

criterion = nn.CrossEntropyLoss()



'''
for t in range(100):

    print("epochs = ",t)      # 要注释掉
    training_parallel(pars, multichain_model, train_loader, test_loader, criterion)
    # parameter0 = list(multichain_model[0].parameters())[0]
    # parameter1 = list(multichain_model[4].parameters())[0]
    # # diff_parameter = parameter0 - parameter1
    # print("parameter0 =", parameter0[0])
    # print("parameter1 =", parameter1[0])
'''

# model0 = nn.DataParallel(resnet.__dict__[pars.model](num_classes = num_classes))
# model1 = nn.DataParallel(resnet.__dict__[pars.model](num_classes = num_classes))

diff_parameter = []
norm_diff = 0
count = len(list(multichain_model[0].parameters()))
print(type(model.parameters().item()))
'''
for i in range(count):
    parameter0 = list(multichain_model[0].parameters())[i]
    parameter1 = list(multichain_model[1].parameters())[i]   # 这样写它们确实会有区别
    temp = parameter0
    diff_parameter = diff_parameter + [temp]
    norm_diff += torch.norm(temp)
    # parameter0 = list(multichain_model[0].parameters())[i]
    # parameter1 = list(multichain_model[1].parameters())[i]
    # diff_parameter = parameter0 - parameter1
    # print(i, "th parameter of model 0 = ", parameter0[0])
    # print(i, "th parameter of model 1 = ", parameter1[0])
    # print(parameter1[0] - parameter2[0])
    # print("norm of diff in", i, "th parameter:", torch.norm(temp))
    # print("norm of ", i, "th parameter:", torch.norm(diff_param))
# print(norm_diff.item())
# print(diff_parameter[0][0])
print(type(diff_parameter))
# direction = np.array(diff_parameter)/norm_diff.item()
# print(direction[0][0])     # 没问题, 但在cuda 下不行
'''
# print(torch.norm(diff_parameter))         # 报错
# print(len(diff_parameter))
# print(diff_parameter == list(multichain_model[0].parameters()))      # output = TRUE. 
# print(torch.norm(diff_parameter[1]))      # 没错


# print(len(list(model.parameters())))    # 65
# for i in range(65):
#     print(len(list(model.parameters())[i])) 
# print(list(model.parameters())[62])

# print(torch.norm(list(model.parameters())[0]))
# print(torch.norm(list(model.parameters())[1]))

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
        
        print(parameters.__dict__)
        
        # state = torch.optim.SGD(model.parameters(), lr=0.1).state[parameters]
        # # state["iteration"] = 0
        # state["momentum"] = torch.zeros_like(parameters)
        # momentum = state["momentum"]
        # momentum.add_(2.)
        # # print(momentum)
        # print(state["momentum"])
        
        break
'''

'''
train_loader, test_loader = LoadData(256, 256, pars)

# print(len(train_loader))  # len(test_loader = 40), len(train_loader = 196)


# for train in enumerate(train_loader):
#     # print(len(train))   # len(train) = 2
#     # print(train[0])   # train[0] = 0
#     print(len(train[1][0][255][2][31][31]))  # train[1] is a large tensor. train[1][0] is an md tensor of size 256*3*32*32  train[1][1] is a tensor of size 256. 3 should be color, 256*256 should be pixels. 
#     # 196*256 = 50176, size of train set is 50000. 40*256  = 10240, size of test set is 10000. 
#     break

for i, (images, labels) in enumerate(train_loader):
    print('batch length', len(images))
    break
'''
    



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
'''
criterion = nn.CrossEntropyLoss()
# accuracy = [BayesEval()] * pars.nchains


         


for t in range(pars.epochs):

    print("epochs = ",t)      # 要注释掉
    training_parallel(pars, pars.eta, pars.T, multichain_model, train_loader, test_loader, criterion)
    # nnet = 0                  # 要注释掉

    # for net in multichain_model:
    #     trainer(net, train_loader, test_loader, pars)
        
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


# accuracy    = [BayesEval()] * pars.nchains    # a list recording the accuracy of model. 
# print(len(accuracy))


# Boltzmann_pd = [1]*3
# Boltzmann_pd = Boltzmann_pd/np.sum(Boltzmann_pd)
# print("Boltzmann_pd = ", Boltzmann_pd)

# nchainsCrossover = math.ceil(pars.crossoverRate * pars.nchains)
# CrossoverInd = random.sample(range(pars.nchains), nchainsCrossover)
# print("how many chains are going through crossover? ans = ", nchainsCrossover)
# print("crossover chains' index = ", CrossoverInd)

# sample_anchor_chain = np.random.multinomial(1, [0.,         0.6846285,  0.22088984, 0.07144526, 0.0230364 ])
# print("anchor_chain = ", anchor_chain)
# # print("index of anchor chain = ", np.nonzero(anchor_chain)[0][0])
# print("index of anchor chain = ", int(np.nonzero(sample_anchor_chain)[0]))
# print("index of anchor chain = ", int(anchor_chain[0]))

# output = dict()
# output["crossoverInd"] = [0,8,1]
# output["anchorInd"] = [2,3,4]
# print("type of output = ",type(output["crossoverInd"]))

# print(type(torch.FloatTensor(3,3).normal_()))
# print(list(torch.FloatTensor(3,3).normal_())[0])
