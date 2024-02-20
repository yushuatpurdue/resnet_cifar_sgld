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
import pandas as pd
import zipfile
import random
import pickle
from os.path import exists
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
import network.resnet as resnet  # all resnet models.
from LoadData import LoadData, LoadUCI
from optimizers.SGLD import SGLD as optimizing_class
from BayesEval import BayesEval
from train.training_parallel import training_parallel
from optimizers.SGLD import SGLD
from optimizers.SGHMC import SGHMC
from network.CNN import CNN
from network.NN import NN


## define an object "parser" to add attributes to it. 
parser = argparse.ArgumentParser()    

## mcmc properties:
parser.add_argument('-nchains', default=20, type=int, help='Total number of chains')     ## how many chains are going to run. 
parser.add_argument('-epochs', default=100, type=int, help='Sampling Epochs')       ## how many epochs (something like iterations) are going to run. 
parser.add_argument('-selectionT', default=1e5, type=float, help='Selection temperature for snooker crossover')      ## used in calculating Boltzmann distribution in snooker crossover. 
parser.add_argument('-crossoverRate', default=0.1, type=float, help='Rate of the chains for doing crossover')      ## how many chains in the population are going to go through snooker crossover. 
parser.add_argument('-T', default=0.005, type=float, help='Temperature for highest temperature chain')      ## lowest temperature of chains. 
parser.add_argument('-Tgap', default=0.65, type=float, help='Temperature gap between chains')       ## the next chain has temperatures = T/Tgap. 
parser.add_argument('-eta', default=2e-6, type=float, help='squared Sampling learning rate')     ## learning_rate^2/mass for highest temperatur chain. see SGHMC paper page 6
parser.add_argument('-etagap', default=0.8, type=float, help='gap between different chains etas. ')    ## eta of the second lowest temperature chain = eta/etagap for lowest temperature chain
parser.add_argument('-period', default=2, type=int, help='estimate adaptive variance every [period] epochs')
parser.add_argument('-cycle', default=1, type=int, help='Number of cycles')
parser.add_argument('-burn', default=0.6, type=float, help='burn in iterations for sampling (sn * burn)')
parser.add_argument('-etaanneal', default=0.984, type=float, help='lr annealing factor')
parser.add_argument('-Tanneal', default=1.02, type=float, help='temperature annealing factor')
parser.add_argument('-wdecay', default=25, type=float, help='Sampling weight decay')
parser.add_argument('-iterationR', default=1, type=int, help='number of epochs for SGLD in snooker scaler r')      ## how many iterations are going to run for SGLD sampling r in snooker. 
parser.add_argument('-lrR', default=1e-4, type=float, help='learning rate of SGLD in snooker scaler r')      ## max learning rate for SGLD sampling r in snooker. If r is close to 0 then choose a smaller learning rate. 1e-4 is good for resnet. CNN need a smaller one. 
parser.add_argument('-rInitScale', default=10, type=float, help='standard deviation for initializing r from a Normal distribution')      ## For resnet a value of 10 is good. For CNN need to have smaller value.   
parser.add_argument('-bias_F', default=3e5, type=float, help='correction factor F')
parser.add_argument('-method', default='sghmc', help='correction factor F')             ## method = emc or sghmc or re

## dataset properties: 
parser.add_argument('-data', default='Boston', help='CIFAR10/ CIFAR100')  ## data to be used (adds an attribute '-data', and names it as 'data'
parser.add_argument('-batch', default=256, type=int, help='Batch size')   ## batch size for both training and testing
parser.add_argument('-total', default=50000, type=int, help='Total data points')
parser.add_argument('--regression_flag', default=True, type=int, help='true for regression and false for classification')

## neural network properties:
parser.add_argument('-model', default='NN', help='which neural networ model to use. Resnet in local package "resnet.py", CNN in "CNN.py" ')
parser.add_argument('-layer', default=1, type=int, help='number of hidden layer for model = NN')
parser.add_argument('-unit', default=[50], type=int, nargs='+', help='number of hidden unit in each layer of model = NN')

## environmental properties: 
parser.add_argument('-seed', default=random.randint(1, 1e6), type=int, help='Random Seed')
parser.add_argument('-jobid', default='1', help='for running seperate jobs and save them seperately')

## add the arguments. 
pars = parser.parse_args()

# torch.manual_seed(pars.seed)           # set seed for torch
# torch.cuda.manual_seed_all(pars.seed)      # set seed for all GPUs. 
np.random.seed(pars.seed)              # set seed for numpy. !!It returns the random numbers with exactly the same order.          
# random.seed(pars.seed)                 # set seed in general. 
# torch.backends.cudnn.deterministic=True # make reproducible.
# boltzmann_weight =  [0.067, 0.064, 0.,    0.063, 0.064, 0.059, 0.057, 0.057, 0.059, 0.057 ,0.052 ,0.039,
#  0.049, 0.044, 0.048, 0.046, 0.049, 0.041, 0.046, 0.038]
# print(np.random.multinomial(1, boltzmann_weight))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# if pars.data == 'cifar100':
#     num_classes = 100
# elif pars.data == 'cifar10':
#     num_classes = 10
# for i in range(8):
#     savepath = './output/model' + str(i) + ".psh"
#     if exists(savepath):
#         print("exists")
#         checkpoint = torch.load(savepath)
#         model = resnet.__dict__[pars.model](num_classes = num_classes)
#         model.load_state_dict(checkpoint['state_dict'])
#         model = nn.DataParallel(model)
#     else:
#         print(["model", i, "not exist"])
#         break

# if "resnet" in pars.model:


    # model = nn.DataParallel(resnet.__dict__[pars.model](num_classes = num_classes))
# print(model.__dict__)
# elif "CNN" in pars.model:
#     model = nn.DataParallel(CNN(num_classes=num_classes))
# else:
#     raise Exception("model specified is not available. ")
# # print(model.parameters())

# model.cpu()
# # model.cuda()


## define model for each chain, and append them into a list. 
# multichain_model = nn.ModuleList()
# for i in range(pars.nchains):
# # for i in range(1):
#     model_add = copy.deepcopy(model)
#     multichain_model.extend([model_add])

    # state = {
    #         'state_dict': model.module.state_dict()
    # }
    # savepath = './output/model' + str(i) + ".psh"
    # torch.save(state,savepath)

# print(len(multichain_model))

# for i in range(len(list(multichain_model[0].parameters()))):
#     ifequal = list(multichain_model[0].parameters())[i] == list(multichain_model[1].parameters())[i]
#     print(torch.prod(ifequal))
'''
count = 0
innerproduct = 0
dimension = 0
for parameter in model.parameters():
    # print("count = ", count)
    print("parameter #", count," head, before modified:", parameter.data[0])
    count += 1
    break
            
'''
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
# innerproduct = 0
# for parameter in model.parameters():
#     print(list((parameter.size())))
            # temp = parameter.data * parameter.data
            # innerproduct += temp.sum()
            # print(innerproduct)
# innerproduct = innerproduct.item()
# print(innerproduct/0.005)
# for name, parameter in model.named_parameters():
#     size =list(parameter.size()) 
#     print(np.prod(size))     # returns sizes
    # print(parameter.sum())
#     break
#     # break
#     print(name)
#     print(torch.norm(parameter))
#     # print(parameter.grad)
#     count+=1
# print(count)

# train_loader, test_loader = LoadData(pars.batch, pars.batch, pars)
# if pars.data in ['cifar10', 'cifar100']:
#     print("T")
# else:
#     print("F")

train_loader, test_loader = LoadUCI(data_name=pars.data, cross_validate_index=1, train_size=pars.batch, test_size=pars.batch)
# for i, (images, labels) in enumerate(test_loader):
#     print(images.get_device())
#     print(labels.get_device())
# for i, (images, labels) in enumerate(train_loader):
#     print(images.get_device())
#     print(labels.get_device())

if pars.data == 'Boston':
        input_dim = 13
        output_dim = 1
multichain_model = nn.ModuleList()
for i in range(pars.nchains):
    savepath = './output/' + pars.method +'_'+ pars.jobid + '_' + str(i) + ".psh"
    if exists(savepath):                               # check if checkpoint of model_i exists 
        checkpoint = torch.load(savepath)              # If exists then load checkpoint
        ## Resume model_i from last checkpoint: 
        model_add =  NN(pars.layer, pars.unit, input_dim=input_dim, output_dim=output_dim)
        model_add.load_state_dict(checkpoint['state_dict'])
        model_add = nn.DataParallel(model_add)
        # model_add.cuda()
        model_add.to(device)
        multichain_model.extend([model_add])

        ## Resume epoch from last checkpoint
        epoch = checkpoint['epoch']
    else:                                        # If one checkpoint missed
        ## Then build multichain_model anew. 
        model = nn.DataParallel(NN(pars.layer, pars.unit, input_dim=input_dim, output_dim=output_dim))
        # model.cuda()
        model.to(device)
        multichain_model = nn.ModuleList()
        for i in range(pars.nchains):
            model_add = copy.deepcopy(model)
            multichain_model.extend([model_add])

        epoch = -1
        break   

# for model in multichain_model:
#     print(next(model.parameters()).device)
criterion = nn.MSELoss()

for t in range(5):
     SGHMC_idx = list(range(pars.nchains))
    #  training_parallel(pars, multichain_model, SGHMC_idx, train_loader, criterion)
     for idx in SGHMC_idx:

        net = multichain_model[idx]
        net.to(device)
        print(next(net.parameters()).device)

        ''' training net '''
        net.train()      # first set net to train mode. 
        # print("net", nnet, "parameter head", list(net.parameters())[0][0])
        eta         = pars.eta / (pars.etagap ** idx)  # different chains occupy different eta
        temperature = pars.T / (pars.Tgap ** idx)    # different chains occupy different temperature
        
        # take a net from the ModuleList: multichain_model, and put it into the optimizer. 
        # optimizer = optim.SGD(net.parameters(), lr=eta, momentum=0.9)        # no ladder here. need to modify later. 

        optimizer = SGHMC(net.parameters(), eta=eta, temperature=temperature, wdecay=pars.wdecay)
        
        # print("nnet = ", nnet) # 要注释掉

        for i, (images, labels) in enumerate(train_loader):

            # first change the data type from 'tensor' to 'Variables'. 
            images = Variable(images).cuda() if torch.cuda.is_available() else Variable(images)
            labels = Variable(labels).cuda() if torch.cuda.is_available() else Variable(labels)
            print(images.get_device())
            print(labels.get_device())

            # then evaluate optimizer, calculate loss and back propagate. 
            optimizer.zero_grad()
            output = net(images)
            nn.utils.clip_grad_norm_(net.parameters(), 0.1)
            loss   = criterion(output, labels) * pars.total
            
            # print(loss.item())       # 要注释掉
            # print("data = ", i)       # 要注释掉

            loss.backward()
            optimizer.step()

        ''' evaluating performance '''
        # accuracy[nnet].eval(net, test_dataset, cumulate = False)

        ''' print results '''
        print('Chain {} Loss: {:0.2f} T: {:.2E} eta: {:.2E}'\
                    .format(idx, \
                    loss.item(), temperature, eta))
# count = 0
# for i, (images, labels) in enumerate(train_loader):
#     print(len(images))
#     count+=len(images)
# print(count)

# temp = pd.read_csv('./data/UCI/kin8nm.csv')
# temp = np.mat(temp)
# print(temp[0:10,:])
# for i, (images, labels) in enumerate(test_loader):
#     count+=len(images)
# print(count)
# criterion = nn.CrossEntropyLoss()


# for t in range(1):

#     print("epochs = ",t)      # 要注释掉
#     training_parallel(pars, multichain_model, range(len(multichain_model)),train_loader, criterion)
# #     parameter0 = list(multichain_model[0].parameters())[0]
# for parameters in multichain_model[0].parameters():
#     parameters.grad.data.zero_()
#     print(parameters.grad.data[0])

    # print(parameter0[0].grad.data())          # 报错：AttributeError: 'NoneType' object has no attribute 'data'
    # parameter1 = list(multichain_model[4].parameters())[0]
    # # diff_parameter = parameter0 - parameter1
    # print("parameter0 =", parameter0[0])
    # print("parameter1 =", parameter1[0])


# model0 = nn.DataParallel(resnet.__dict__[pars.model](num_classes = num_classes))
# model1 = nn.DataParallel(resnet.__dict__[pars.model](num_classes = num_classes))
'''
diff_parameter = []
norm_diff = 0
count = len(list(multichain_model[0].parameters()))
print(type(model.parameters()))
'''
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

# r = np.random.normal(scale = 10)
# print(r)
# r = np.random.normal(scale = 10)
# print(r)
# result = [r]
# print(result+[r])

# SGHMC_idx = list(range(pars.nchains))
# for i in [1,3]:
#     SGHMC_idx.remove(i)

# print(SGHMC_idx)

''''''
# for model in multichain_model:
#     model.train()

# Boltzmann_pd = [0]*len(multichain_model)
# loss         = []

# for i, (images, labels) in enumerate(train_loader):

#     # first change the data type from 'tensor' to 'Variables'. 
#     images = Variable(images).cuda() if torch.cuda.is_available() else Variable(images)
#     labels = Variable(labels).cuda() if torch.cuda.is_available() else Variable(labels)
#     for j in range(len(multichain_model)):
#         model = multichain_model[j]
#         # then evaluate optimizer, calculate loss and back propagate. 
#         output = model(images)
#         loss   += [criterion(output, labels) * pars.total]

#     print(loss)
#     break