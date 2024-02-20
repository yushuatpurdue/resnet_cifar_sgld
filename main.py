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
import time
import csv
import argparse
from torch.optim import optimizer
from tqdm import tqdm ## better progressbar
from math import exp
from sys import getsizeof
import numpy as np
import random
from os.path import exists

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
import network.resnet as resnet         ## all resnet models. 
from LoadData import LoadData, LoadUCI
from optimizers.SGLD import SGLD
from train.training_parallel import training_parallel
from train.training_re import training_re
from train.training_emc import training_emc
from network.CNN import CNN
from network.NN import NN

'''
define the main function
'''
def main():
    ''' first define addon arguments for running this file. '''
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
    parser.add_argument('-method', default='emc', help='correction factor F')             ## method = emc or sghmc or re

    ## dataset properties: 
    parser.add_argument('-data', default='Boston', help='CIFAR10/ CIFAR100')  ## data to be used (adds an attribute '-data', and names it as 'data'
    parser.add_argument('-batch', default=256, type=int, help='Batch size')   ## batch size for both training and testing
    parser.add_argument('-total', default=50000, type=int, help='Total data points')

    ## neural network properties:
    parser.add_argument('-model', default='NN', help='which neural networ model to use. Resnet in local package "resnet.py", CNN in "CNN.py" ')
    parser.add_argument('-layer', default=1, type=int, help='number of hidden layer for model = NN')
    parser.add_argument('-unit', default=[50], type=int, nargs='+', help='number of hidden unit in each layer of model = NN')

    ## environmental properties: 
    parser.add_argument('-seed', default=random.randint(1, 1e6), type=int, help='Random Seed')
    parser.add_argument('-jobid', default='1', help='for running seperate jobs and save them seperately')
    # parser.add_argument('-seed', default=426, type=int, help='Random Seed')

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


    ''' set data '''
    if pars.data in ['cifar10', 'cifar100']:
        train_loader, test_loader = LoadData(pars.batch, pars.batch, pars)
    if pars.data in ['Boston','Concrete','Energy','Yacht','Wine','kin8nm','CCPP','Naval','Protein','Year']:
        train_loader, test_loader = LoadUCI(data_name=pars.data, cross_validate_index=1, train_size=pars.batch, test_size=pars.batch)

    ''' set model '''

    if pars.data == 'cifar100':
        num_classes = 100
    elif pars.data == 'cifar10':
        num_classes = 10
    elif pars.data == 'Boston':
        input_dim = 13
        output_dim = 1
    elif pars.data == 'Concrete':
        input_dim = 8
        output_dim = 1


    multichain_model = nn.ModuleList()

    if "resnet" in pars.model:                                 # if using resnet as the training model
        for i in range(pars.nchains):
            savepath = './output/' + pars.method +'_'+ pars.jobid + '_' + str(i) + ".psh"
            if exists(savepath):                               # check if checkpoint of model_i exists 
                checkpoint = torch.load(savepath)              # If exists then load checkpoint
                ## Resume model_i from last checkpoint: 
                model_add =resnet.__dict__[pars.model](num_classes = num_classes)
                model_add.load_state_dict(checkpoint['state_dict'])
                model_add = nn.DataParallel(model_add)
                model_add.cuda()
                multichain_model.extend([model_add])

                ## Resume epoch from last checkpoint
                epoch = checkpoint['epoch']
            else:                                        # If one checkpoint missed
                ## Then build multichain_model anew. 
                model = nn.DataParallel(resnet.__dict__[pars.model](num_classes = num_classes))
                model.cuda()
                multichain_model = nn.ModuleList()
                for i in range(pars.nchains):
                    model_add = copy.deepcopy(model)
                    multichain_model.extend([model_add])

                epoch = -1
                break             # break for loop
    elif "CNN" in pars.model:
        for i in range(pars.nchains):
            savepath = './output/' + pars.method +'_'+ pars.jobid + '_' + str(i) + ".psh"
            if exists(savepath):                               # check if checkpoint of model_i exists 
                checkpoint = torch.load(savepath)              # If exists then load checkpoint
                ## Resume model_i from last checkpoint: 
                model_add =  CNN(num_classes=num_classes)
                model_add.load_state_dict(checkpoint['state_dict'])
                model_add = nn.DataParallel(model_add)
                model_add.cuda()
                multichain_model.extend([model_add])

                ## Resume epoch from last checkpoint
                epoch = checkpoint['epoch']
            else:                                        # If one checkpoint missed
                ## Then build multichain_model anew. 
                model = nn.DataParallel(CNN(num_classes=num_classes))
                model.cuda()
                multichain_model = nn.ModuleList()
                for i in range(pars.nchains):
                    model_add = copy.deepcopy(model)
                    multichain_model.extend([model_add])

                epoch = -1
                break             # break for loop
    elif "NN" in pars.model:
        for i in range(pars.nchains):
            savepath = './output/' + pars.method +'_'+ pars.jobid + '_' + str(i) + ".psh"
            if exists(savepath):                               # check if checkpoint of model_i exists 
                checkpoint = torch.load(savepath)              # If exists then load checkpoint
                ## Resume model_i from last checkpoint: 
                model_add =  NN(pars.layer, pars.unit, input_dim=input_dim, output_dim=output_dim)
                model_add.load_state_dict(checkpoint['state_dict'])
                model_add = nn.DataParallel(model_add)
                model_add.cuda()
                multichain_model.extend([model_add])

                ## Resume epoch from last checkpoint
                epoch = checkpoint['epoch']
            else:                                        # If one checkpoint missed
                ## Then build multichain_model anew. 
                model = nn.DataParallel(CNN(num_classes=num_classes))
                model.cuda()
                multichain_model = nn.ModuleList()
                for i in range(pars.nchains):
                    model_add = copy.deepcopy(model)
                    multichain_model.extend([model_add])

                epoch = -1
                break             # break for loop
    else:
        raise Exception("model specified is not available. ")
    


    ''' training loop '''
    criterion = nn.CrossEntropyLoss()            # loss function for training. CrossEntropyLoss generates the potential. 
    # accuracy = [BayesEval()] * pars.nchains
    accuracy = [BayesEval()]
    for i in range(pars.nchains - 1):
        accuracy = accuracy + [BayesEval()]

    crossoverRate = copy.deepcopy(pars.crossoverRate)

    if pars.method == 'sghmc':

    
        ## initializing T & eta from checkpoint (if checkpoint exists)
        # if epoch < pars.burn * pars.epochs:
        if epoch < 600:
            pars.T /= pars.Tanneal**epoch
        else:
            pars.T /= pars.Tanneal**(600)

        # if epoch > 0.4 * pars.epochs:
        if epoch > 400:
            pars.eta *= pars.etaanneal**(epoch - 400)

        start = time.time()
        for t in range(epoch+1, pars.epochs):
            # if t < pars.burn * pars.epochs:
            if t < 600:
                pars.T /= pars.Tanneal
            # if t > 0.4 * pars.epochs:
            if t > 400:
                pars.eta *= pars.etaanneal

            # crossoverGap = 30
            # if t <30:
            #     crossoverGap = 30
            # # elif t>200:
            # #     crossoverGap = 200
            # else:

            print("epochs = ",t)      

            SGHMC_idx = list(range(pars.nchains))
            training_parallel(pars, multichain_model, SGHMC_idx, train_loader, criterion)
            ''' print accuracy '''
            for i in range(pars.nchains):
                accuracy[i].eval(multichain_model[i], test_loader, cumulate = False)
                print('Chain {} Acc: {:0.2f} BestAcc: {:0.2f}'\
                        .format(i, accuracy[i].OneEpoch_acc, accuracy[i].best_OneEpoch_acc))

        end = time.time()
        print("total running time = ", end - start)
        ''' save models'''
        for i in range(pars.nchains):
            state = {
                    'state_dict': multichain_model[i].module.state_dict(),
                    'epoch'     : t
            }
            savepath = './output/' + pars.method +'_'+ pars.jobid + '_' + str(i) + ".psh"
            torch.save(state, savepath)
            

    elif pars.method == 're':
        ## initializing T & eta from checkpoint (if checkpoint exists)
        # if epoch < pars.burn * pars.epochs:
        if epoch < pars.burn * 1000:
            pars.T /= pars.Tanneal**epoch
        else:
            pars.T /= pars.Tanneal**(pars.burn * 1000)

        # if epoch > 0.4 * pars.epochs:
        if epoch > 0.4 * 1000:
            pars.eta *= pars.etaanneal**(epoch - 0.4 * 1000)

        start = time.time()
        for t in range(epoch+1, pars.epochs):
            # if t < pars.burn * pars.epochs:
            if t < pars.burn * 1000:
                pars.T /= pars.Tanneal
            # if t > 0.4 * pars.epochs:
            if t > 0.4 * 1000:
                pars.eta *= pars.etaanneal

            # crossoverGap = 30
            # if t <30:
            #     crossoverGap = 30
            # # elif t>200:
            # #     crossoverGap = 200
            # else:

            print("epochs = ",t)  

            if t<50:
                training_parallel(pars, multichain_model, range(pars.nchains), train_loader, criterion)
            else:
                output = training_re(pars, multichain_model, train_loader, criterion)
                print("total swap = ", output["swap"])
            ''' print accuracy '''
            for i in range(pars.nchains):
                accuracy[i].eval(multichain_model[i], test_loader, cumulate = False)
                print('Chain {} Acc: {:0.2f} BestAcc: {:0.2f}'\
                        .format(i, accuracy[i].OneEpoch_acc, accuracy[i].best_OneEpoch_acc))

        end = time.time()
        print("total running time = ", end - start)
        ''' save models'''
        for i in range(pars.nchains):
            state = {
                    'state_dict': multichain_model[i].module.state_dict(),
                    'epoch'     : t
            }
            savepath = './output/' + pars.method +'_'+ pars.jobid + '_' + str(i) + ".psh"
            torch.save(state, savepath)

                            
    elif pars.method == 'emc':
                ## initializing T & eta from checkpoint (if checkpoint exists)
        # if epoch < pars.burn * pars.epochs:
        if epoch < 600:
            pars.T /= pars.Tanneal**epoch
        else:
            pars.T /= pars.Tanneal**(600)

        # if epoch > 0.4 * pars.epochs:
        if epoch > 400:
            pars.eta *= pars.etaanneal**(epoch - 400)

        start = time.time()
        for t in range(epoch+1, pars.epochs):
            # if t < pars.burn * pars.epochs:
            if t < 600:
                pars.T /= pars.Tanneal
            # if t > 0.4 * pars.epochs:
            if t > 400:
                pars.eta *= pars.etaanneal

            # crossoverGap = 30
            # if t <30:
            #     crossoverGap = 30
            # # elif t>200:
            # #     crossoverGap = 200
            # else:
            crossoverGap = t

            print("epochs = ",t)  

            if t<500: 
                if t<50 or t%10 != 0:
                    training_parallel(pars, multichain_model, range(pars.nchains), train_loader, criterion)
                else: 
                    training_emc(pars, multichain_model, train_loader, crossoverRate, crossoverGap, criterion)
            else:
                if t%20 != 0:
                    training_parallel(pars, multichain_model, range(pars.nchains), train_loader, criterion)
                else:
                    training_emc(pars, multichain_model, train_loader, crossoverRate, crossoverGap, criterion)

        
            ''' print accuracy '''
            for i in range(pars.nchains):
                accuracy[i].eval(multichain_model[i], test_loader, cumulate = False)
                print('Chain {} Acc: {:0.2f} BestAcc: {:0.2f}'\
                        .format(i, accuracy[i].OneEpoch_acc, accuracy[i].best_OneEpoch_acc))

        end = time.time()
        print("total running time = ", end - start)

        ''' save models'''
        for i in range(pars.nchains):
            state = {
                    'state_dict': multichain_model[i].module.state_dict(),
                    'epoch'     : t
            }
            savepath = './output/' + pars.method +'_'+ pars.jobid + '_' + str(i) + ".psh"
            torch.save(state, savepath)


main()

