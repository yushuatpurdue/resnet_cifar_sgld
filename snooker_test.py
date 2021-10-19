'''
The files are created by Yushu Huang for the EMC-SGLD algorithm. YEAR 2021. 
They aim to apply the algorithm to the training of resnet under CIFAR dataset. 
algorithm reference: Jun Liu: Monte Carlo Strategies in Scientific Computing, p232. 
In this file: a class -- SGHMC is created as the optimizer for the network. 
use 'from optimizers.SHMC import SGLD' to import this file. 
'''
from numpy.core.numeric import cross
from numpy.lib.twodim_base import _trilu_indices_form_dispatcher
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
import time
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
import resnet            ## all resnet models. 
from LoadData import LoadData
from optimizers.SGHMC import SGHMC
from train.training_parallel import training_parallel




class snooker: 
    '''Implements snooker crossover algorithm for one step update.
    Args:
        modulelist         : nn.ModuleList. the population of chains. 
        selection_t        : selection temperature. see reference.  
    '''

    def __init__(
        self,
        # modulelist,
        train_loader,
        criterion, 
        selection_t = 0.1
        ) -> None:

        if selection_t < 0:
            raise ValueError("Negative selection temperature: {}".format(selection_t))
        super(snooker, self).__init__()

        # self.modulelist   = modulelist
        self.train_loader = train_loader
        self.selection_t  = selection_t
        self.criterion    = criterion
        # self.nchains      = len(modulelist)


    def sample_chainpairs(self, pars, Boltzmann_pd):
        '''
        This function returns the indices of chains to go through snooker crossover, and their correspondent anchor chains. 
        Input:
            pars:          argument parser. 
            Boltzmann_pd:  unnormalized boltzmann pd.
            modulelist:    the model population. 
        '''

        nchainsCrossover = math.ceil(pars.crossoverRate * pars.nchains)           # how many chains are going through crossover
        CrossoverInd = random.sample(range(pars.nchains), nchainsCrossover)       # sample the index of those crossover chains

        Anchor_ind = [pars.nchains] * nchainsCrossover

        ## note: modify Boltzmann function so that the modulelist is always the full modulelist while the crossover chain has p=0 to be sampled. 
        ## (probably no need to modify Boltzmann. simply do it in step function using another for loop. )
        ## this gives a way for sampling in the full modulelist while avoiding sampling the crossover chain. 
        
        for i in range(nchainsCrossover):
            crossoverInd = CrossoverInd[i]
            boltzmann_weight = copy.deepcopy(Boltzmann_pd)                    # make a copy of unnormalized boltzmann pd. 
            boltzmann_weight[crossoverInd] = 0                                # put the crossover entry's weight down to 0. 
            boltzmann_weight = boltzmann_weight/np.sum(boltzmann_weight)      # normalize the weight. 
            sample_anchor_chain = np.random.multinomial(1, boltzmann_weight)
            anchor_chain = int(np.nonzero(sample_anchor_chain)[0])
            Anchor_ind[i] = anchor_chain

        output = dict()
        output["crossoverchains"] = CrossoverInd
        output["anchorchains"] = Anchor_ind
        # print (output)

        return output

    def compute_direction(self, crossoverchain, anchorchain): 
        '''
        this function returns e in snooker crossover algorithm step 3 given a crossover chain and corresponding anchor chain. 
        Input: 
            crossoverchain:        crossover chain. type = nn.module
            anchorchain:           anchor chain. type = nn.module 

        Output:
            snooker crossover direction e.  type = list of tensor, each of the same shape as model.parameters.data
        The output will be used in sampling r. 
        '''

        diff_parameter = []
        norm_diff = 0

        # copy_crossover = copy.deepcopy(crossoverchain)
        # copy_anchor    = copy.deepcopy(anchorchain)
        count = len(list(crossoverchain.parameters()))

        for i in range(count):
            parameter_crossover = list(crossoverchain.parameters())[i]
            parameter_anchor = list(anchorchain.parameters())[i]   # 这样写它们确实会有区别
            temp = parameter_anchor - parameter_crossover 
            diff_parameter = diff_parameter + [temp]
            norm_diff += torch.norm(temp)
            # print("norm of diff in", i, "th parameter:", torch.norm(diff_parameter))
        
        for i in range(count):
            diff_parameter[i] = diff_parameter[i]/norm_diff.item()
        
        return diff_parameter

    def r_SGLD(self, pars, r, model_anchor_modified, temperature, direction):
        '''Step function for sampling r. Given the parameters gradient and direction, return one step updated r. '''
        count = 0
        innerproduct = 0
        dimension = 0
        for parameter in model_anchor_modified.parameters():
            # print("count = ", count)
            # print("parameter #", count," head, before modified:", parameter.data[0])
            
            gradient_parameter = parameter.grad.data 
            # print("count =", count, "head of parameter = ", gradient_parameter[0])
            # print("count =", count, "head of direction = ", direction[count][0])
            temp = gradient_parameter * direction[count]
            # print("count =", count, "elementwise product of gradient and direction = ", temp[0])
            innerproduct += temp.sum()
            # print("count =", count, "inner product of gradient and direction (cumulated) = ", innerproduct )
            size = list(parameter.size())
            # print("torch.size at count ", count, "=", size )
            dimension += np.prod(size)
            

            # print("parameter #", count," head, after modified:", parameter.data[0])
            count += 1

        gradient_r = - (dimension - 1)/r + innerproduct/temperature         # gradient_r = nabla( -log f(r))
        # print("gradient of r = ", gradient_r, "parameter dimension d = ", dimension)
        if abs(r) < pars.lrR:
            lr = abs(r)
        else:
            lr = pars.lrR
        r = r - gradient_r*lr + np.sqrt(2*lr)*np.random.normal()

        return r



    def step(self, pars, modulelist, Boltzmann_pd, train_loader):

        if pars.crossoverRate == 0:
            crossover_idx = []
        else:

            ''' obtain indices of chain pairs for snooker crossover. '''
            chainpairs    = self.sample_chainpairs(pars, Boltzmann_pd)
            crossover_idx = chainpairs["crossoverchains"]              # chains to go through crossover
            anchor_idx    = chainpairs["anchorchains"]                 # their corresponding anchor chains
            length        = len(crossover_idx)
            
            # crossover_idx = [2,3]             # chains to go through crossover
            # anchor_idx    = [0,1]                # their corresponding anchor chains
            # length        = len(crossover_idx)

            ''' compute the offspring of each crossover chain'''
            for i in range(length):
                model_crossover = modulelist[crossover_idx[i]]
                # print("crossover chain = #", crossover_idx[i])
                model_anchor = modulelist[anchor_idx[i]]
                direction = self.compute_direction(model_crossover, model_anchor)     # compute e
                temperature = pars.T / (pars.Tgap ** anchor_idx[i])                   # record temperature

                ''' SGLD to obtain r. '''
                r = np.random.normal(scale=10)                   # initialize r with N(0,10^2)
                # result = [r]
                for t in range(pars.iterationR):
                    # print("iteration # ", t, "for sampling r. ")
                    for j, (images, labels) in enumerate(train_loader):

                        model_cp = copy.deepcopy(model_anchor)
                        count = 0
                        for parameter in model_cp.parameters():
                            parameter.data.add_(r * direction[count])
                            # print("parameter.data.gradient:", parameter.grad.data[0])
                            # parameter.grad.data.zero_()        # zero out the gradients. 
                            # print("parameter #", count," head, after modified:", parameter.data[0])
                            count += 1

                        # first change the data type from 'tensor' to 'Variables'. 
                        images = Variable(images).cuda() if torch.cuda.is_available() else Variable(images)
                        labels = Variable(labels).cuda() if torch.cuda.is_available() else Variable(labels)
                        model_cp.zero_grad()
                        # then evaluate optimizer, calculate loss and back propagate. 
                        output = model_cp(images)
                        loss   = self.criterion(output, labels) * pars.total
                        loss.backward()    # 这样就在parameter.grad 上累加 gradient了
                        r = self.r_SGLD(pars, r, model_cp, temperature, direction)
                        # print("r at batch ", j, "=", r)
                ''' update parameter of crossover model using direction and r. '''
                # print("crossover chain = #", crossover_idx[i], "r = ", r)
                count = 0
                # for parameter in model_crossover.parameters():
                #     print("before crossover, parameter 0 head =", parameter.data[0])
                #     break
                for parameter in model_crossover.parameters():
                    # print("before crossover, parameter #", count, "head =", parameter.data[0])
                    parameter.data.add_(r * direction[count])
                    # print("after crossover, parameter #", count, "head =", parameter.data[0])
                    count += 1
                # for parameter in model_crossover.parameters():
                #     print("after crossover, parameter 0 head =", parameter.data[0])
                #     break
                


                
                # scalers = np.arange(-40,40,1)
                # for j in scalers:
                #     if j == 0:
                #         pass
                #     else:
                #         gradient_r = self.scaler_gradient(j, model_anchor, temperature, direction, train_loader)
                #         print("scaler r = {} it's gradient {:0.4f} anchor chain {} " .format( j, gradient_r, anchor_idx[i]))
            
        return crossover_idx
            # print(i, "th direction head =", direction[0][0], ", direction length = ", len(direction))
            # print("snooker between chain", crossover_idx[i], "as crossover and", anchor_idx[i], "as anchor. ")
            # print("anchor vs. anchor + r*e:")
            
                
        
        # model_crossover = modulelist[0]
        # model_anchor = modulelist[1]
        # direction = self.compute_direction(model_crossover, model_anchor)




        # return direction

    

''' main '''
## define an object "parser" to add attributes to it. 
parser = argparse.ArgumentParser()    

## mcmc properties:
parser.add_argument('-nchains', default=10, type=int, help='Total number of chains')     ## how many chains are going to run. 
parser.add_argument('-epochs', default=1000, type=int, help='Sampling Epochs')       ## how many epochs (something like iterations) are going to run. 
parser.add_argument('-selectionT', default=1e5, type=float, help='Selection temperature for snooker crossover')      ## used in calculating Boltzmann distribution in snooker crossover. 
parser.add_argument('-crossoverRate', default=0.1, type=float, help='Rate of the chains for doing crossover')      ## how many chains in the population are going to go through snooker crossover. 
parser.add_argument('-T', default=0.005, type=float, help='Temperature for highest temperature chain')      ## highest temperature of chains. 
parser.add_argument('-Tgap', default=0.5, type=float, help='Temperature gap between chains')       ## the rest of the chains have temperatures = T/Tgap. 
parser.add_argument('-eta', default=2e-6, type=float, help='squared Sampling learning rate')     ## learning_rate^2/mass for highest temperatur chain. see SGHMC paper page 6
parser.add_argument('-etagap', default=0.66, type=float, help='gap between different chains etas. ')    ## eta of the second high temperature chain = eta/etagap for highest temperatur chain
parser.add_argument('-period', default=2, type=int, help='estimate adaptive variance every [period] epochs')
parser.add_argument('-cycle', default=1, type=int, help='Number of cycles')
parser.add_argument('-burn', default=0.6, type=float, help='burn in iterations for sampling (sn * burn)')
parser.add_argument('-etaanneal', default=0.984, type=float, help='lr annealing factor')
parser.add_argument('-Tanneal', default=1.02, type=float, help='temperature annealing factor')
parser.add_argument('-wdecay', default=25, type=float, help='Sampling weight decay')
parser.add_argument('-iterationR', default=1, type=int, help='number of iteration for SGLD in snooker scaler r')      ## how many iterations are going to run for SGLD sampling r in snooker. 
parser.add_argument('-lrR', default=1e-4, type=float, help='learning rate of SGLD in snooker scaler r')      ## max learning rate for SGLD sampling r in snooker. If r is close to 0 then choose a smaller learning rate.  
parser.add_argument('-bias_F', default=1.5e5, type=float, help='correction factor F')

## dataset properties: 
parser.add_argument('-data', default='cifar100', help='CIFAR10/ CIFAR100')  ## data to be used (adds an attribute '-data', and names it as 'data'
parser.add_argument('-batch', default=256, type=int, help='Batch size')   ## batch size for both training and testing
parser.add_argument('-total', default=50000, type=int, help='Total data points')

## neural network properties:
parser.add_argument('-model', default='resnet20', help='which resnet model to use. In local package "resnet.py." ')

## environmental properties: 
# parser.add_argument('-seed', default=random.randint(1, 1e6), type=int, help='Random Seed')
parser.add_argument('-seed', default=50, type=int, help='Random Seed')

## add the arguments. 
pars = parser.parse_args()

torch.manual_seed(pars.seed)           # set seed for torch
torch.cuda.manual_seed_all(pars.seed)      # set seed for all GPUs. 
np.random.seed(pars.seed)              # set seed for numpy. !!It returns the random numbers with exactly the same order.          
random.seed(pars.seed)                 # set seed in general. 
torch.backends.cudnn.deterministic=True # make reproducible.


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
for i in range(pars.nchains-1):
    model_add = copy.deepcopy(model)
    multichain_model.extend([model_add])

train_loader, test_loader = LoadData(pars.batch, pars.batch, pars)

criterion = nn.CrossEntropyLoss()

accuracy = [BayesEval()]
for i in range(pars.nchains - 1):
    accuracy = accuracy + [BayesEval()]

''''''
''' pretrain: run SGHMC '''
for t in range(15):

    print("epochs = ",t)      # 要注释掉
    training_parallel(pars, multichain_model, range(pars.nchains), train_loader, criterion)
    ''' print accuracy '''
    for i in range(pars.nchains):
        accuracy[i].eval(multichain_model[i], test_loader, cumulate = False)
        print('Chain {} Acc: {:0.2f}'\
                .format(i, accuracy[i].OneEpoch_acc))

''' a few steps of snooker '''
''' part of chains go through one step snooker'''
crossoverRate = copy.deepcopy(pars.crossoverRate)
for t in range(50):
    print("epochs = ",t+15)

    for model in multichain_model:
        model.train()

    Boltzmann_pd = [0]*len(multichain_model)                # unnormalized Boltzmann pd for sampling anchor chain

    for i, (images, labels) in enumerate(train_loader):

        if i % 5 != 0:
            pars.crossoverRate = 0
        else:
            pars.crossoverRate = crossoverRate

        # first change the data type from 'tensor' to 'Variables'. 
        images = Variable(images).cuda() if torch.cuda.is_available() else Variable(images)
        labels = Variable(labels).cuda() if torch.cuda.is_available() else Variable(labels)
        Loss   = []                                 # record loss for every model. 
        for j in range(len(multichain_model)):
            model = multichain_model[j]
            model.zero_grad()
            # then evaluate optimizer, calculate loss and back propagate. 
            output = model(images)
            temp   = criterion(output, labels) * pars.total
            Loss   += [temp]
            if pars.crossoverRate != 0:
                Boltzmann_pd[j] = np.exp(-temp.item()/pars.selectionT)
            # print(len(loss))

        Snooker_test = snooker(train_loader, criterion = nn.CrossEntropyLoss(), selection_t = pars.selectionT)
        crossover_idx = Snooker_test.step(pars, multichain_model, Boltzmann_pd, train_loader)
        # print("crossover idx =", crossover_idx)

        ''' index of chains to go through SGHMC '''
        SGHMC_idx = list(range(pars.nchains))
        for i in crossover_idx:
            SGHMC_idx.remove(i)


        ''' the other part go through SGHMC '''
        for idx in SGHMC_idx:
            eta         = pars.eta / (pars.etagap ** idx)  # different chains occupy different eta
            temperature = pars.T / (pars.Tgap ** idx)    # different chains occupy different temperature
            model = multichain_model[idx]
            loss = Loss[idx]
        
        # take a net from the ModuleList: multichain_model, and put it into the optimizer. 
        # optimizer = optim.SGD(net.parameters(), lr=eta, momentum=0.9)        # no ladder here. need to modify later. 

            optimizer = SGHMC(model.parameters(), eta=eta, temperature=temperature, wdecay=pars.wdecay)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    ''' print accuracy '''
    for i in range(pars.nchains):
        accuracy[i].eval(multichain_model[i], test_loader, cumulate = False)
        print('Chain {} Acc: {:0.2f}'\
                .format(i, accuracy[i].OneEpoch_acc))

# ''' run SGHMC '''
# start = time.time()
# for t in range(15):

#     print("epochs = ",t)      # 要注释掉
#     training_parallel(pars, multichain_model, range(pars.nchains), train_loader, criterion)
#     ''' print accuracy '''
#     for i in range(pars.nchains):
#         accuracy[i].eval(multichain_model[i], test_loader, cumulate = False)
#         print('Chain {} Acc: {:0.2f}'\
#                 .format(i, accuracy[i].OneEpoch_acc))
# # ''' print last accuracy again '''
# # print("print again acc for the last epoch of SGHMC")
# # for i in range(pars.nchains):
# #     accuracy[i].eval(multichain_model[i], test_loader, cumulate = False)
# #     print('Chain {} Acc: {:0.2f}'\
# #             .format(i, accuracy[i].OneEpoch_acc))
        
# # chain0afterSGHMC = copy.deepcopy(multichain_model[0])
# # parameter_chain0afterSGHMC = list(chain0afterSGHMC.parameters())

# end = time.time()
# print("time for 15 SGHMC = ", end - start)

# # '''print parameter head'''
# # for i in range(pars.nchains):
# #     for parameter in multichain_model[i].parameters():
# #         print("after 15 iteration of SGHMC, parameter head of chain",i,":", parameter.data[0])
# #         break

# for t in range(50):
#     print("epochs = ",t+15)

#     ''' snooker'''
#     for model in multichain_model:
#         model.train()
#     Snooker_test = snooker(train_loader, criterion = nn.CrossEntropyLoss(), selection_t = pars.selectionT)   # selection_t = 100000 is a good choice for next to burn-in period. 
#     # direction = Snooker_test.compute_direction(multichain_model[0],multichain_model[1])

#     start = time.time()
#     crossover_idx = Snooker_test.step(pars, multichain_model, train_loader)
#     # print("SGLD chain of r: ", r_chain)
#     end = time.time()
#     print("time for 1 step snooker :", end - start)

#     SGHMC_idx = list(range(pars.nchains))
#     for i in crossover_idx:
#         SGHMC_idx.remove(i)

#     '''print accuracy'''
#     print("snooker step:")
#     for i in range(pars.nchains):
#         print("evaluating model #", i)
#         accuracy[i].eval(multichain_model[i], test_loader, cumulate = False)
#         print('Chain {} Acc: {:0.2f}'\
#                     .format(i, accuracy[i].OneEpoch_acc))

#     ''' SGHMC '''
#     training_parallel(pars, multichain_model, SGHMC_idx, train_loader, criterion)
#     ''' print accuracy '''
#     for i in range(pars.nchains):
#         accuracy[i].eval(multichain_model[i], test_loader, cumulate = True)
#         print('Chain {} Acc: {:0.2f}'\
#                 .format(i, accuracy[i].best_OneEpoch_acc))



# chain0aftersnooker = copy.deepcopy(multichain_model[0])
# parameter_chain0aftersnooker = list(chain0aftersnooker.parameters())

# print("see if parameters of chain 0 is the same before and after snooker. If the same then output = tensor(1):")
# for i in range(len(parameter_chain0aftersnooker)):
#     ifequal = parameter_chain0aftersnooker[i] == parameter_chain0afterSGHMC[i]
#     print(torch.prod(ifequal))




'''print parameter head'''
# for i in range(pars.nchains):
#     for parameter in multichain_model[i].parameters():
#         print("parameter head of chain",i,":", parameter.data[0])
#         break



# gradient_r = Snooker_test.sample_r(0.1, multichain_model[0], pars.T, direction, train_loader)

# boltzmann_pd = Boltzmann(0.1, multichain_model, train_loader)
# print('output', chainpairs)
# print('output data type:', type(boltzmann_pd))

''' add a phase of SGHMC '''

for t in range(3):

    print("epochs = ",t+65)      # 要注释掉
    training_parallel(pars, multichain_model, SGHMC_idx, train_loader, criterion)
    ''' print accuracy '''
    for i in range(pars.nchains):
        accuracy[i].eval(multichain_model[i], test_loader, cumulate = False)
        print('Chain {} Acc: {:0.2f}'\
                .format(i, accuracy[i].OneEpoch_acc))
