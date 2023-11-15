'''
The files are created by Yushu Huang for the EMC-SGLD algorithm. YEAR 2021. 
They aim to apply the algorithm to the training of resnet under CIFAR dataset. 
algorithm reference: Jun Liu: Monte Carlo Strategies in Scientific Computing, p232. 
In this file: a class -- SGHMC is created as the optimizer for the network. 
use 'from optimizers.snooker import snooker' to import class in this file. 
'''

import sys
import math
import copy
import numpy as np
from numpy.core.numeric import NaN
from numpy.lib.function_base import gradient
import random
from numpy.lib.type_check import nan_to_num
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.autograd import Variable

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
        self.criterion    = criterion
        self.train_loader = train_loader
        self.selection_t  = selection_t
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
        for i in range(nchainsCrossover):
            crossoverInd = CrossoverInd[i]
            boltzmann_weight = copy.deepcopy(Boltzmann_pd)                    # make a copy of unnormalized boltzmann pd. 
            boltzmann_weight[crossoverInd] = 0                                # put the crossover entry's weight down to 0. 
            boltzmann_weight = np.asarray(boltzmann_weight).astype('float64')
            boltzmann_weight = boltzmann_weight/np.sum(boltzmann_weight)      # normalize the weight. 
            # print("boltzmann_weight = ", boltzmann_weight)
            for weight in boltzmann_weight:
                if weight<0:
                    print("weight<0 in", boltzmann_weight)
                if weight>1:
                    print("weight>1 in", boltzmann_weight)
                if weight == NaN:
                    print("NaN weight in", boltzmann_weight)
            sample_anchor_chain = np.random.multinomial(1, boltzmann_weight)
            anchor_chain = int(np.nonzero(sample_anchor_chain)[0])
            Anchor_ind[i] = anchor_chain

        output = dict()
        output["crossoverchains"] = CrossoverInd
        output["anchorchains"] = Anchor_ind

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
        count = len(list(crossoverchain.parameters()))

        for i in range(count):
            parameter_crossover = list(crossoverchain.parameters())[i]
            parameter_anchor = list(anchorchain.parameters())[i]   # 这样写它们确实会有区别
            temp = parameter_anchor - parameter_crossover
            diff_parameter = diff_parameter + [temp]
            norm_diff += torch.norm(temp)
        for i in range(count):
            diff_parameter[i] = diff_parameter[i]/norm_diff.item()
        
        return diff_parameter

    def r_SGLD(self, pars, r, model_anchor_modified, temperature, direction):
        '''
        Step function for sampling r. Given the parameters gradient and direction, return one step updated r. 
        
        Computing gradient of r:
        f(r) propto abs(r)^(d-1) * pi(xj + r*e) is the distribution of r. 
        d  = dimension of xj. 
        pi = target distribution of xj. 
        xj = anchor model parameters here.
        e  = direction from self.compute_direction. 

        gradient(r) = -nabla(log f(r)) -- this is what we are computing in this function. 
        in this setting, 
        gradient(r) = -(d-1)/r + < nabla(H(xj + r*e)), e > / Temperature. 

        Input:
            pars
            r:                          snooker scaler. type = float. 
            model_anchor_modified:      pi(xj + r*e). type = nn.module. 
            temperature:       temperature of the anchor model. type = float. 
            direction:         snooker crossover direction e. type = list of tensor, each of the same shape as model.parameters.data

        Output:
            one step update of r. 
        '''

        ''' compute the gradient. '''
        count = 0
        innerproduct = 0
        dimension = 0
        for parameter in model_anchor_modified.parameters():
            gradient_parameter = parameter.grad.data 
            temp = gradient_parameter * direction[count]
            innerproduct += temp.sum()
            size = list(parameter.size())
            dimension += np.prod(size)
            
            count += 1
        innerproduct = innerproduct.item()

        gradient_r = - (dimension - 1)/r + innerproduct/temperature         # gradient_r = nabla( -log f(r))
        if gradient_r >0:
            gradient_r = min(gradient_r, 1e6)
        elif gradient_r <0:
            gradient_r = max(gradient_r, -1e6)            # gradient clipping. 
        # print("gradient_r = ", gradient_r)

        ''' adaptively decrease learning rate if needed. '''
        if abs(r) < pars.lrR:
            lr = abs(r)
        else:
            lr = pars.lrR

        ''' update r. '''
        r = r - gradient_r*lr + np.sqrt(2*lr)*np.random.normal()*np.sqrt(temperature)

        return r



    def step(self, pars, modulelist, Boltzmann_pd, train_loader):
        if pars.crossoverRate == 0:
            crossover_idx = []
        else:
            '''obtain indices of chain pairs for snooker crossover. '''
            chainpairs    = self.sample_chainpairs(pars, Boltzmann_pd)
            crossover_idx = chainpairs["crossoverchains"]              # chains to go through crossover
            anchor_idx    = chainpairs["anchorchains"]                 # their corresponding anchor chains
            length        = len(crossover_idx)

            for i in range(length):
                model_crossover = modulelist[crossover_idx[i]]
                model_anchor = modulelist[anchor_idx[i]]
                direction = self.compute_direction(model_crossover, model_anchor)
                temperature = pars.T / (pars.Tgap ** anchor_idx[i])
                # print(i, "th direction head =", direction[0][0], ", direction length = ", len(direction))

                ''' SGLD to obtain r. '''
                r = np.random.normal(scale= pars.rInitScale)                   # initialize r with N(0,10^2)
                for t in range(pars.iterationR):                 # epoch
                    # print("iteration # ", t, "for sampling r. ")
                    for j, (images, labels) in enumerate(train_loader):      # batch
                        
                        ''' modify parameter: pi(xj + r*e) '''
                        model_cp = copy.deepcopy(model_anchor)
                        count = 0
                        for parameter in model_cp.parameters():
                            parameter.data.add_(r * direction[count])
                            count += 1

                        ''' batch update r.  '''
                        # first change the data type from 'tensor' to 'Variables'. 
                        images = Variable(images).cuda() if torch.cuda.is_available() else Variable(images)
                        labels = Variable(labels).cuda() if torch.cuda.is_available() else Variable(labels)
                        # then evaluate optimizer, calculate loss and back propagate. 
                        output = model_cp(images)
                        loss   = self.criterion(output, labels) * pars.total
                        model_cp.zero_grad()
                        loss.backward()    # 这样就在parameter.grad 上累加 gradient了
                        r = self.r_SGLD(pars, r, model_cp, temperature, direction)
                        print("r at batch ", j, "=", r)
                # print("r = ", r)
                # if math.isnan(r) == True:
                # try sampling r for another 3 times if r is nan:
                for i in range(3):
                    if math.isnan(r) == True:      # if r is nan then resample r. 
                        r = np.random.normal(scale=pars.rInitScale)                   # initialize r with N(0,10^2)
                        for t in range(pars.iterationR):                 # epoch
                            # print("iteration # ", t, "for sampling r. ")
                            for j, (images, labels) in enumerate(train_loader):      # batch
                                
                                ''' modify parameter: pi(xj + r*e) '''
                                model_cp = copy.deepcopy(model_anchor)
                                count = 0
                                for parameter in model_cp.parameters():
                                    parameter.data.add_(r * direction[count])
                                    count += 1

                                ''' batch update r.  '''
                                # first change the data type from 'tensor' to 'Variables'. 
                                images = Variable(images).cuda() if torch.cuda.is_available() else Variable(images)
                                labels = Variable(labels).cuda() if torch.cuda.is_available() else Variable(labels)
                                # then evaluate optimizer, calculate loss and back propagate. 
                                output = model_cp(images)
                                loss   = self.criterion(output, labels) * pars.total
                                model_cp.zero_grad()
                                loss.backward()    # 这样就在parameter.grad 上累加 gradient了
                                r = self.r_SGLD(pars, r, model_cp, temperature, direction)
                        # print("r = ", r)
                    else: 
                        break



                ''' update parameter of crossover model using direction and r. '''
                count = 0
                # for parameter in model_crossover.parameters():
                #     print("before crossover, parameter 0 head =", parameter.data[0])
                #     break
                for parameter in model_crossover.parameters():
                    # print("before crossover, parameter #", count, "head =", parameter.data[0])
                    parameter.data.add_(r * direction[count])
                    # print("after crossover, parameter #", count, "head =", parameter.data[0])
                    count += 1

        return crossover_idx








