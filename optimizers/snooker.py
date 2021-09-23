'''
The files are created by Yushu Huang for the EMC-SGLD algorithm. YEAR 2021. 
They aim to apply the algorithm to the training of resnet under CIFAR dataset. 
algorithm reference: Jun Liu: Monte Carlo Strategies in Scientific Computing, p232. 
In this file: a class -- SGHMC is created as the optimizer for the network. 
use 'from optimizers.SHMC import SGLD' to import this file. 
'''

import sys
import math
import copy
import numpy as np
from numpy.lib.function_base import gradient
import random
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

    def Boltzmann(self, selection_t, crossoverInd, modulelist, train_loader):
        '''
        Boltzman calculates the Boltzmann density of a model: exp(-loss(model)/selection_t), where loss is the cross entropy loss. 

        It takes each module in modulelist, calculate it's full loss, then calculate the Boltzmann density. 
        '''
        Boltzmann_pd = [0]*len(modulelist)          # Boltzmann_pd records the Boltzmann probability density for each model. It is a list of length = nchains. 
        loss = 0                                    # loss is a temporary record of the loss. 

        for j in range(len(modulelist)):
            net = modulelist[j]
            net.train()

            if j == crossoverInd: 
                Boltzmann_pd[j] = 0
            else:
                for i, (images, labels) in enumerate(train_loader):
                    ''' calculate the full loss of the net. '''

                    # first change the data type from 'tensor' to 'Variables'. 
                    images = Variable(images).cuda() if torch.cuda.is_available() else Variable(images)
                    labels = Variable(labels).cuda() if torch.cuda.is_available() else Variable(labels)

                    # then evaluate optimizer, calculate loss and back propagate. 
                    output = net(images)
                    temp   = self.criterion(output, labels) * len(images)
                    loss   += temp.item()
                
                Boltzmann_pd[j] = np.exp(-loss/selection_t)
            
        Boltzmann_pd = Boltzmann_pd/np.sum(Boltzmann_pd)
        return Boltzmann_pd

    def sample_chainpairs(self, pars, modulelist):
        '''
        This function returns the indices of chains to go through snooker crossover, and their correspondent anchor chains. 
        Input:
            pars:          argument parser. 
            modulelist:    the model population. 
        '''

        if pars.nchains != len(modulelist):
            raise ValueError("length of modulelist does not match number of chains")
        nchainsCrossover = math.ceil(pars.crossoverRate * pars.nchains)           # how many chains are going through crossover
        CrossoverInd = random.sample(range(pars.nchains), nchainsCrossover)       # sample the index of those crossover chains

        Anchor_ind = [pars.nchains] * nchainsCrossover

        ## note: modify Boltzmann function so that the modulelist is always the full modulelist while the crossover chain has p=0 to be sampled. 
        ## (probably no need to modify Boltzmann. simply do it in step function using another for loop. )
        ## this gives a way for sampling in the full modulelist while avoiding sampling the crossover chain. 
        
        for i in range(nchainsCrossover):
            crossoverInd = CrossoverInd[i]
            boltzmann_pd = self.Boltzmann(self.selection_t, crossoverInd, modulelist, self.train_loader)
            sample_anchor_chain = np.random.multinomial(1, boltzmann_pd)
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
            temp = parameter_crossover - parameter_anchor
            diff_parameter = diff_parameter + [temp]
            norm_diff += torch.norm(temp)
            # print("norm of diff in", i, "th parameter:", torch.norm(diff_parameter))
        for i in range(count):
            diff_parameter[i] = diff_parameter[i]/norm_diff.item()
        
        return diff_parameter

    def scaler_gradient(self, r, model_anchor, temperature, direction, train_loader):
        '''
        this function returns gradient for scaler r that will be used for SGLD of r. 

        f(r) propto abs(r)^(d-1) * pi(xj + r*e) is the distribution of r. 
        d  = dimension of xj. 
        pi = target distribution of xj. 
        xj = anchor model parameters here.
        e  = direction from self.compute_direction. 

        gradient(r) = -nabla(log f(r)) -- this is what we are computing in this function. 
        in this setting, 
        gradient(r) = -(d-1)/r + < nabla(H(xj + r*e)), e > / Temperature. 

        Input:
            r:                 snooker scaler. type = float. 
            model_anchor:      anchor model. type = nn.module. 
            temperature:       temperature of the anchor model. type = float. 
            direction:         snooker crossover direction e. type = list of tensor, each of the same shape as model.parameters.data
            train_loader:      augmented training data. 

        Output:
            gradient of r. type = scaler. 
        '''
        model_cp = copy.deepcopy(model_anchor)        # deep copy anchor model to avoid material modify in anchor model's parameters when adding r*direction. 
        # note: deepcopy preserves gradient = TRUE but doesn't copy gradients, so gradient = None after copying. 
        count = 0
        for parameter in model_cp.parameters():       # for every parameter, add to it r*direction. 
            parameter.data.add_(r * direction[count])
            count += 1

        for i, (images, labels) in enumerate(train_loader):
            ''' calculate the full gradient: nabla(H(xj + r*e)) at xj + r*e. '''

            # first change the data type from 'tensor' to 'Variables'. 
            images = Variable(images).cuda() if torch.cuda.is_available() else Variable(images)
            labels = Variable(labels).cuda() if torch.cuda.is_available() else Variable(labels)

            # then evaluate optimizer, calculate loss and back propagate. 
            output = model_cp(images)
            loss   = self.criterion(output, labels) * len(images)
            loss.backward()    # 这样就在parameter.grad 上累加 gradient了

        count = 0
        innerproduct = 0
        dimension = 0
        for parameter in model_cp.parameters():         # compute <nabla(H(xj + r*e)), e>, and compute d. 
            
            gradient_parameter = parameter.grad.data 
            temp = gradient_parameter * direction[count]
            innerproduct += temp.sum()                  # innerproduct = <nabla(H(xj + r*e)), e>. 

            size = list(parameter.size())
            dimension += np.prod(size)                  # dimension = d. 
            
            count += 1

        gradient_r = - (dimension - 1)/r + innerproduct/temperature         # gradient_r = nabla( -log f(r))
        return gradient_r



        


    def step(self, pars, modulelist):
        
        '''obtain indices of chain pairs for snooker crossover. '''
        chainpairs    = self.sample_chainpairs(pars, modulelist)
        crossover_idx = chainpairs["crossoverchains"]              # chains to go through crossover
        anchor_idx    = chainpairs["anchorchains"]                 # their corresponding anchor chains
        length        = len(crossover_idx)

        for i in range(length):
            model_crossover = modulelist[crossover_idx[i]]
            model_anchor = modulelist[anchor_idx[i]]
            direction = self.compute_direction(model_crossover, model_anchor)
            temperature = pars.T / (pars.Tgap ** anchor_idx[i])
            # print(i, "th direction head =", direction[0][0], ", direction length = ", len(direction))

        # 这样return direction 只能return 到最后一个direction. 这是不对的。要写sample_r 然后update parameter，把它们放到for loop 里面。

        # return direction








