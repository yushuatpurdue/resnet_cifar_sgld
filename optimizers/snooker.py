'''
The files are created by Yushu Huang for the EMC-SGLD algorithm. YEAR 2021. 
They aim to apply the algorithm to the training of resnet under CIFAR dataset. 
algorithm reference: Jun Liu: Monte Carlo Strategies in Scientific Computing, p232. 
In this file: a class -- SGHMC is created as the optimizer for the network. 
use 'from optimizers.SHMC import SGLD' to import this file. 
'''

import sys
import math
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
        selection_t = 0.1
        ) -> None:

        if selection_t < 0:
            raise ValueError("Negative selection temperature: {}".format(selection_t))
        super(snooker, self).__init__()

        # self.modulelist   = modulelist
        self.train_loader = train_loader
        self.selection_t  = selection_t
        # self.nchains      = len(modulelist)

    def Boltzmann(self, selection_t, crossoverInd, modulelist, train_loader):
        '''
        Boltzman calculates the Boltzmann density of a model: exp(-loss(model)/selection_t), where loss is the cross entropy loss. 

        It takes each module in modulelist, calculate it's full loss, then calculate the Boltzmann density. 
        '''
        criterion = nn.CrossEntropyLoss()           # cross entropy loss generates the potential. 
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
                    temp   = criterion(output, labels) * len(images)
                    loss   += temp.item()
                
                Boltzmann_pd[j] = np.exp(-loss/selection_t)
            
        Boltzmann_pd = Boltzmann_pd/np.sum(Boltzmann_pd)
        return Boltzmann_pd

    def sample_chainpairs(self, pars, modulelist):

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
            anchor_chain = int(np.nonzero(sample_anchor_chain[0]))
            Anchor_ind[i] = anchor_chain

        output = dict()
        output["crossoverchains"] = CrossoverInd
        output["anchorchains"] = Anchor_ind

        return output



    def step(self, pars, modulelist):
        chainpairs = self.sample_chainpairs(pars, modulelist)

        return chainpairs 






