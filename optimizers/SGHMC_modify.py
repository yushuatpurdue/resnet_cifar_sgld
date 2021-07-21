"""
Note that in Bayesian settings, the lr 2e-6 and weight decay 25 are equivalent to lr 0.1 and weight decay 5e-4 in standard setups.
"""

import sys, copy
import numpy as np
import torch
import random
from torch.autograd import Variable

class SGHMC_modify(optimizer):
    def __init__(self, net, criterion, momentum=0.9, lr=0.1, wdecay=5e-4, T=0.05, total=50000):
        self.net = net
        self.eta = lr
        self.momentum = momentum
        self.T = T
        self.wdecay = wdecay
        self.V = 0.1
        self.velocity = []
        self.criterion = criterion
        self.total = total

        self.beta = 0.5 * self.V * self.eta
        self.alpha = 1 - self.momentum
        
        if self.beta > self.alpha:
            sys.exit('Momentum is too large')
        
        self.sigma = np.sqrt(2.0 * self.eta * (self.alpha - self.beta))
        self.scale = self.sigma * np.sqrt(self.T)

        for param in net.parameters():
            p = torch.zeros_like(param.data)
            self.velocity.append(p)


    def step(self, x, y, closure = None):
        loss = None

        if closure is not None:
            loss = closure()

        for i, param in enumerate(self.net.parameters()):
            proposal = torch.cuda.FloatTensor(param.data.size()).normal_().mul(self.scale)
            grads = param.grad.data
            if self.wdecay != 0:
                grads.add_(self.wdecay, param.data)
            self.velocity[i].mul_(self.momentum).add_(-self.eta, grads).add_(proposal)
            param.data.add_(self.velocity[i])
        return loss
