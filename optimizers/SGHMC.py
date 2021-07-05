'''
The files are created by Yushu Huang for the EMC-SGLD algorithm. YEAR 2021. 
They aim to apply the algorithm to the training of resnet under CIFAR dataset. 
codes reference: https://pysgmcmc.readthedocs.io/en/pytorch/_modules/pysgmcmc/optimizers/sghmc.html
algorithm based on paper: Stochastic Gradient Hamilton Monte Carlo (Chen et al). 

In this file: a class -- SGHMC is created as the optimizer for the network. 
use 'from optimizers.SHMC import SGLD' to import this file. 
'''

import torch
from torch.optim import Optimizer, optimizer

class SGHMC(Optimizer):
    '''Implements stochastic gradient Hamilton Monte Carlo algorithm for one step update.

    Args:
        params (iterable)  : iterable of parameters to optimize or dicts defining parameter groups. eg. model.parameters()
        lr (float)         : learning rate
        temperature        : temperature
        friction           : friction (denoted as C in SGHMC paper)

    '''
    def __init__(
        self,
        params, 
        lr = 0.1, 
        temperature = 1, 
        friction = 0.05
        ) -> None:
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if temperature < 0:
            raise ValueError("Negative temperature: {}".format(temperature))
        defaults = dict(
            lr=lr, 
            temperature = temperature, 
            friction = friction
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        ''' one step forward. '''
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:      # group is a dictionary. group.keys() are the arguments defined in __init__. It contains the parameters of the full model. 
            # If running one model a time, then there is only one group in self.param_group, taking everything.
            for parameter in group["params"]:      # parameter is a torch.nn.parameter.Parameter. It has two arguments: .data, which gives a tensor, and .requires_grad, default = True. 
                if parameter.grad is None:       # parameter.grad will be created by loss.bakward()
                    continue

                state = self.state[parameter]    # state is an empty list before running optimization. Can use this list to save the states of the optimization process. 
                
                #  State initialization 

                if len(state) == 0:
                    state["iteration"] = 0
                    state["momentum"] = torch.zeros_like(parameter)

                state["iteration"] += 1      # record the iteration

                lr = group["lr"]
                temperature = group["temperature"]
                friction = group["friction"]
                momentum = state["momentum"]
                gradient = parameter.grad.data

                sigma = torch.sqrt(torch.tensor(2*friction*lr*temperature))
                diffusion = torch.normal(
                        mean=torch.zeros_like(gradient),
                        std=torch.ones_like(gradient)
                    ) * sigma

                momentum = momentum.mul_(1. - lr*friction) - lr * gradient + diffusion
                parameter.data.add_(momentum*lr) 

        return loss