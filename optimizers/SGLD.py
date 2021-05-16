'''
The files are created by Yushu Huang for the EMC-SGLD algorithm. YEAR 2021. 
They aim to apply the algorithm to the training of resnet under CIFAR dataset. 
codes reference: https://pysgmcmc.readthedocs.io/en/pytorch/_modules/pysgmcmc/optimizers/sgld.html

In this file: a class -- SGLD is created as the optimizer for the network. 
use 'from optimizers.SGLD import SGLD' to import this file. 
'''

import torch
from torch.optim import Optimizer

class SGLD(Optimizer):
    '''Implements stochastic gradient Langevin dynamic.

    Args:
        params (iterable)  : iterable of parameters to optimize or dicts defining parameter groups. eg. model.parameters()
        lr (float)         : learning rate
        temperature        : temperature

    '''
    def __init__(
        self,
        params, 
        lr = 0.1, 
        temperature = 1 
        ) -> None:
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if temperature < 0:
            raise ValueError("Negative temperature: {}".format(temperature))
        defaults = dict(
            lr=lr, 
            temperature = temperature
        )
        super().__init__(params, defaults)


    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:      # group is a dictionary. group.keys() are the arguments defined in __init__. It contains the parameters of the full model. 
            # If running one model a time, then there is only one group in self.param_group, taking everything.
            for parameter in group["params"]:      # parameter is a torch.nn.parameter.Parameter. It has two arguments: .data, which gives a tensor, and .requires_grad, default = True. 
                if parameter.grad is None:       # parameter.grad will be created by loss.bakward()
                    continue

                state = self.state[parameter]    # state is an empty list before running optimization. Can use this list to save the states of the optimization process. 
                lr = group["lr"]
                temperature = group["temperature"]
                gradient = parameter.grad.data

                #  State initialization 

                if len(state) == 0:
                    state["iteration"] = 0

                state["iteration"] += 1      # record the iteration

                drift = - 0.5 * gradient * lr
                sigma = torch.sqrt(torch.tensor(lr * temperature))
                diffusion = torch.normal(
                        mean=torch.zeros_like(gradient),
                        std=torch.ones_like(gradient)
                    ) * sigma

                update = drift + diffusion
                parameter.data.add_(update)

        return(loss)

