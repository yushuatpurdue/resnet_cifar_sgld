'''
The files are created by Yushu Huang for the EMC-SGLD algorithm. YEAR 2021. 
They aim to apply the algorithm to the training of resnet under CIFAR dataset. 
codes reference: https://pysgmcmc.readthedocs.io/en/pytorch/_modules/pysgmcmc/optimizers/sghmc.html
algorithm reference: Stochastic Gradient Hamilton Monte Carlo (Chen et al), page 6. 

In this file: a class -- SGHMC is created as the optimizer for the network. 
use 'from optimizers.SHMC import SGLD' to import this file. 
'''

import sys
import numpy as np
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
        eta = 0.1, 
        alpha =0.9,
        temperature = 1, 
        wdecay = 5e-4
        ) -> None:
        if eta < 0.0:
            raise ValueError("Invalid learning rate: {}".format(eta))
        if temperature < 0:
            raise ValueError("Negative temperature: {}".format(temperature))
        defaults = dict(
            eta=eta, 
            alpha = alpha,
            temperature = temperature, 
            wdecay = wdecay
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

                state = self.state[parameter]    # state is an empty list before running optimization. Can use this list as the updated states of the optimization. 
                
                #  State initialization 

                if len(state) == 0:
                    state["iteration"] = 0
                    state["lrTimesVelocity"] = torch.zeros_like(parameter)

                state["iteration"] += 1      # record the iteration

                # extract the inputs from initialization by group[" "]
                alpha = group["alpha"]
                eta = group["eta"]
                temperature = group["temperature"]
                lrTimesVelocity = state["lrTimesVelocity"]          # modifying momentum = modifying state["momentum"]
                wdecay = group["wdecay"]
                gradient = parameter.grad.data
                if wdecay != 0:
                    gradient.add_(wdecay, parameter.data)


                B_hat = 0.1
                beta = 0.5* B_hat * eta
                if beta > alpha:
                    sys.exit('Momentum is too large')
                
                sigma = np.sqrt(2.0 * eta * (alpha - beta))
                print("sigma = ", sigma)
                scale = sigma * np.sqrt(temperature)
                '''
                diffusion = torch.normal(
                        mean=torch.zeros_like(gradient),
                        std=torch.ones_like(gradient)
                    ) * scale
                    '''
                diffusion = torch.cuda.FloatTensor(parameter.data.size()).normal_().mul(scale)


                lrTimesVelocity.mul_(1-alpha).add_(-eta, gradient).add_(diffusion)
                parameter.data.add_(lrTimesVelocity) 

        return loss
        