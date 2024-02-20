'''
This file is created by Yushu Huang for the ESGLD algorithm. YEAR 2024. 
Aims to apply the algorithm to the training of single hidden layer NN with UCI dataset. 

In this file: a single hidden layer NN is created as the model for training. 
use 'from network.NN import NN' to import CNN model in this file. 
'''
import torch
import torch.nn as nn

class NN(nn.Module):
    def __init__(self, num_hidden, hidden_dim, input_dim, output_dim, device):
        super(NN, self).__init__()
        self.num_hidden = num_hidden

        self.fc = nn.Linear(input_dim, hidden_dim[0], device=device)
        self.fc_list = []

        for i in range(num_hidden - 1):
            self.fc_list.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1], device=device))
            self.add_module('fc' + str(i + 2), self.fc_list[-1])
        self.fc_list.append(nn.Linear(hidden_dim[-1], output_dim, device=device))
        self.add_module('fc' + str(num_hidden + 1), self.fc_list[-1])

    def forward(self, x):
        x = torch.tanh(self.fc(x))
        for i in range(self.num_hidden - 1):
            x = torch.tanh(self.fc_list[i](x))
        x = self.fc_list[-1](x)
        return x