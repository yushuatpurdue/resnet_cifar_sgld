'''
The files are created by Yushu Huang for the EMC-SGLD algorithm. YEAR 2021. 
They aim to apply the algorithm to the training of resnet under CIFAR dataset. 

In this file: calculating the accuracy for classification. 
'''


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
# from LoadData import LoadData
from optimizers.SGLD import SGLD


class BayesEval:
    def __init__(self):
        self.counter            = 0           # count the epoches
        self.PixelScore         = []          # records the outputs (scores) for each pixel over epoches. eg. datasize = 50000 batchsize = 256, PixelScore has 50000 tensors (partitioned in fashion of 256 per fraction), each has 100 scores (for 100 labels) cumulated over epoches. 
        self.OneEpoch_acc       = 0           # accuracy (percentage) by scores from current epoch
        self.Cumulated_acc      = 0           # accuracy (percentage) by cumulated scores of past epoches
        self.best_OneEpoch_acc  = 0           # historically best accuracy
        self.best_Cumulated_acc = 0           # historically best cumulative accuracy

    def eval(self, net, data_loader, weight=1, cumulate=False):
        # for parameter in net.parameters():
        #     print("in evaluation, parameter 0 head =", parameter.data[0])
        #     break
        net.eval()               # set net to evaluating mode. 
        one_correct, cumulate_correct = 0, 0      # count how many of the pixels are labeled correctly. 
        for cnt, (images, labels) in enumerate(data_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            # images, labels = Variable(images).cpu(), Variable(labels).cpu()
            outputs = net.forward(images).data           # returns a tensor of batchsize*labelsize (eg. 256 pixels per batch, 100 labels, output: tensor of 256*100)
            outputlable = outputs.max(1)[1]         # outputs.max(1): returns for each pixel the highest score and its label. outputs.max(1)[1]: highest scored label for each pixel in one batch, ie. labels by net. 
            matchedlable = outputlable.eq(labels.data)      # decided whether labels by net matches the true labels. 
            one_correct += matchedlable.sum().item()        # matchedlable.sum().item(): how many pixels are labeled correctly. one_correct: cumulate it over all batches within one epoch. 
            
            if cumulate == True:          # if cumulate = False, Cumulated_acc == 0. 
                outputs = outputs * weight
                if self.counter == 0:
                    self.PixelScore.append(outputs)       # score of the 100 labels for each pixel in dataset. 
                else:
                    self.PixelScore[cnt] += outputs       # cumulate this score over epoches. 
                cumulate_correct += self.PixelScore[cnt].max(1)[1].eq(labels.data).sum().item()  # how many pixels are labeled correctly according to the cumulative scores.
        if cumulate == True:
            self.counter += 1
        self.OneEpoch_acc        = 100.0 * one_correct / len(data_loader.dataset)
        self.Cumulated_acc       = 100.0 * cumulate_correct / len(data_loader.dataset)
        self.best_OneEpoch_acc   = max(self.best_OneEpoch_acc, self.OneEpoch_acc)
        self.best_Cumulated_acc  = max(self.best_Cumulated_acc, self.Cumulated_acc)
