'''
The files are created by Yushu Huang for the EMC-SGLD algorithm. YEAR 2021. 
They aim to apply the algorithm to the training of resnet under CIFAR dataset. 

In this file: LoadData function for data loading
use 'from LoadData import LoadData' to import the LoadData function. 
'''

from torch import Tensor

import torchvision.transforms as transforms

import torch.utils.data as data
import torchvision.datasets as datasets

def LoadData(train_size, test_size, pars) -> Tensor:

    '''
    # input: 
           train_size :       size of the training set
           test_size  :       size of the testing set
           pars       :       pars defined in 'main.py'. need pars.data here. 
    
    # output: 
            train_loader, test_loader:   they are augmented data that are iterable by enumerate(), created by data.DataLoader. 
    '''

    # dataset to use: 
    if pars.data == 'cifar10':
        dataset = datasets.CIFAR10
    elif pars.data == 'cifar100':
        dataset = datasets.CIFAR100
    else:
        exit('Unknown dataset')

    # define the transformation to use for training set. 
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p = 0.5, scale = (0.02, 0.33), ratio = (0.3, 3.3), )])
    
    # define the transformation to use for testing set. 
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    # define the training set and testing set. 
    train_set = dataset('./data/' + pars.data.upper(), train=True, download=True, transform=transform_train)
    test_set = dataset(root='./data/' + pars.data.upper(), train=False, download=False, transform=transform_test)    # pars.data.upper(): return in upper-case form. 
    
    # augment the training and testing set, using data.DataLoader(). It provides an iterable that could be visited by enumerate()
    # caution: no shuffle on test dataset
    train_loader = data.DataLoader(train_set, batch_size=train_size, shuffle=True, num_workers=0) # num_workers=0 is crucial for seed
    test_loader = data.DataLoader(test_set, batch_size=test_size, shuffle=False, num_workers=0)


    return train_loader, test_loader