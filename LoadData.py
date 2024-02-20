'''
The files are created by Yushu Huang for the EMC-SGLD algorithm. YEAR 2021. 
They aim to apply the algorithm to the training of resnet under CIFAR dataset. 

In this file: LoadData function for data loading
use 'from LoadData import LoadData' to import the LoadData function. 
'''
import numpy as np
import pandas as pd
import os
from os import path
import urllib.request
import zipfile

import torch
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

def LoadUCI(data_name, cross_validate_index, train_size, test_size, seed = 1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not path.exists('./data/'+"UCI"):
        os.mkdir('./data/'+"UCI")
    if data_name == "Boston":
        if not path.exists('./data/UCI/housing.data'):
            urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', './data/UCI/housing.data')
        # temp = np.loadtxt('./data/Boston/housing.data')
        temp = np.loadtxt('./data/UCI/housing.data')
        x_data = temp[:, 0:-1]
        y_data = temp[:, -1].reshape([temp.shape[0], 1])
    elif data_name == 'Concrete':
        if not path.exists('./data/UCI/Concrete_Data.xls'):
            urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls', './data/UCI/concrete.data')
        temp = pd.read_excel('./data/UCI/Concrete_Data.xls')
        temp = np.mat(temp)
        x_data = temp[:, :8]
        y_data = temp[:, 8]

    elif data_name == 'Energy':
        if not path.exists('./data/UCI/ENB2012_data.xlsx'):
            urllib.request.urlretrieve('http://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx', './data/UCI/ENB2012_data.xlsx')
        temp = pd.read_excel('./data/UCI/ENB2012_data.xlsx')
        temp = np.mat(temp)
        x_data = temp[:768, :8]
        y_data = temp[:768, 9]

    elif data_name == 'Wine':
        if not path.exists('./data/UCI/winequality-red.csv'):
            urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', './data/UCI/winequality-red.csv')
        temp = pd.read_csv('./data/UCI/winequality-red.csv', sep=';')
        temp = np.mat(temp)
        x_data = temp[:, 0:11]
        y_data = temp[:, 11]

    elif data_name == 'Yacht':
        if not path.exists('./data/UCI/yacht_hydrodynamics.data'):
            urllib.request.urlretrieve('http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data', './data/UCI/yacht_hydrodynamics.data')
        temp = np.loadtxt('./data/UCI/yacht_hydrodynamics.data')
        x_data = temp[:, 0:6]
        y_data = temp[:, 6].reshape([temp.shape[0], 1])
    
    elif data_name == 'kin8nm':
        if not path.exists('./data/UCI/kin8nm.csv'):
            urllib.request.urlretrieve('https://www.openml.org/data/get_csv/3626/dataset_2175_kin8nm.arff', './data/UCI/kin8nm.csv')
        temp = pd.read_csv('./data/UCI/kin8nm.csv')

        temp = np.mat(temp)
        x_data = temp[:, 0:8]
        y_data = temp[:, 8]
    elif data_name == 'CCPP':
        if not path.exists('./data/UCI/CCPP/Folds5x2_pp.xlsx'):
            urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip', './data/UCI/CCPP.zip')
            with zipfile.ZipFile('./data/UCI/CCPP.zip', 'r') as zip_ref:
                zip_ref.extractall('./data/UCI/')
        temp = pd.read_excel('./data/UCI/CCPP/Folds5x2_pp.xlsx')
        temp = np.mat(temp)
        x_data = temp[:, :4]
        y_data = temp[:, 4]

    elif data_name == 'Naval':
        if not path.exists('./data/UCI/UCI CBM Dataset/data.txt'):
            urllib.request.urlretrieve('https://archive.ics.uci.edu/static/public/316/condition+based+maintenance+of+naval+propulsion+plants.zip', './data/UCI/Naval.zip')
            with zipfile.ZipFile('./data/UCI/Naval.zip', 'r') as zip_ref:
                zip_ref.extractall('./data/UCI/')
        temp = np.loadtxt('./data/UCI/UCI CBM Dataset/data.txt')
        x_data = temp[:, 0:16]
        y_data = temp[:, 16].reshape([temp.shape[0], 1])

    elif data_name == 'Protein':
        if not path.exists('./data/UCI/CASP.csv'):
            urllib.request.urlretrieve('https://archive.ics.uci.edu/static/public/265/physicochemical+properties+of+protein+tertiary+structure.zip', './data/UCI/protein.zip')
            with zipfile.ZipFile('./data/UCI/protein.zip', 'r') as zip_ref:
                zip_ref.extractall('./data/UCI/')
        temp = pd.read_csv(',.data/UCI/CASP.csv')
        temp = np.matrix(temp)
        x_data = temp[:, 1:10]
        y_data = temp[:, 0]
        
    elif data_name == 'Year':
        if not path.exists('./data/UCI/YearPredictionMSD.txt'):
            urllib.request.urlretrieve('https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip', './data/UCI/yearpredictionmsd.zip')
            with zipfile.ZipFile('./data/UCI/yearpredictionmsd.zip', 'r') as zip_ref:
                zip_ref.extractall('./data/UCI/')
        temp = pd.read_table('./data/UCI/YearPredictionMSD.txt', header = None, sep = ',')

        temp = np.matrix(temp)
        x_data = temp[:, 1:]
        y_data = temp[:, 0]


    permutation = np.random.choice(range(x_data.shape[0]), x_data.shape[0], replace=False)
    size_test = np.round(x_data.shape[0] * 0.1).astype(int)
    divid_index = np.arange(x_data.shape[0])

    lower_bound = cross_validate_index * size_test
    upper_bound = (cross_validate_index + 1) * size_test
    test_index = (divid_index >= lower_bound) * (divid_index < upper_bound)

    index_train = permutation[[not _ for _ in test_index]]
    index_test = permutation[test_index]

    x_train = x_data[index_train, :]
    y_train = y_data[index_train]

    x_test = x_data[index_test, :]
    y_test = y_data[index_test]

    x_train_std = np.std(x_train, 0)
    x_train_std[x_train_std == 0] = 1
    x_train_mean = np.mean(x_train, 0)

    x_train = (x_train - np.full(x_train.shape, x_train_mean)) / np.full(x_train.shape, x_train_std)

    x_test = (x_test - np.full(x_test.shape, x_train_mean)) / np.full(x_test.shape, x_train_std)

    x_train = torch.FloatTensor(x_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    x_test = torch.FloatTensor(x_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)

    train_set = data.TensorDataset(x_train, y_train)

    test_set = data.TensorDataset(x_train, y_train)

    train_loader = data.DataLoader(train_set, batch_size=train_size, shuffle=True, num_workers=0) # num_workers=0 is crucial for seed
    test_loader = data.DataLoader(test_set, batch_size=test_size, shuffle=False, num_workers=0)

    return train_loader, test_loader

        # x_train = torch.FloatTensor(x_train).to(device)
        # y_train = torch.FloatTensor(y_train).to(device)
        # x_test = torch.FloatTensor(x_test).to(device)
        # y_test = torch.FloatTensor(y_test).to(device)