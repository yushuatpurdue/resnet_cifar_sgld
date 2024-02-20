'''
The files are created by Yushu Huang for the EMC-SGLD algorithm. YEAR 2021. 
They aim to apply the algorithm to the training of resnet under CIFAR dataset. 
reference: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

In this file: a class -- resnets are created as the model for training. 
use 'import network.resnet' to import this file. 
use, eg., 'network.resnet.resnet20' to call funtion 'resnet20' as the class of resnet model depth 20. 
'''

from __future__ import absolute_import     # import the most recent version of packages

import torch.nn as nn
import torch
import math

## define attributes to be publicly imported
__all__ = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']       # this line means that the only thing that would got publicly imported if we ran "import resnet" will be resnet (in the end of the file). We also defined a lot of other modules but they are all hiddenly working and are not expected to be used explicitly. 


## define the convolution layers. 
def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:     
    """3x3 convolution with padding. used as the convolution in BasicBlock"""
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride, padding = 1, bias = False)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution, used as one of the convolutions in BottleneckBlock"""
    return nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride = stride, bias = False)


## define the residual block, for resnets not too deep. 
class BasicBlock(nn.Module):
    '''the residual block for depth < 44. '''

    expansion = 1        # a parameter that will be called in constructing class ResNet

    def __init__(
        self, 
        inplanes: int, 
        planes: int, 
        stride: int = 1, 
        downsample = None
        ) -> None:
        super(BasicBlock, self).__init__()       ## super solves the __init__ conflict in BasicBlock and its inheritants froms nn.Module

        ## define the functions to be used in one step forward: 
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    ## one step forward: 
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


## define the residual block for deep resnets. 
class Bottleneck(nn.Module):
    '''
    # This Bottleneck uses the same structure as in torchvision,
    # It places the stride for downsampling at 3x3 convolution(self.conv2),
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    '''

    expansion: int = 4        # a parameter that will be called in constructing class ResNet

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        ) -> None:
        super(Bottleneck, self).__init__()
        
        ## define the functions to be used in one step forward: 
        ## Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    ## one step forward: 
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        layers,
        depth: int,
        num_classes: int = 1000
    ) -> None:
        super(ResNet, self).__init__()
        
        block = Bottleneck if depth >=44 else BasicBlock     ## inline if... else
        self.inplanes = 16

        ## define the functions to be used in one step forward: 
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride = 2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride = 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():           ## self.modules were defined above. They are the operators like relu, conv, batchnorm, etc. 
            if isinstance(m, nn.Conv2d):    ## if the self.module is an instance of Conv2d. 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride = stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


## define the resnets to be used: 
def resnet20(**kwargs):         ## **kwargs: any other arguments in ResNet. 
    return ResNet(layers = [3,3,3], depth = 20, **kwargs)

def resnet32(**kwargs):
    return ResNet([5,5,5],32, **kwargs)
    
def resnet44(**kwargs):    
    return ResNet([7, 7, 7], 44, **kwargs)

def resnet56(**kwargs):
    return ResNet([9, 9, 9], 56, **kwargs)

def resnet110(**kwargs):
    return ResNet([18, 18, 18], 110, **kwargs)

def resnet1202(**kwargs):
    return ResNet([200, 200, 200], 1202, **kwargs)

    
    