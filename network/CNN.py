'''
The files are created by Yushu Huang for the EMC-SGLD algorithm. YEAR 2021. 
They aim to apply the algorithm to the training of resnet under CIFAR dataset. 

In this file: a CNN model is created as the model for training. 
use 'from network.CNN import CNN' to import CNN model in this file. 
'''


import torch.nn as nn
import torch.nn.functional as F
# define the CNN architecture

class CNN(nn.Module):
  def __init__(self,
               num_classes: int = 1000):
    super(CNN, self).__init__()
    # self.conv1 = nn.Conv2d(3, 6, 5)     # a batch = 32*32. 32*32*3 through 5*5*6 kernel grt 28*28*6. 
    # self.pool = nn.MaxPool2d(2, 2)      # 28*28*6 becomes 14*14*6
    # self.conv2 = nn.Conv2d(6, 16, 5)    # 14*14*6 through 5*5*16 kernel becomes 10*10*16
    # self.fc1 = nn.Linear(16 * 5 * 5, 120)  # after maxpooling, 10*10*16 becomes 5*5*16. linear transform it to 120
    # self.fc2 = nn.Linear(120, 84)       # linearly transform 120 to 84
    # self.fc3 = nn.Linear(84, num_classes)   # linearly transform 120 to number_of_classes. 

    self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )


    self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(8*8*128, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_classes)
        )
  def forward(self, x):
    # x = self.pool(F.relu(self.conv1(x)))
    # x = self.pool(F.relu(self.conv2(x)))
    # x = x.view(-1, 16 * 5 * 5)
    # x = F.relu(self.fc1(x))
    # x = F.relu(self.fc2(x))
    # x = self.fc3(x)
    # conv layers
    x = self.conv_layer(x)
    
    # flatten
    x = x.view(x.size(0), -1)
    
    # fc layer
    x = self.fc_layer(x)
    return x