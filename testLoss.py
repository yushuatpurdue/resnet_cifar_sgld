'''
The files are created by Yushu Huang for the ESGLD algorithm. YEAR 2024. 
They aim to apply the algorithm to the training of small neural networks under UCi datasets. 

In this file: calculating the testing loss for regression. 
'''

from torch.autograd import Variable

def testLoss(net, test_loader, criterion):
    net.eval() 
    cum_loss = 0
    for cnt, (images, labels) in enumerate(test_loader):
        inputs, targets = Variable(images).cuda(), Variable(labels).cuda()
        # inputs, response = Variable(images).cpu(), Variable(labels).cpu()
        predicts = net.forward(inputs).data           # returns a batch of response variables predicted by the model. 
        test_loss = criterion(predicts, targets)
        cum_loss += test_loss
    avg_loss = cum_loss/len(test_loader.dataset)
    return avg_loss
