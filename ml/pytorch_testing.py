

# Bit about auto gradient calculations with torch.autograd Variable's

# import torch
# from torch.autograd import Variable
#
# # create Variable which keeps track of operations
# # to allow for auto computing of gradients.
# # requires_grad=True means this Variable needs
# # gradients computed for it
# x = Variable(torch.ones(2, 2), requires_grad=True)
# print(x)
# # x.grad_fn is the gradient function that created this vVariable
# # grad_fn only exists if the Variable was created by an operation
# # i.e. user-created Variable's are None
#
# # Do some operations
# y = x * 2
# y = y * y * 2
# print(y.grad_fn)  # y has a grad_fn
#
# gradients = torch.FloatTensor([.1, 1.0])
# y.backward(gradients)
# print(x.grad)
# # More info about autograd:
# http://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html


# Update the weights

#
# The simplest update rule used in practice is the Stochastic Gradient Descent (SGD):
#
#     weight = weight - learning_rate * gradient
#
# We can implement this using simple python code:
#
# learning_rate = 0.01
# for f in net.parameters():
#     f.data.sub_(f.grad.data * learning_rate)
#
# However, as you use neural networks, you want to use various different update
# rules such as SGD, Nesterov-SGD, Adam, RMSProp, etc. To enable this, we built a
# small package: torch.optim that implements all these methods. Using it is very simple:
#
# `import torch.optim as optim`
#
# # create your optimizer
# optimizer = optim.SGD(net.parameters(), lr=0.01)
#
# # in your training loop:
# optimizer.zero_grad()   # zero the gradient buffers
# output = net(input)
# loss = criterion(output, target)
# loss.backward()
# optimizer.step()    # Does the update
#
# More info about this code here:
# http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html


# Image recognition example using pyTorch
# Code following: http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import ImageFolder


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128 * 12 * 12, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

dataset = ImageFolder(root='./images/train', transform=transform)
dataloader = data.DataLoader(dataset, batch_size=20,
                             shuffle=True, num_workers=2)
testset = ImageFolder(root='./images/test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=20,
                                         shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print('beginning training...')

for epoch in range(5):
    print('epoch ' + str(epoch))
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        if i % 200 == 0:
            print('dataloader: ' + str(i))
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        # uncomment following and comment one after to enable GPU processing
        # inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        # labels = torch.unsqueeze(torch.ones([]), 0)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
        print('finished {}. running_loss: {}'.format(i, running_loss))

print('Finished Training')

# Check prediction accuracy
correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# Example output of a session with five epochs
# Files already downloaded and verified
# Files already downloaded and verified
# [1,  2000] loss: 2.206
# [1,  4000] loss: 1.850
# [1,  6000] loss: 1.652
# [1,  8000] loss: 1.551
# [1, 10000] loss: 1.500
# [1, 12000] loss: 1.477
# [2,  2000] loss: 1.403
# [2,  4000] loss: 1.358
# [2,  6000] loss: 1.332
# [2,  8000] loss: 1.313
# [2, 10000] loss: 1.321
# [2, 12000] loss: 1.265
# [3,  2000] loss: 1.211
# [3,  4000] loss: 1.209
# [3,  6000] loss: 1.201
# [3,  8000] loss: 1.157
# [3, 10000] loss: 1.190
# [3, 12000] loss: 1.174
# [4,  2000] loss: 1.098
# [4,  4000] loss: 1.081
# [4,  6000] loss: 1.097
# [4,  8000] loss: 1.112
# [4, 10000] loss: 1.100
# [4, 12000] loss: 1.076
# [5,  2000] loss: 1.007
# [5,  4000] loss: 1.002
# [5,  6000] loss: 1.017
# [5,  8000] loss: 1.034
# [5, 10000] loss: 1.027
# [5, 12000] loss: 1.038
# Finished Training
# Accuracy of the network on the 10000 test images: 61 %
