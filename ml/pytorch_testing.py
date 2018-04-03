

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

# ======================================================================
#     Neural Network for draw-and-guess project
#
# In the comments I will explain how the neural network is setup using pyTorch.
# Note: This is how I (David-Mc) understand it. Take everything I
# say with a grain of salt. If someone more smarter than me
# tells you otherwise, go with that.
#
# --Referencing--
# I will use references like so [#]
# References are related or explain in detail some part of the code
# The # is a number denoting the reference
# Find references at the end of the file
# ======================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.datasets import ImageFolder


class Net(nn.Module):
    # This is the neural network.
    # nn is the pyTorch neural network module.
    # nn.Module is a neural network.
    # we override functions to build the network
    def __init__(self):
        self.total_classes = 10
        # The init function is where we setup the different layers
        # of the neural network.
        # We define them in the order we use them, by convention

        # As far as I understand it, there isn't any rhyme or reason to
        # the in/out values or kernel sizes and resulting accuracy.
        # You play around with them to see what works for your data.
        # For each layer, the out and in values need to match
        # For example, conv1 has out=64, conv2 has in=64
        super(Net, self).__init__()  # Init the underlying neural network (nn.Module)
        self.conv1 = nn.Conv2d(3, 64, 5)  # 2D convolution layer(in, out, kernel) [1]
        self.pool = nn.MaxPool2d(4, 4)  # Max pooling layer(kernel_size, stride) [2]
        self.conv2 = nn.Conv2d(64, 128, 5)  # Another 2D convolution layer (in, out, kernel) [1]
        self.fc1 = nn.Linear(128 * 12 * 12, 120)  # Linear transform (in, out) [4]
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.total_classes)  # Last layer need output neuron = to total_classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv Layer1 -> ReLU (activation) -> Max Pool -> x
        x = self.pool(F.relu(self.conv2(x)))  # Conv Layer2 -> ReLU (activation) -> Max Pool -> x
        # The following line transforms the dimension to be
        # 128 * 12 * 12 columns and however many rows fit
        # that many columns. -1 will end up being the batch size
        # I figured out 128 * 12 * 12 by looking at x in debug right before
        # it was transformed by the below line.
        # x.shape was 20, 128, 12, 12
        # 20 is the batch size (1st dimension)
        # Doubt that's how you're supposed to get the number
        # but, it still worked!
        x = x.view(-1, 128 * 12 * 12)  # view transform to [batch_size, 18432]
        x = F.relu(self.fc1(x))  # Linear layer -> ReLU (activation) -> x
        x = F.relu(self.fc2(x))  # Linear layer -> ReLU (activation) -> x
        x = self.fc3(x)  # Linear layer -> x
        return x  # x.shape = [batch_size, total_classes]


# This is the transform applied to all images
# The Normalize portion comes after it's turned into a tensor
# I'm using the defaults from ResNet for the transform
# The transforms can be anything, as long as the image
# is normalized and all images as the same size
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Create the data loader (training images) and the test loader (test iamges)
dataset = ImageFolder(root='./images/train', transform=transform)  # [6]
dataloader = data.DataLoader(dataset, batch_size=20,
                             shuffle=True, num_workers=2)
testset = ImageFolder(root='./images/test', transform=transform)  # [6]
testloader = torch.utils.data.DataLoader(testset, batch_size=20,
                                         shuffle=False, num_workers=2)
net = Net()  # Create instance of our neural network, untrained
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print('beginning training...')

# epoch is just the number of times we want to
# train using all the training images
for epoch in range(5):
    print('epoch ' + str(epoch))
    running_loss = 0.0  # Keeps track of how badly the network is classifying
    for i, data in enumerate(dataloader, 0):  # Iterate through the mini-batches
        if i % 200 == 0:
            print('dataloader: ' + str(i))
        # get the inputs
        inputs, labels = data  # inputs = mini-batch images, labels = mini-batch labels

        # wrap them in Variable
        # uncomment following and comment one after to enable GPU processing
        # inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        inputs, labels = Variable(inputs), Variable(labels)
        # Variable's are used to allow for automatic back propagation (shown later) [5]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)  # get the guesses for all the images in the mini-batch
        loss = criterion(outputs, labels)  # Get loss function (compare guesses with correct labels)
        loss.backward()  # autograd automatic back-propagation [5]
        optimizer.step()  # optimization step

        # # print statistics -- optional
        # running_loss += loss.data[0]
        # if i % 2000 == 1999:  # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 2000))
        #     running_loss = 0.0
        # print('finished {}. running_loss: {}'.format(i, running_loss))

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

# ___References___
# [1] Convolutional Neural Networks (CNN): https://hooktube.com/watch?v=FTr3n7uBIuE
# [2] Max pooling explained w/CNN details: https://cs231n.github.io/convolutional-networks/
# [3] View function: https://stackoverflow.com/questions/42479902/how-view-method-works-for-tensor-in-torch/42482819#42482819
# [4] Linear transform layers: http://pytorch.org/docs/master/_modules/torch/nn/modules/linear.html
# [5] Variables and autograd: http://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_autograd.html
# [6] ImageFolder: http://pytorch.org/docs/master/torchvision/datasets.html#imagefolder
