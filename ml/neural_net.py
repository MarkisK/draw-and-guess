

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
import datetime
import math
import pathlib
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torchvision.datasets import ImageFolder

run_time = datetime.datetime.now()


def log(message):
    with open('logfile.txt', 'a+') as f:
        f.write('\n{}\n{}'.format(run_time, message))


def get_class_count(file_type='.ndjson'):
    count = 0
    for p in pathlib.Path('ml/data/').iterdir():
        if p.name.endswith(file_type):
            count += 1
    return count


def get_class_list(file_type='.ndjson'):
    classes = []
    for p in pathlib.Path('ml/data/').iterdir():
        if p.name.endswith(file_type):
            classes.append(p.name[p.name.rfind('_')+1:p.name.rfind('.')])
    return classes


def get_class_dict(file_type='.ndjson'):
    classes = {}
    for p in pathlib.Path('ml/data/').iterdir():
        if p.name.endswith(file_type):
            classes[p.name[p.name.rfind('_')+1:p.name.rfind('.')]] = 0
    return classes


def train_image_count():
    count = 0
    train_dir = pathlib.Path('ml/images/train/')
    for d in train_dir.iterdir():
        for _ in d.iterdir():
            count += 1
        return count
    return 0


def save_model(model, path='ml/models/{}.pth'.format(run_time)):
    return torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
    return


def list_class_loss(cl):
    print("Class Loss")
    tups = [(k, v) for k, v in cl.items()]
    tups.sort(key=lambda x: x[1])  # sort by second value in tuple
    for tup in tups:
        print('{0:20}: {1}'.format(tup[0], tup[1]))
    return


class Net(nn.Module):
    # This is the neural network.
    # nn is the pyTorch neural network module.
    # nn.Module is a neural network.
    # we override functions to build the network
    def __init__(self, total_classes=10):
        self.total_classes = total_classes
        self.reshape = 64 * 25 * 25
        # The init function is where we setup the different layers
        # of the neural network.
        # We define them in the order we use them, by convention

        # As far as I understand it, there isn't any rhyme or reason to
        # the in/out values or kernel sizes and resulting accuracy.
        # You play around with them to see what works for your data.
        # For each layer, the out and in values need to match
        # For example, conv1 has out=64, conv2 has in=64
        super(Net, self).__init__()  # Init the underlying neural network (nn.Module)
        self.conv0 = nn.Conv2d(1, 24, 1, stride=1)
        self.conv1 = nn.Conv2d(24, 96, 3, padding=1)  # 2D convolution layer(in, out, kernel) [1]
        self.pool1 = nn.MaxPool2d(2, 2)  # Max pooling layer(kernel_size, stride) [2]
        self.pool2 = nn.MaxPool2d(4, 2)
        self.conv2 = nn.Conv2d(96, 64, 8, padding=1)  # Another 2D convolution layer (in, out, kernel) [1]
        self.fc1 = nn.Linear(self.reshape, 512)  # Linear transform (in, out) [4]
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, self.total_classes)  # Last layer need output neuron = to total_classes
        self.drop = nn.Dropout2d(p=.2)  # Dropout [7]
        self.debug = [self.conv0, self.conv1, self.conv2,
                      self.pool1, self.pool2, self.fc1, self.fc2, self.fc3,
                      self.drop, 'reshape: {}'.format(self.reshape)]

    def forward(self, x):
        x = self.pool1(F.relu(self.conv0(x)))
        x = self.pool1(F.relu(self.conv1(x)))  # Conv Layer1 -> ReLU (activation) -> Max Pool -> x
        x = self.pool1(F.relu(self.conv2(x)))  # Conv Layer2 -> ReLU (activation) -> Max Pool -> x
        # The following line transforms the dimension to be
        # 128 * 12 * 12 columns and however many rows fit
        # that many columns. -1 will end up being the batch size
        # I figured out 128 * 12 * 12 by looking at x in debug right before
        # it was transformed by the below line.
        # x.shape was 20, 128, 12, 12
        # 20 is the batch size (1st dimension)
        # Doubt that's how you're supposed to get the number
        # but, it still worked!
        x = x.view(-1, self.reshape)  # view transform to [batch_size, 18432]
        x = F.relu(self.fc1(x))  # Linear layer -> ReLU (activation) -> x
        x = self.drop(x)  # dropout [7]
        x = F.relu(self.fc2(x))  # Linear layer -> ReLU (activation) -> x
        x = self.drop(x)
        x = self.fc3(x)  # Linear layer -> x
        return x  # x.shape = [batch_size, total_classes]


def make_guess(model, image, classes=None):
    if classes is None:
        classes = ['apple', 'axe', 'banana', 'baseball',
                   'book', 'bucket', 'car', 'cat',
                   'church', 'circle', 'clock', 'cloud',
                   'coffee cup', 'cookie', 'cow', 'diamond',
                   'donut', 'envelope', 'fork',
                   'hand', 'hexagon', 'hockey stick',
                   'hourglass', 'house', 'ice cream',
                   'jail', 'knife', 'ladder', 'leaf',
                   'light bulb', 'lightning', 'line',
                   'lollipop', 'microphone', 'moon',
                   'mountain', 'oven', 'pizza',
                   'power outlet', 'sandwich', 'shovel',
                   'smiley face', 'square', 'star', 'sun',
                   'teddy-bear', 'toaster', 'tree']

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    if isinstance(image, type('')):  # is path string
        img = Image.open(image)
    else:
        img = image
    tensor = transform(img).unsqueeze(0)
    _v = Variable(tensor)
    result = model(_v)
    _, predicted = torch.max(result.data, 1)
    guess = classes[predicted[0]]
    return guess


if __name__ == "__main__":
    mini_batch_size = 10
    # This is the transform applied to all images
    # The Normalize portion comes after it's turned into a tensor
    # I'm using the defaults from ResNet for the transform
    # The transforms can be anything, as long as the image
    # is normalized and all images as the same size
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Create the data loader (training images) and the test loader (test iamges)
    dataset = ImageFolder(root='./images/train',
                          transform=transform)  # [6]
    dataloader = data.DataLoader(dataset,
                                 batch_size=mini_batch_size,
                                 shuffle=True,
                                 num_workers=2)

    testset = ImageFolder(root='./images/test',
                          transform=transform)  # [6]
    testloader = data.DataLoader(testset,
                                 batch_size=mini_batch_size,
                                 shuffle=False,
                                 num_workers=2)

    class_loss = get_class_dict()

    # Create instance of our neural network, untrained
    net = Net(get_class_count(file_type='.ndjson'))
    # load_model(net, 'models/2018-04-09 20:13:48.191474.pth')

    # Create loss function
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()

    # Create optimizer (gradient descent)
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, nesterov=True)
    # optimizer = optim.ASGD(net.parameters(), lr=0.01, weight_decay=1e-4)
    # optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)

    print('beginning training...')

    # epoch is just the number of times we want to
    # train using all the training images
    epochs = 2
    prev_loss = 0.0
    for epoch in range(epochs):
        print('epoch ' + str(epoch))
        running_loss = 0.0  # Keeps track of how badly the network is classifying
        for i, data in enumerate(dataloader, 0):  # Iterate through the mini-batches
            # if i % 200 == 0:
            #     print('{}: {}'.format('dataloader', i))
            # get the inputs
            inputs, labels = data  # inputs = mini-batch images, labels = mini-batch labels

            # wrap them in Variab   le
            # uncomment following and comment one after to enable GPU processing
            inputs, labels = Variable(inputs), Variable(labels)
            # inputs, labels = Variable(inputs), Variable(labels)
            # Variable's are used to allow for automatic back propagation (shown later) [5]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)  # get the guesses for all the images in the mini-batch
            loss = criterion(outputs, labels)  # Get loss function (compare guesses with correct labels)
            loss.backward()  # autograd automatic back-propagation [5]
            optimizer.step()  # optimization step

            # print statistics
            running_loss += loss.data[0]
            if i % 1000 == 0:  # print every 1000 mini-batches
                curr_loss = running_loss / 1000
                try:
                    change = -1 * (prev_loss / curr_loss - 1) * 100  # % Change from last calculation
                except ZeroDivisionError:
                    change = 0.0
                diff = -1 * (prev_loss - curr_loss)
                prev_loss = curr_loss
                print('[{0}, {1}] loss: {2:.3f} change: {3:.3f}% difference: {4:.3f}'.format(
                    epoch + 1, i + 1, running_loss / 1000, change, diff
                ))
                running_loss = 0.0
            print('finished {0}. running_loss: {1:16.10f} +{2:.2f}'.format(
                i, running_loss, loss.data[0]
            ))

            # The following is an attempt to determine which
            # classes cause the network the most trouble.
            # Compute how often a label was seen in this mini-batch
            class_count = get_class_dict()  # Holds count of label occurrence
            for label in labels:
                idx = label.data[0]
                key = dataset.classes[idx]
                class_count[key] += 1
            # Compute and add the total portion of loss contribution
            for key, total in class_count.items():
                if total > 0:
                    perc = total/mini_batch_size  # total occurrences / mini-batch size
                    class_loss[key] += loss.data[0]*perc  # loss * % of mini-batch

    print('Finished Training')

    save_model(net)
    print('Model saved.')

    print('2s wait before testing...')
    time.sleep(2)
    print('Evaluating...')
    # Check prediction accuracy
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        images, labels = images, labels
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    train_count = train_image_count()
    score = 100 * correct / total
    out = 'Score: {}\n{}\n{}\nConfig:\n\t{}\n\t{}\n\t{}\n\t'.format(
        score,
        '{} train images'.format(train_count-1),
        '{} epochs'.format(epochs),
        '\n\t'.join([str(d) for d in net.debug]),
        'Optimizer: {}'.format(str(optimizer)),
        'Loss Function: {}'.format('CrossEntropyLoss'),
        'Class list: {}'.format(get_class_list())
    )
    print('Accuracy of the network on the {} test images: {} %'.format(
        train_count, math.ceil(score*100)/100
    ))
    print(out)
    log(out)

    print(list_class_loss(class_loss))

# ___References___
# [1] Convolutional Neural Networks (CNN): https://hooktube.com/watch?v=FTr3n7uBIuE
# [2] Max pooling explained w/CNN details: https://cs231n.github.io/convolutional-networks/
# [3] View function: https://stackoverflow.com/questions/42479902/how-view-method-works-for-tensor-in-torch/42482819#42482819
# [4] Linear transform layers: http://pytorch.org/docs/master/_modules/torch/nn/modules/linear.html
# [5] Variables and autograd: http://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_autograd.html
# [6] ImageFolder: http://pytorch.org/docs/master/torchvision/datasets.html#imagefolder
# [7] Dropout to avoid overfitting: http://pytorch.org/docs/master/_modules/torch/nn/modules/dropout.html
# [8] Save and Load models: https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch#43819235
