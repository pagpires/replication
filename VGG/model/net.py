"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
        module.bias.data.zero_()
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.uniform_()
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


class Net(nn.Module):
    """ 
    Implementation of VGG model
    """
    def __init__(self, params):
        super(Net, self).__init__()

        # set up nn configuration, may be better if kept in a sep config file
        n_classes = params.n_classes
        input_shape = params.input_shape

        self.use_bn = params.use_bn

        self.stage1 = self._make_stage(3, 64, 2)
        self.stage2 = self._make_stage(64, 128, 2)
        self.stage3 = self._make_stage(128, 256, 3)
        self.stage4 = self._make_stage(256, 512, 3)
        self.stage5 = self._make_stage(512, 512, 3)

        # compute conv feature size
        with torch.no_grad():
            self.feature_size = self._forward_conv(
                torch.zeros(*input_shape)).view(-1).shape[0]

        self.fc = nn.Linear(self.feature_size, n_classes)

        # initialize weights
        self.apply(initialize_weights)

    def _make_stage(self, in_channels, out_channels, n_blocks):
        stage = nn.Sequential()
        for index in range(n_blocks):
            
            if index == 0:
                input_channel = in_channels
            else:
                input_channel = out_channels
            
            conv = nn.Conv2d(input_channel, out_channels, kernel_size=3, stride=1, padding=1)
            
            stage.add_module('conv{}'.format(index), conv)
            if self.use_bn:
                stage.add_module('bn{}'.format(index),
                                 nn.BatchNorm2d(out_channels))
            stage.add_module('relu', nn.ReLU(inplace=True))
        stage.add_module('pool', nn.MaxPool2d(kernel_size=2, stride=2))
        return stage

    def _forward_conv(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # compute log soft max for bettter num stability
        return F.log_softmax(x, dim=1)


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    """
    num_examples = outputs.size()[0]
    return -torch.sum(outputs[range(num_examples), labels])/num_examples


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x len(labels) - log softmax output of the model
        labels: (np.ndarray) dimension batch_size

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
