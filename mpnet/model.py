#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def adjust_weight_decay(optimizer, epoch):
    """Sets the weight decay to the initial WC decayed by 0.1 every 9 epochs"""
    weight_decay = weight_decay * (0.1 ** (epoch // 9))
    for param_group in optimizer.param_groups:
        param_group['weight_decay'] = weight_decay
    return optimizer

class MPNet(nn.Module) :

    def __init__(self) :

        super(MPNet, self).__init__()

        # Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=7, stride=1, padding=3))

        # Layer 2
        self.layer2 = nn.Sequential(
            nn.BatchNorm2d(num_features=64, eps=1e-4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU())

        # Layer 3
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU())

        # Layer 4
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU())

        # Layer 5
        self.layer5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU())

        # Layer 6
        self.layer6 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU())

        self.m = nn.UpsamplingBilinear2d(scale_factor=2)

        # Layer 1d
        self.layer1d = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU())

        # Layer 2d
        self.layer2d = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU())

        # Layer 3d
        self.layer3d = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU())

        # Layer 4d
        self.layer4d = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=1),
            nn.Sigmoid())

    def forward(self, input) :

        # Encoder part
        x1 = self.layer1(input)
        x2 = self.layer2(x1)
        x3 = self.layer3(x3)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)

        # Decoder part
        x6 = self.m(x6)
        x1d = torch.cat((x6, x5), 3)
        x1d = self.layer1d(x1d)

        x1d = self.m(x1d)
        x2d = torch.cat((x1d, x4), 3)
        x2d = self.layer2d(x2d)

        x2d = self.m(x2d)
        x3d = torch.cat((x2d, x3), 3)
        x3d = self.layer3d(x3d)

        x3d = self.m(x3d)
        x4d = torch.cat((x3d, x2), 3)
        x4d = self.layer4d(x4d)

        return x4d

