#!/usr/bin/env python 

import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from convgru import ConvGRU

# Generate a ConvGRU with 3 cells
# input_size and hidden_sizes reflect feature map depths.
# Height and Width are preserved by zero padding within the module.
model = ConvGRU(input_size=8, hidden_sizes=[32,64,16],
                  kernel_sizes=[3, 5, 3], number_layers=3)

x = Variable(torch.FloatTensor(1,8,64,64))
output = model(x)

print(output)

# output is a list of sequential hidden representation tensors
print(type(output)) # list

# final output size
print(output[-1].size()) # torch.Size([1, 16, 64, 64])

