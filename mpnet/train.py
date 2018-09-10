#!/usr/bin/env python

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import MPNet
from python_pfm import readPFM, writePFM

###################################

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MPNet example')
parser.add_argument('--batch-size', type=int, default=13,
                    help='input batch size for training (default: 13)')
parser.add_argument('--epochs', type=int, default=27,
                    help='number of epochs to train (default: 27)')
parser.add_argument('--lr', type=float, default=0.003,
                    help='learning rate (default: 0.003)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.005,
		    help='SGD weight decat (default: 0.005)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
args = parser.parse_args()

# Cuda
args.cuda = not args.no_cuda and torch.cuda.is_available()
USE_CUDA = args.cuda

# Set the seed
torch.manual_seed(args.seed)
if args.cuda :
    torch.cuda.manuel_seed(args.seed)

# Prepare the dataset
path = '/home/raymond/Sampler/FlyingThings3D/optical_flow'
dataset = []
for i in ['backward', 'forward'] :
    new_path = path + '/' + i
    for images in os.listdir(new_path) :
        data, scale = readPFM(new_path + '/' + images)
        data = np.transpose(data[:,:,0:2], (2, 0, 1))
        dataset.append(data)
print(len(dataset))

# Create MPNet Model
model = MPNet()
if USE_CUDA :
    model.cuda()
else :
    model.float()
print(model)

# Define the optimizer and the loss
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),
			    lr=args.lr,
			    momentum=args.momentum,
			    weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
						 milestones=[9,18,27],
						 gamma=0.1)

# Update weight decay
def adjust_weight_decay(optimizer, epoch):
    """Sets the weight decay to the initial WC decayed by 0.1 every 9 epochs"""
    weight_decay = weight_decay * (0.1 ** (epoch // 9))
    for param_group in optimizer.param_groups:
        param_group['weight_decay'] = weight_decay
    return optimizer

# Train the network
total_step = len(train_loader)
for epoch in range(1, num_epochs) :

    if epoch % 9 == 0 :
        scheduler.step()
        optimizer = adjust_weight_decay(optimizer, epoch)

    for i, (images, labels) in enumerate(train_loader) :
        if USE_CUDA :
            images = images.cuda()
            labels = labels.cuda()

        # Forward
        outputs = net(images)
        loss = criterion(outputs, labels)

        # Backward and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the network
model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        if USE_CUDA :
            images = images.cuda()
            labels = labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Test Accuracy of the model on the 10000 test images: {} %'
      .format(100 * correct / total))

# Save the model
torch.save(model.state_dict(), 'model.pth')

