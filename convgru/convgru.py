#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
from torch.autograd import Variable

class ConvGRUCell(nn.Module) :
    """ Generate a Convolutional GRU cell """

    def __init__(self, input_size, hidden_size, kernel_size) :

        super(ConvGRUCell, self).__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.reset_gate = nn.Conv2d(in_channels = input_size + hidden_size, 
        							out_channels = hidden_size, 
        							kernel_size = kernel_size, 
        							padding = padding, 
        							bias = True)
        self.update_gate = nn.Conv2d(in_channels = input_size + hidden_size, 
        							out_channels = hidden_size, 
        							kernel_size = kernel_size, 
        							padding = padding, 
        							bias = True)
        self.out_gate = nn.Conv2d(in_channels = input_size + hidden_size, 
        							out_channels = hidden_size, 
        							kernel_size = kernel_size, 
        							padding = padding, 
        							bias = True)

        init.xavier_uniform_(self.reset_gate.weight)
        init.xavier_uniform_(self.update_gate.weight)
        init.xavier_uniform_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0)
        init.constant_(self.update_gate.bias, 0)
        init.constant_(self.out_gate.bias, 0)

        print(self.reset_gate.weight)

    def forward(self, input, prev_state) :

    	# get batch and spatial sizes
    	batch_size = input.size()[0]
    	spatial_size = input.size()[2:]

    	# generate empty state, if None is provided
    	if prev_state is None : 
    		state_size = [batch_size, self.hidden_size] + list(spatial_size)
    		if torch.cuda.is_available() : 
    			prev_state = Variable(torch.zeros(state_size)).cuda()
    		else : 
    			prev_state = Variable(torch.zeros(state_size))

    	# data size is [batch_size, number_channels, height, width]
    	stacked_inputs = torch.cat([input, prev_state], dim = 1)
    	update = torch.sigmoid(self.update_gate(stacked_inputs))
    	reset = torch.sigmoid(self.reset_gate(stacked_inputs))
    	out_inputs = torch.tanh(self.out_gate(torch.cat([input, prev_state * reset], dim = 1)))
    	new_state = prev_state * (1 - update) + out_inputs * update

    	return new_state

class ConvGRU(nn.Module) : 
	""" Generate a Convolutional GRU """

	def __init__(self, input_size, hidden_sizes, kernel_sizes, number_layers) : 
		'''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.
        Parameters
        ----------
        input_size : integer. depth dimension of input tensors.
        hidden_sizes : integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        kernel_sizes : integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        n_layers : integer. number of chained `ConvGRUCell`.
        '''

		super(ConvGRU, self).__init__()
		self.input_size = input_size
		self.number_layers = number_layers

		if type(hidden_sizes) != list : 
			self.hidden_sizes = [hidden_sizes] * number_layers
		else : 
			self.hidden_sizes = hidden_sizes

		if type(kernel_sizes) != list : 
			self.kernel_sizes = [kernel_sizes] * number_layers
		else : 
			self.kernel_sizes = kernel_sizes


		cells = []
		for i in range(self.number_layers) : 
			if i == 0 : 
				input_dim = self.input_size
			else : 
				input_dim = self.hidden_sizes[i-1]

			cell = ConvGRUCell(input_size=input_dim, hidden_size=hidden_sizes[i], kernel_size=kernel_sizes[i])
			name = "ConvGRUCell_" + str(i).zfill(2)

			setattr(self, name, cell)
			cells.append(getattr(self, name))

		self.cells = cells

	def forward(self, input, hidden=None) : 

		if hidden==None: 
			hidden = [None] * self.number_layers

		upd_hidden = []

		for layer in range(self.number_layers) : 
			cell = self.cells[layer]
			cell_hidden = hidden[layer]

			# Pass through layers
			upd_cell_hidden = cell(input, cell_hidden)
			upd_hidden.append(upd_cell_hidden)

			# Update input to the last updated hidden layer next pass
			input = upd_cell_hidden

		return upd_hidden