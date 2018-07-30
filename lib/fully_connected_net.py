from torch import nn
import torch
import os


class FullyConnectedNet(nn.Module):
    """Fully connected network builder for any number of layers."""
    def __init__(self, input_size,
                       output_size,

                       fcs_dropout,
                       batch_norm,

                       fcs_hidden_size,
                       fcs_num_hidden_layers):
        super(FullyConnectedNet, self).__init__()

        self.batch_norm = batch_norm

        # input connects to first hidden layer
        self.layers = nn.ModuleList([nn.Linear(input_size, fcs_hidden_size)])
        for i in range(fcs_num_hidden_layers - 1):
            self.layers.append(nn.Linear(fcs_hidden_size, fcs_hidden_size))
        # last hidden connects to output layer
        self.layers.append(nn.Linear(fcs_hidden_size, output_size))


        # build as many batch_norm layers minus the last one
        if self.batch_norm == True:
            self.batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(fcs_hidden_size)]) # TODO: not input_size?
            for i in range(fcs_num_hidden_layers - 1):
                self.batch_norm_layers.append(nn.BatchNorm1d(fcs_hidden_size))
        else:
            warnings.warn('fully_connected_net: not using batch_norm.')


        # Activation and fcs_dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(fcs_dropout)


        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            if self.batch_norm == True:
                x = self.batch_norm_layers[i](x)
            x = self.relu(x)
            x = self.dropout(x) # is dropout learnable?

        # no dropout or activtion function on the last layer
        x = self.layers[-1](x)

        return x

    def _initialize_weights(self):
        for i in range(len(self.layers)):
            nn.init.kaiming_normal_(self.layers[i].weight.data)
            self.layers[i].bias.data.fill_(0)
