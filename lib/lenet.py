import torch
import torch.nn.functional as F
import torch.nn as nn

from fully_connected_net import FullyConnectedNet


class LeNet(nn.Module):
    def __init__(self, input_size,
                       output_size,

                       batch_norm,

                       use_pooling,
                       pooling_method,

                       conv1_kernel_size,
                       conv1_num_kernels,
                       conv1_stride,
                       conv1_dropout,

                       pool1_kernel_size,
                       pool1_stride,

                       conv2_kernel_size,
                       conv2_num_kernels,
                       conv2_stride,
                       conv2_dropout,

                       pool2_kernel_size,
                       pool2_stride,

                       fcs_hidden_size,
                       fcs_num_hidden_layers,
                       fcs_dropout):

        super(LeNet, self).__init__()

        self.input_size = input_size
        self.hidden_size = fcs_hidden_size
        self.output_size = output_size

        self.batch_norm = batch_norm

        input_channel = 1


        # conv1
        conv1_output_size = (conv1_num_kernels, (input_size - conv1_kernel_size) / conv1_stride + 1)

        self.conv1 = nn.Conv2d(input_channel, conv1_num_kernels, conv1_kernel_size, stride=conv1_stride)
        nn.init.kaiming_normal_(self.conv1.weight.data)
        self.conv1.bias.data.fill_(0)


        # pool1
        pool1_output_size = (conv1_num_kernels, (conv1_output_size[1] - pool1_kernel_size) / pool1_stride + 1)

        self.pool1 = nn.MaxPool2d(pool1_kernel_size, stride=pool1_stride) # stride=pool1_kernel_size by default


        # conv2
        conv2_output_size = (conv2_num_kernels, (pool1_output_size[1] - conv2_kernel_size) / conv2_stride + 1)

        self.conv2 = nn.Conv2d(conv1_num_kernels, conv2_num_kernels, conv2_kernel_size, stride=conv2_stride)
        nn.init.kaiming_normal_(self.conv2.weight.data)
        self.conv2.bias.data.fill_(0)


        # pool2
        pool2_output_size = (conv2_num_kernels, (conv2_output_size[1] - pool2_kernel_size) / pool2_stride + 1)

        self.pool2 = nn.MaxPool2d(pool2_kernel_size, stride=pool2_stride) # stride=pool1_kernel_size by default


        # fcs (fully-connected layers)
        fcs_input_size = pool2_output_size[0] * pool2_output_size[1]
        self.fcs = FullyConnectedNet(fcs_input_size,
                                     output_size,

                                     fcs_dropout,
                                     batch_norm,

                                     fcs_hidden_size,
                                     fcs_num_hidden_layers)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, x):
        # pytorch.conv1d accepts shape (Batch, Channel, Width)
        # pytorch.conv2d accepts shape (Batch, Channel, Height, Width)
        # import code; code.interact(local=dict(globals(), **locals()))
        # N_elements = int(x.shape[1] / 2)
        # x = x.view(-1, 2, N_elements)

        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(x)
        x = self.conv1_drop(x)
        if self.batch_norm == True:
            x = self.batch_norm1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = F.relu(x)
        x = self.conv2_drop(x)
        if self.batch_norm == True:
            x = self.batch_norm2(x)

        x = x.view(-1, x.size(1) * x.size(2))

        x = self.fcs.forward(x)

        return x

    def name(self):
        return "LeNet"
