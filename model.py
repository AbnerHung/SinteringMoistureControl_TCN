import torch.nn.functional as F
from torch import nn
from tcn import TemporalConvNet
import torch


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.linear2 = nn.Linear(16,2)

    def forward(self, x):
        """ Inputs have to have dimension (N, C_in, L_in) """
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        output = self.linear(output).double()
        output = output.view(1,16*1*1)
        output = self.linear2((output).float())
        output = output.view(2)
        #print("output.shape=",output.shape)
        return output
