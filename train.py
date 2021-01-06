from model import *
from data.data import data_generator
from torch.autograd import Variable
from torch.optim import optimizer
from sklearn.model_selection import train_test_split

import torch
import argparse
import numpy as np
import torch.nn as nn

from numpy.core.fromnumeric import clip

data, add_water = data_generator()
# print(data.shape)
add_water = torch.Tensor(add_water)
num_channels = [16, 8, 4]  # 隐层每层的hidden_channel数
model = TCN(16, 1, num_channels, 2, 0.2)  # in,out,channels,kernel,dropout
LossF = nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters(),
                                lr=0.01,
                                alpha=0.99,
                                eps=1e-08,
                                weight_decay=0,
                                momentum=0,
                                centered=False)
clipGrad = 0


def train(ep):
    model.train()
    total_loss = 0
    count = 0
    acc = 0

    # print(torch.Tensor(input[0:16, :]).shape)

    for i in range(38095):
        x = torch.Tensor(data[i:i+16, :])
        x = x.unsqueeze(0)
        # print(x.shape)
        optimizer.zero_grad()
        output = model(x)
        output = output.view(2)
        # print(output[0])
        # print(add_water[i + 15][0])
        loss = LossF(output[0], add_water[i+15][0]) + LossF(output[1], add_water[i+15][1])
        acc1 = (1. * output[0]) / add_water[i+15][0]
        acc2 = (1. * output[1]) / add_water[i+15][1]
        #print("acc1.shape=",acc1.shape)
        acc += acc1
        acc += acc2
        total_loss += loss.item()
        count += output.size(0)

        if clipGrad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipGrad)
        loss.backward()
        optimizer.step()
        if i > 0 and i % 100 == 0:
            cur_loss = total_loss / count
            acc /= count
            print("Epoch {:2d} |  | loss {:.5f}".format(ep,  cur_loss))
            total_loss = 0.0
            count = 0
            acc = 0


train(0)
