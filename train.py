from model import *
from data.data import *
from data.data import data_generator
from torch.autograd import Variable
from torch.optim import optimizer
from sklearn.model_selection import train_test_split

import torch
import argparse
import numpy as np
import torch.nn as nn

from numpy.core.fromnumeric import clip


data, add_water=data_generator()
#print(data.shape)

train_set, test_set = train_test_split(data, test_size=0.1, random_state=0)
train_water, test_water = train_test_split(add_water, test_size=0.1, random_state=0)
#print(len(train_set))
#print(add_water.shape)

num_channels=[16,16,16]#隐层每层的hidden_channel数
model = TCN(16,2,num_channels,2,0.2)#in,out,channels,kernel,dropout
LossF = nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters(),
                    lr=0.01,
                    alpha=0.99,
                    eps=1e-08,
                    weight_decay=0,
                    momentum=0,
                    centered=False)
clipGrad=0

train_set=torch.Tensor(train_set)
test_set=torch.Tensor(test_set)
train_water=torch.Tensor(train_water)
test_water=torch.Tensor(test_water)

def train(ep):
    model.train()
    total_loss = 0
    count = 0
    acc=0
    train_idx_list = np.arange(len(train_set), dtype="int32")

    for idx in train_idx_list:
        data_line = train_set[idx]
        x = Variable(data_line[:-1])
        print(x.shape)
        x = x.cuda()

        optimizer.zero_grad()
        output = model(x)
        loss =LossF(output[0],train_water[idx][0])+LossF(output[1],train_water[idx][1])
        acc1=(1.*output[0])/train_water[idx][0]
        acc2=(1.*output[1])/train_water[idx][1]
        acc+=acc1
        acc+=acc2
        total_loss += loss.item()
        count += output.size(0)

        if clipGrad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipGrad)
        loss.backward()
        optimizer.step()
        if idx > 0 and idx % 100 == 0:
            cur_loss = total_loss / count
            acc /= count
            print("Epoch {:2d} | acc {:.5f} | loss {:.5f}".format(ep, acc , cur_loss))
            total_loss = 0.0
            count = 0
            acc=0
train(0)
