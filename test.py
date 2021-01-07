from model import *
import sys
import signal
from data import data_generator
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


def test():
    model.load_state_dict(torch.load('modeldata_latest.pkl'))
    count = 0
    count1 = 0
    count2 = 0     
    rmse = 0
    rmser = 0

    for i in range(34285, 38095):
        x = torch.Tensor(data[i:i+16, :])
        x = x.unsqueeze(0)

        output = model(x)
        output = output.view(2)

        rmse += (output[0]+output[1] - add_water[i + 15][0]-add_water[i + 15][1])**2
        rmser += ((output[0]+output[1] - add_water[i + 15][0]-add_water[i + 15][1]) /
                  (add_water[i + 15][0]+add_water[i + 15][1]))**2

    '''   
        if abs((output[0]-add_water[i+15][0])/add_water[i+15][0]) <= 0.05:
            count1 += 1
        if abs((output[1]-add_water[i+15][1])/add_water[i+15][1]) <= 0.05:
            count2 += 1
        if (abs((output[0]-add_water[i+15][0])/add_water[i+15][0]) <= 0.05) and 
                (abs((output[1] - add_water[i + 15][1]) / add_water[i + 15][1]) <= 0.05):
            count += 1
        if i > 0 and i % 1000 == 0:
            print(i/381.0, "%")
        
    print("acc1 : ", count1 / 3810)
    print("acc2 : ", count2 / 3810)
    print("acc : ", count / 3810)
    '''
    print("rmse  : ", math.sqrt(rmse/3810))
    print("rmser : ", math.sqrt(rmser/3810))


test()

