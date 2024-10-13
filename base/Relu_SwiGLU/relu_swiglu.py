#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   relu_swiglu.py
@Time    :   2024/10/13 16:32:52
@Author  :   Lifeng Xu 
@desc :   
'''
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

def swish(x, beat=1):
    return x / (1 + np.exp(-beat * x))

def relu(x):
    return np.maximum(0, x)

def show():
    x = np.linspace(-10, 10, 1000)
    y_relu = relu(x)
    y_swish = swish(x, 1)

    plt.plot(x, y_relu, label='ReLU')
    plt.plot(x, y_swish, label='Swish')
    plt.legend()
    plt.show()

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SwishFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.w3 = nn.Linear(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


