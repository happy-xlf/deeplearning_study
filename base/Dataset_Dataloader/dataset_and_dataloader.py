#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   dataset_and_dataloader.py
@Time    :   2024/10/14 16:52:10
@Author  :   Lifeng Xu 
@desc :   
'''
import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self):
        self.data = torch.arange(0, 20)

    def __getitem__(self, index):
        x = self.data[index]
        y = x * 2
        return y

    def __len__(self):
        return len(self.data)

dataset = MyDataset()
print(len(dataset))
print(dataset[3])

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
print(next(iter(dataloader)))
for i, data in enumerate(dataloader):
    print(i, data)



