import torch.nn as nn
import numpy as np


class AutoEncoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = input_size
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, self.input_size),
            nn.Sigmoid()
        )

    def forward(self, input_x):
        # encode
        input_x = self.encoder(input_x)
        input_x = self.decoder(input_x)
        predictions = input_x
        return predictions


def MedianFilter(data, stride):
    # 中值滤波
    h = data.shape[-1]
    out = data

    # 卷积的过程
    for x in range((np.floor(stride/2)).astype(int), h - (np.floor(stride/2)).astype(int)):
        out[x] = np.median(data[(x - np.floor(stride/2)).astype(int): (x + np.floor(stride/2) + 1).astype(int)])

    return out

def MeanFilter(data, stride):
    # 均值滤波
    h = data.shape[-1]
    out = data

    # 卷积的过程
    for x in range((np.floor(stride/2)).astype(int), h - (np.floor(stride/2)).astype(int)):
        out[x] = np.mean(data[(x - np.floor(stride/2)).astype(int): (x + np.floor(stride/2) + 1).astype(int)])

    return out

