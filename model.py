import torch.nn as nn
import pytorch_lightning as pl

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense = nn.Linear(4,4)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense(x))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(4, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense(self.flatten(x)))