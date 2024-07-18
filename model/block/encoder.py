import torch
import torch.nn as nn
from .mamba import MambaBlock


class Encoder(nn.Module):
    def __init__(self, channels, depth=2):
        super(Encoder, self).__init__()
        self.network = nn.ModuleList()
        for _ in range(depth):
            self.network.append(MambaBlock(channels))

    def forward(self, x):
        for layer in self.network:
            x = layer(x)

        return x


