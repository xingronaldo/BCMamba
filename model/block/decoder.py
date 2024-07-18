import torch
import torch.nn as nn
from .dsconv_pro import DSCBlock
from .mamba import MambaBlock


class BodyBranch(nn.Module):
    def __init__(self, channels, depth=2):
        super(BodyBranch, self).__init__()
        self.network = nn.ModuleList()
        for _ in range(depth):
            self.network.append(MambaBlock(channels))

    def forward(self, x):
        for layer in self.network:
            x = layer(x)

        return x


class BoundaryBranch(nn.Module):
    def __init__(self, channels, depth=2, kernel_size=9):
        super(BoundaryBranch, self).__init__()
        self.network = nn.ModuleList()
        for _ in range(depth):
            self.network.append(DSCBlock(channels, channels, kernel_size))

    def forward(self, x):
        for layer in self.network:
            x = layer(x)

        return x






