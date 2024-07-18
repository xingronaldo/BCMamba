import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, channels):
        super(FeedForward, self).__init__()

        self.project_in = nn.Conv2d(channels, channels, 1, bias=True)
        self.dwconv = nn.Conv2d(channels, channels, 3, stride=1, padding=1, groups=channels, bias=True)
        self.project_out = nn.Conv2d(channels // 2, channels, 1, bias=True)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.silu(x1) * x2
        x = self.project_out(x)
        return x

