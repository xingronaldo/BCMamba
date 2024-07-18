import torch
import torch.nn as nn
from mamba_ssm import Mamba
from einops import rearrange
from .ffn import FeedForward
from .pos import ScaledSinuEmbedding


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba_type="v2"
            )

    def forward(self, x):
        x_mamba = self.mamba(x)
        return x_mamba


class MambaBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.ffn_norm = nn.LayerNorm(channels)
        self.pos = ScaledSinuEmbedding(channels)
        self.attn = MambaLayer(channels)
        self.ffn = FeedForward(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        pos = self.pos(x)
        x = x + self.attn(self.norm(x) + pos)
        #x = x + self.attn(self.norm(x))

        #x_ = x.clone()
        #x = self.ffn_norm(x)
        #x_ = rearrange(x_, 'b (h w) c -> b c h w', h=H, w=W)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        #x = x_ + self.ffn(x)
        return x

