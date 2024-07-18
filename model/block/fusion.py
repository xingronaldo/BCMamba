import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmcv.ops import MultiScaleDeformableAttention
from .pos import ScaledSinuEmbedding
from .ffn import FeedForward


class Diff(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = FeedForward(channels)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x = torch.abs(x1 - x2)
        x_ = x.clone()
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.ffn_norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = x_ + self.ffn(x)
        return x


class CrossAttn(nn.Module):
    def __init__(self, channels):
        super(CrossAttn, self).__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.pos = ScaledSinuEmbedding(channels)
        self.crossattn = MultiScaleDeformableAttention(embed_dims=channels, num_levels=1, num_heads=4,
                                                       num_points=4, batch_first=True, dropout=0.)

    def get_deform_inputs(self, x1, x2):
        _, _, H1, W1 = x1.size()
        _, _, H2, W2 = x2.size()
        spatial_shapes = torch.as_tensor([(H2, W2)], dtype=torch.long, device=x2.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = get_reference_points([(H1, W1)], x1.device)

        return reference_points, spatial_shapes, level_start_index

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        reference_points, spatial_shapes, level_start_index = self.get_deform_inputs(x1, x2)
        x1 = rearrange(x1, 'b c h w -> b (h w) c')
        x2 = rearrange(x2, 'b c h w -> b (h w) c')
        query_pos = self.pos(x1)
        x = self.crossattn(query=self.norm1(x1), value=self.norm2(x2), identity=x2,
                           reference_points=reference_points, spatial_shapes=spatial_shapes,
                           level_start_index=level_start_index, query_pos=query_pos)

        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]

    return reference_points


