import warnings

from torch import nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d, Conv2d

from .doppler_attention import DopplerGuidedLocalAttention

class SECOND(nn.Module):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
    """

    def __init__(self,
                 in_channels=128,
                 out_channels=[128, 128, 256],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 use_doppler_attention=False,
                 doppler_attention_stages=None,
                 doppler_attn_channels=64,
                 doppler_attn_kernel=3,
                 doppler_lambda=2.0,
                 doppler_sigma=0.5,
                 doppler_reliability_sigma=1.0):
        super().__init__()
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        self.use_doppler_attention = use_doppler_attention
        if doppler_attention_stages is None:
            doppler_attention_stages = [True] * len(layer_nums)
        assert len(doppler_attention_stages) == len(layer_nums)
        self.doppler_attention_stages = doppler_attention_stages

        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            block = [
                Conv2d(in_filters[i],
                       out_channels[i],
                       3,
                       stride=layer_strides[i],
                       padding=1, 
                       bias=False),
                BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ]
            for j in range(layer_num):
                block.append(
                    Conv2d(out_channels[i],
                       out_channels[i],
                       3,
                       padding=1, 
                       bias=False),)
                block.append(BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
                block.append(nn.ReLU(inplace=True))

            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

        if self.use_doppler_attention:
            attn_layers = []
            for i in range(len(layer_nums)):
                if self.doppler_attention_stages[i]:
                    attn_layers.append(
                        DopplerGuidedLocalAttention(
                            channels=out_channels[i],
                            attn_channels=doppler_attn_channels,
                            kernel_size=doppler_attn_kernel,
                            lambda_doppler=doppler_lambda,
                            sigma_doppler=doppler_sigma,
                            sigma_reliability=doppler_reliability_sigma))
                else:
                    attn_layers.append(nn.Identity())
            self.doppler_attn_layers = nn.ModuleList(attn_layers)
        else:
            self.doppler_attn_layers = None

    def forward(self, x, doppler_mean_map=None, doppler_std_map=None):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            if self.use_doppler_attention and doppler_mean_map is not None:
                doppler_mean_stage = F.interpolate(
                    doppler_mean_map,
                    size=x.shape[-2:],
                    mode='nearest')
                doppler_std_stage = None
                if doppler_std_map is not None:
                    doppler_std_stage = F.interpolate(
                        doppler_std_map,
                        size=x.shape[-2:],
                        mode='nearest')
                x = self.doppler_attn_layers[i](
                    x,
                    doppler_mean=doppler_mean_stage,
                    doppler_std=doppler_std_stage)

            outs.append(x)
        return tuple(outs)