import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict

class Up(nn.Module):
    """Upscaling"""

    def __init__(self, in_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x1):
        x1 = self.up(x1)

        return x1

def concatenate_layers(x1, x2):
    # input is CHW
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])

    x = torch.cat([x2, x1], dim=1)
    return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DAFTBlock(nn.Module):
    # Block for ZeCatNet
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_momentum: float = 0.1,
        stride: int = 2,
        ndim_non_img: int = 4, #4 quadrants, change to 6 for 6-regions
        location: int = 0,
        activation: str = "linear",
        scale: bool = True,
        shift: bool = True,
        bottleneck_dim: int = 512,
    ) -> None:

        super(DAFTBlock, self).__init__()

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.film_dims = 0
        if location in {0, 1, 2}:
            self.film_dims = in_channels

        self.bottleneck_dim = bottleneck_dim
        aux_input_dims = self.film_dims
        # shift and scale decoding
        self.split_size = 0
        if scale and shift:  
            self.split_size = self.film_dims #self.split_size = 1024
            self.scale = None
            self.shift = None
            self.film_dims = 2 * self.film_dims #film_dims = 1024*2
        elif not scale:
            self.scale = 1
            self.shift = None
        elif not shift:
            self.shift = 0
            self.scale = None

        # create aux net
        layers = [
            ("#aux_base1", nn.Linear(ndim_non_img + aux_input_dims, self.bottleneck_dim, bias=False)),
            ("aux_relu1", nn.ReLU()),
            ("aux_out", nn.Linear(self.bottleneck_dim, self.film_dims, bias=False)), #self.film_dims = 1024*2
        ]
        self.aux = nn.Sequential(OrderedDict(layers))
        self.scale_activation = nn.Sigmoid()

    def forward(self, feature_map, x_aux):

        #feature_map.shape = (8, 1024, 14, 14), x_aux.shape = (8, 1, 4)

        squeeze = self.global_pool(feature_map)
        #squeeze.shape = (8, 1024, 1, 1)

        squeeze = squeeze.view(squeeze.size(0), -1)
        x_aux = x_aux.view(x_aux.size(0), -1)
        #squeeze.shape = (8, 1024), x_aux.shape = (8, 4)

        squeeze = torch.cat((squeeze, x_aux), dim=1)
        #squeeze.shape = (8, 1024 + 4)

        attention = self.aux(squeeze)
        #attention.shape = (8, 2*1024)

        if self.scale == self.shift:
            v_scale, v_shift = torch.split(attention, self.split_size, dim=1)
            #v_scale.shape = (8, 1024), v_shift.shape = (8, 1024)
            
            v_scale = v_scale.view(*v_scale.size(), 1, 1).expand_as(feature_map)
            v_shift = v_shift.view(*v_shift.size(), 1, 1).expand_as(feature_map)
            #v_scale.shape = (8, 1024, 14, 14), v_shift.shape = (8, 1024, 14, 14)

            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.scale is None:
            v_scale = attention
            v_scale = v_scale.view(*v_scale.size(), 1, 1).expand_as(feature_map)
            v_shift = self.shift
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.shift is None:
            v_scale = self.scale
            v_shift = attention
            v_shift = v_shift.view(*v_shift.size(), 1, 1).expand_as(feature_map)
        else:
            raise AssertionError(
                f"Sanity checking on scale and shift failed. Must be of type bool or None: {self.scale}, {self.shift}"
            )

        return (v_scale * feature_map) + v_shift
