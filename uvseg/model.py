"""Five-stream attention U-Net (UNet5C) for underwater bridge component segmentation.

The network takes a 5-channel input built from one RGB image:

    channel 0  red
    channel 1  green
    channel 2  blue
    channel 3  illumination  (CIELAB L*, see ``uvseg.data.build_input``)
    channel 4  depth         (Depth Anything, see ``uvseg.depth``)

Each channel is encoded by its own CBAM-attention encoder. The red, green and blue
encoders use 3x3, 5x5 and 7x7 kernels respectively. Encoder features from the five
streams are fused at every scale by ``FusionConv`` and decoded by a standard U-Net
decoder.

Attribute names are kept identical to the original research code so that checkpoints
trained with it load without any key remapping.

The CBAM block is adapted from https://github.com/Jongchan/attention-module (MIT License).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

# --------------------------------------------------------------------------------------
# CBAM: Convolutional Block Attention Module (Woo et al., ECCV 2018)
# --------------------------------------------------------------------------------------


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, relu=True, bn=False, bias=False):
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    return s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=("avg", "max")):
        super().__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                pooled = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            elif pool_type == "max":
                pooled = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            elif pool_type == "lp":
                pooled = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            elif pool_type == "lse":
                pooled = logsumexp_2d(x)
            else:
                raise ValueError(f"Unknown pool type: {pool_type}")
            channel_att_raw = self.mlp(pooled)
            channel_att_sum = channel_att_raw if channel_att_sum is None else channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        scale = torch.sigmoid(self.spatial(self.compress(x)))
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=("avg", "max"), no_spatial=False):
        super().__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x = self.ChannelGate(x)
        if not self.no_spatial:
            x = self.SpatialGate(x)
        return x


# --------------------------------------------------------------------------------------
# U-Net building blocks
# --------------------------------------------------------------------------------------


def _double_conv(in_channels, out_channels, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=False, padding="same"),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, bias=False, padding="same"),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class DoubleConvAtt(nn.Module):
    """Two (conv -> BN -> ReLU) layers followed by CBAM attention. Used in the encoders."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = _double_conv(in_channels, out_channels, kernel_size)
        self.cbam = CBAM(gate_channels=out_channels)

    def forward(self, x):
        return self.cbam(self.conv(x))


class DoubleConv(nn.Module):
    """Two (conv -> BN -> ReLU) layers. Used in the decoder."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = _double_conv(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return self.conv(x)


class FusionConv(nn.Module):
    """Fuses the five encoder streams by concatenation followed by a double convolution."""

    def __init__(self, in_channels, out_channels, num_streams=5):
        super().__init__()
        self.conv = _double_conv(num_streams * in_channels, out_channels, kernel_size=3)

    def forward(self, *streams):
        return self.conv(torch.cat(streams, dim=1))


# --------------------------------------------------------------------------------------
# UNet5C
# --------------------------------------------------------------------------------------


class UNet5C(nn.Module):
    """Five-stream CBAM U-Net.

    Args:
        out_channels: number of output channels (1 for binary segmentation; the model
            returns logits, apply ``torch.sigmoid`` for probabilities).
        features: encoder widths per level. Input height and width must be divisible by
            ``2 ** len(features)`` (16 for the default), e.g. 160 x 240.
    """

    #: Order of the input channels expected by ``forward``.
    INPUT_CHANNELS = ("red", "green", "blue", "illumination", "depth")

    def __init__(self, out_channels: int = 1, features: Sequence[int] = (64, 128, 256, 512)):
        super().__init__()
        features = list(features)
        self.ups = nn.ModuleList()
        self.downs_red = nn.ModuleList()
        self.downs_green = nn.ModuleList()
        self.downs_blue = nn.ModuleList()
        # Encoder for the illumination channel. The attribute keeps its original name
        # ("reflectance") so that existing checkpoints load unchanged.
        self.downs_reflectance = nn.ModuleList()
        self.downs_depth = nn.ModuleList()
        self.fusion = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        in_channels = 1  # every stream receives a single channel
        for feature in features:
            self.downs_red.append(DoubleConvAtt(in_channels, feature, kernel_size=3))
            self.downs_green.append(DoubleConvAtt(in_channels, feature, kernel_size=5))
            self.downs_blue.append(DoubleConvAtt(in_channels, feature, kernel_size=7))
            self.downs_reflectance.append(DoubleConvAtt(in_channels, feature, kernel_size=3))
            self.downs_depth.append(DoubleConvAtt(in_channels, feature, kernel_size=3))
            in_channels = feature

        # One fusion block per skip level, plus one for the bottleneck.
        for feature in features:
            self.fusion.append(FusionConv(feature, feature))
        self.fusion.append(FusionConv(features[-1] * 2, features[-1] * 2))

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature, kernel_size=3))

        # Note: the bottleneck is shared by all five streams.
        self.bottleneck = DoubleConvAtt(features[-1], features[-1] * 2, kernel_size=3)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def _encode(self, x, downs):
        skips = []
        for down in downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        return self.bottleneck(x), skips

    def forward(self, x):
        if x.shape[1] != 5:
            raise ValueError(f"UNet5C expects a 5-channel input {self.INPUT_CHANNELS}, got {x.shape[1]} channels")

        encoders = (self.downs_red, self.downs_green, self.downs_blue, self.downs_reflectance, self.downs_depth)
        bottlenecks, skips = [], []
        for c, downs in enumerate(encoders):
            b, s = self._encode(x[:, c:c + 1], downs)
            bottlenecks.append(b)
            skips.append(s)

        # Fuse the five streams at every skip level and at the bottleneck.
        fused_skips = [self.fusion[i](*(s[i] for s in skips)) for i in range(len(skips[0]))][::-1]
        x = self.fusion[-1](*bottlenecks)

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = fused_skips[idx // 2]
            if x.shape[2:] != skip.shape[2:]:  # only when H or W is not divisible by 16
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = self.ups[idx + 1](torch.cat((skip, x), dim=1))

        return self.final_conv(x)


# Backward-compatible name used in the original code.
UNET_5C = UNet5C


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
