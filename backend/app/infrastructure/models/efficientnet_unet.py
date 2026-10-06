"""
EfficientNet-Enhanced U-Net Architecture for Brain Tumor Segmentation
Inspired by Tiwary et al. (2025): "Deep Learning-Based MRI Brain Tumor Segmentation
With EfficientNet-Enhanced UNet", IEEE Access.

Encoder: EfficientNet-B0 feature extractor stages
Skip Connections: Retain multi-scale spatial boundaries
Bottleneck: Deepest MBConv representations
Decoder: Transposed convolutions with double-convolution refinement
Output: 1-channel binary tumor segmentation map
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from typing import Optional

class ConvBlock(nn.Module):
    """Double 3x3 convolution block with BatchNorm and ReLU."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class DecoderBlock(nn.Module):
    """
    Decoder upsampling block:
    Transposed convolution (stride 2) + concatenation with skip connection + double conv.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class EfficientNetUNet(nn.Module):
    """
    EfficientNet-B0 Backbone U-Net.
    Captures hierarchical MRI features while preserving tumor boundary delineation.
    """
    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        f = backbone.features

        # Encoder stages
        self.enc0 = nn.Sequential(f[0], f[1])       # 16 channels,  112x112
        self.enc1 = f[2]                            # 24 channels,  56x56
        self.enc2 = f[3]                            # 40 channels,  28x28
        self.enc3 = nn.Sequential(f[4], f[5])       # 112 channels, 14x14
        self.bottleneck = nn.Sequential(f[6], f[7], f[8]) # 1280 channels, 7x7

        # Decoder stages
        self.dec3 = DecoderBlock(1280, 112, 256)    # 7x7   -> 14x14
        self.dec2 = DecoderBlock(256, 40, 128)      # 14x14 -> 28x28
        self.dec1 = DecoderBlock(128, 24, 64)       # 28x28 -> 56x56
        self.dec0 = DecoderBlock(64, 16, 32)        # 56x56 -> 112x112

        # Final upsample from 112x112 to 224x224
        self.final_up = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_refine = ConvBlock(16, 16)
        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder contracting path
        s0 = self.enc0(x)          # (B, 16, 112, 112)
        s1 = self.enc1(s0)         # (B, 24, 56, 56)
        s2 = self.enc2(s1)         # (B, 40, 28, 28)
        s3 = self.enc3(s2)         # (B, 112, 14, 14)
        bn = self.bottleneck(s3)   # (B, 1280, 7, 7)

        # Decoder expanding path with skip connections
        d3 = self.dec3(bn, s3)     # (B, 256, 14, 14)
        d2 = self.dec2(d3, s2)     # (B, 128, 28, 28)
        d1 = self.dec1(d2, s1)     # (B, 64, 56, 56)
        d0 = self.dec0(d1, s0)     # (B, 32, 112, 112)

        out = self.final_up(d0)    # (B, 16, 224, 224)
        out = self.final_refine(out)
        logits = self.classifier(out) # (B, 1, 224, 224)
        return logits

def build_efficientnet_unet(num_classes: int = 1, pretrained: bool = True) -> nn.Module:
    return EfficientNetUNet(num_classes=num_classes, pretrained=pretrained)
