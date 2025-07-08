# models/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .common_layers import DoubleConv, DownSample, UpSample_UNet, OutConv

class UNet(nn.Module):
    """
    The full UNet architecture for image segmentation.
    """
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()

        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_convolution_1 = UpSample_UNet(1024, 512)
        self.up_convolution_2 = UpSample_UNet(512, 256)
        self.up_convolution_3 = UpSample_UNet(256, 128)
        self.up_convolution_4 = UpSample_UNet(128, 64)

        self.out = OutConv(in_channels=64, out_channels=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
        return out