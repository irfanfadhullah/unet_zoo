# unet_zoo/models/resunet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the new modules from common_layers
from .common_layers import ResidualConv, UpsampleResUnet 

class ResUnet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1, filters: list = None):
        super(ResUnet, self).__init__()
        

        if filters is None:
            filters = [64, 128, 256, 512]

        if num_classes > 1:
            print(f"Warning: ResUnet output layer is set for 1 class by default. "
                  f"For {num_classes} classes, consider changing the final Conv2d output channel.")

            self.final_conv_out_channels = num_classes
        else:
            self.final_conv_out_channels = 1

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = UpsampleResUnet(filters[3], filters[2], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[2] + filters[2], filters[2], 1, 1)

        self.upsample_2 = UpsampleResUnet(filters[2], filters[1], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[1] + filters[1], filters[1], 1, 1)

        self.upsample_3 = UpsampleResUnet(filters[1], filters[0], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[0] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], self.final_conv_out_channels, 1, 1),

        )

    def forward(self, x):

        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)

        x4 = self.bridge(x3)

        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return output