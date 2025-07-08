from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    r""" ConvNeXt Block based structure.
    Used in MMUNet encoder and decoder.
    """

    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv1 = nn.Conv2d(dim // 4, dim // 4, kernel_size=3, padding=1, groups=dim // 4)
        self.act1 = nn.GELU()
        self.norm1 = nn.BatchNorm2d(dim // 4)
        self.dwconv2 = nn.Conv2d(dim // 4, dim // 4, kernel_size=5, padding=2, groups=dim // 4)
        self.norm2 = nn.BatchNorm2d(dim // 4)
        self.act2 = nn.GELU()
        self.dwconv3 = nn.Conv2d(dim // 4, dim // 4, kernel_size=7, padding=3, groups=dim // 4)
        self.norm3 = nn.BatchNorm2d(dim // 4)
        self.act3 = nn.GELU()

        self.norm4 = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, int(4 * dim)) 
        self.act4 = nn.GELU()
        self.pwconv2 = nn.Linear(int(4 * dim), dim)

        self.width = dim // 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x 
        

        x_chunks = torch.split(x, self.width, 1)
        x1, x2, x3, x4 = x_chunks[0], x_chunks[1], x_chunks[2], x_chunks[3]

        x1 = self.dwconv1(x1)
        x1 = self.norm1(x1)
        x1 = self.act1(x1)
        
        x2 = self.dwconv2(x1 + x2)
        x2 = self.norm2(x2)
        x2 = self.act2(x2)
        
        x3 = self.dwconv3(x2 + x3) 
        x3 = self.norm3(x3)
        x3 = self.act3(x3)
        
        x = torch.cat((x1, x2, x3, x4), dim=1) 
        x = self.norm4(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act4(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        
        x = shortcut + x

        return x

class Block1(nn.Module):
    r""" ConvNeXt Block based structure with added External Attention mechanism (EA) logic.
    Used in MMUNet encoder and decoder.
    """

    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv1 = nn.Conv2d(dim // 4, dim // 4, kernel_size=3, padding=1, groups=dim // 4)
        self.act1 = nn.GELU()
        self.norm1 = nn.BatchNorm2d(dim // 4)
        self.dwconv2 = nn.Conv2d(dim // 4, dim // 4, kernel_size=5, padding=2, groups=dim // 4)
        self.norm2 = nn.BatchNorm2d(dim // 4)
        self.act2 = nn.GELU()
        self.dwconv3 = nn.Conv2d(dim // 4, dim // 4, kernel_size=7, padding=3, groups=dim // 4)
        self.norm3 = nn.BatchNorm2d(dim // 4)
        self.act3 = nn.GELU()

        self.norm4 = nn.BatchNorm2d(dim)

        self.pwconv1 = nn.Linear(dim, int(4 * dim))
        self.act4 = nn.GELU()
        self.pwconv2 = nn.Linear(int(4 * dim), dim)

        self.width = dim // 4
        self.norm_ea = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.k = 64
        self.linear_0 = nn.Conv1d(dim, self.k, 1, bias=False)
        self.linear_1 = nn.Conv1d(self.k, dim, 1, bias=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x_chunks = torch.split(x, self.width, 1)
        x1, x2, x3, x4 = x_chunks[0], x_chunks[1], x_chunks[2], x_chunks[3]
        x1 = self.dwconv1(x1)
        x1 = self.norm1(x1)
        x1 = self.act1(x1)
        x2 = self.dwconv2(x1 + x2)
        x2 = self.norm2(x2)
        x2 = self.act2(x2)
        x3 = self.dwconv3(x2 + x3)
        x3 = self.norm3(x3)
        x3 = self.act3(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.norm4(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act4(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        x = shortcut + x
        shortcut1 = x
        x = self.norm_ea(x)
        x2_conv1 = self.conv1(x)
        b, c, h, w = x2_conv1.size()
        n = h * w
        x2_conv1 = x2_conv1.view(b, c, n)

        attn = self.linear_0(x2_conv1)
        attn = F.softmax(attn, dim=-1)
        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))

        x2_conv1 = self.linear_1(attn)
        x2_conv1 = x2_conv1.view(b, c, h, w)
        x2_conv2 = self.conv2(x2_conv1)
        x2 = shortcut1 + x2_conv2
        x = F.gelu(x2)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, layer_scale_init_value: float = 1e-6, use_erode=False):
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                Block1(dim=out_channels, drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),
                Block1(dim=out_channels, drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=4, stride=4)
        self.softmax = nn.Softmax(dim=1)
        self.maxpool = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        self.softmax1 = nn.Softmax(dim=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        self.linear1 = nn.Conv2d(in_channels//2, in_channels//2, kernel_size=1)
        shortcut_channels = in_channels // 2
        self.mlp = Mlp(
            in_features=shortcut_channels,
            hidden_features=shortcut_channels,
            out_features=shortcut_channels // 2
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x3 = x1 + x2
        x3_short = self.mlp(x3)
        x2_erode = -self.maxpool(self.maxpool(-self.softmax(x2)))
        x2_dilate = self.maxpool1(self.maxpool1(self.softmax1(x2)))
        x2_processed = torch.sigmoid(self.linear1(x2_erode + x2)) * x2 + torch.sigmoid(x2_erode) * torch.tanh(x2_dilate)
        
        x = torch.cat([x2_processed, x1], dim=1)
        x = self.conv(x) + x3_short

        return x

class Mlp(nn.Module):
    """
    Simple MLP using 1x1 convolutions for feature transformation.
    Used in MMUNet's Up block.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features,kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features,kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Up1(nn.Module):
    """
    MMUNet Decoder Up-block (variant 1) with erosion/dilation logic.
    """
    def __init__(self, in_channels, out_channels, bilinear=True, layer_scale_init_value: float = 1e-6, use_erode=False):
        super(Up1, self).__init__()
        
        if bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels , kernel_size=1),
                nn.BatchNorm2d(out_channels ),
                Block(dim=out_channels , drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),
                Block(dim=out_channels, drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=4, stride=4)
        self.softmax = nn.Softmax(dim=1)
        self.maxpool = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        self.softmax1 = nn.Softmax(dim=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)

        self.linear1=nn.Conv2d(in_channels//2,in_channels//2,kernel_size=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x2_erode = -self.maxpool(self.maxpool(-self.softmax(x2)))
        x2_dilate = self.maxpool1(self.maxpool1(self.softmax1(x2)))
        x2_processed = torch.sigmoid(self.linear1(x2_erode + x2)) * x2 + torch.sigmoid(x2_erode) * torch.tanh(x2_dilate)
        
        x = torch.cat([x2_processed,  x1], dim=1)
        x = self.conv(x)
        return x

class Up2(nn.Module):
    """
    MMUNet Decoder Up-block (variant 2) without explicit skip connection fusion logic in forward.
    This is effectively just an upsample followed by two Blocks.
    """
    def __init__(self, in_channels, out_channels, bilinear=True, layer_scale_init_value: float = 1e-6, use_erode=False):
        super(Up2, self).__init__()
        
        if bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            self.conv = nn.Sequential(
                Block(dim=out_channels , drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),
                Block(dim=out_channels, drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=4, stride=4)
            
    def forward(self, x1: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x = self.conv(x1)
        return x

class OutConv(nn.Sequential):
    """
    Final output convolution layer for MMUNet.
    """
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class EFM(nn.Module):
    """
    Edge Feature Module (EFM) for MMUNet.
    Combines edge features from different resolutions.
    """
    def __init__(self, in_dim, out_dim):
        super(EFM, self).__init__()
        self.up_x2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                   nn.Conv2d(in_channels=in_dim, out_channels=out_dim,kernel_size=3,bias=False,padding=1,groups=out_dim),
                                   nn.BatchNorm2d(out_dim),
                                   nn.GELU()
                                   )
        self.linear1 = nn.Conv2d(2 * out_dim, out_dim, kernel_size=1)
        self.maxpool1=nn.MaxPool2d(kernel_size=7,stride=1,padding=3)
        self.softmax=nn.Softmax(dim=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
         x2_upsampled = self.up_x2(x2)
         x1_dilate = self.maxpool1(self.softmax(x1))
         x1_erode = -self.maxpool1(-self.softmax(x1))
         x1_edge = x1_dilate - x1_erode
         x2_dilate = self.maxpool1(self.softmax(x2_upsampled))
         x2_erode = -self.maxpool1(-self.softmax(x2_upsampled))
         x2_edge = x2_dilate - x2_erode
         new_edge=self.linear1(torch.cat((x2_edge,x1_edge),dim=1))
         x3 = x3 + new_edge

         return x3

class MMUNet(nn.Module):
    """
    MMUNet: Multi-Modal UNet for segmentation.
    Integrates attention blocks, multi-scale feature fusion, and edge feature enhancement.
    Outputs raw logits for segmentation.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1, bilinear: bool = True, base_channels: int = 96,
                 layer_scale_init_value: float = 1e-6, se_ratio: float = 0.25):
        super(MMUNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.first_down = nn.Sequential(
            nn.Conv2d(in_channels, int(base_channels), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(base_channels)),
            Block(dim=int(base_channels), drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),
            nn.BatchNorm2d(base_channels),
            Block(dim=base_channels, drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),
            nn.GELU()
        )

        self.down0 = nn.Sequential(
            nn.Conv2d(base_channels, int(base_channels * 2), kernel_size=2, stride=2),
            nn.BatchNorm2d(int(base_channels * 2)),
            Block(dim=int(base_channels * 2), drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),
            nn.BatchNorm2d(2 * base_channels),
            Block(dim=int(base_channels * 2), drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),
            nn.GELU()
        )
        self.down0_1 = nn.Sequential(
            nn.Conv2d(2 * base_channels, 2 * base_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(2 * base_channels),
            Block(dim=int(base_channels * 2), drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),
            nn.BatchNorm2d(2 * base_channels),
            Block(dim=base_channels * 2, drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),
            nn.GELU()
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels * 4),
            Block(dim=base_channels * 4, drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),
            nn.BatchNorm2d(4 * base_channels),
            Block(dim=base_channels * 4, drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),
            nn.GELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels * 8),
            Block1(dim=base_channels * 8, drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),
            nn.BatchNorm2d(8 * base_channels),
            Block1(dim=base_channels * 8, drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),
           nn.GELU()
        )
        
        factor = 2 if bilinear else 1
        self.down3 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 16 // factor, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels * 16 // factor),
            Block1(dim=base_channels * 16 // factor, drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),
            nn.BatchNorm2d(base_channels * 16 // factor),
            Block1(dim=base_channels * 16 // factor, drop_rate=0.0, layer_scale_init_value=layer_scale_init_value),
            nn.GELU()
        )
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up1(base_channels * 4, base_channels * 2, bilinear)
        self.up4 = Up1(base_channels * 4, base_channels, bilinear)
        self.up5 = Up2(base_channels , base_channels, bilinear)
        self.eam=EFM(base_channels*2,base_channels)

        self.out_conv = OutConv(base_channels, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.first_down(x)
        x2 = self.down0(x1)
        x3 = self.down0_1(x2)
        x4 = self.down1(x3)
        x5 = self.down2(x4)
        x6 = self.down3(x5)
        x_up = self.up1(x6, x5)
        x_up = self.up2(x_up, x4)
        x_up = self.up3(x_up, x3)
        x_up = self.up4(x_up, x2)
        x_up = self.up5(x_up)
        x_fused = self.eam(x1, x2, x_up)

        logits = self.out_conv(x_fused)

        return {"out": logits}