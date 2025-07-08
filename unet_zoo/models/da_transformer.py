import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
import copy
import numpy as np

from .common_layers import ConfigDict, DoubleConv

def get_da_transformer_config():
    config = ConfigDict()
    config.patches = ConfigDict({'size': (16, 16)})
    config.hidden_size = 768 
    config.transformer = ConfigDict() 
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None 
    config.pretrained_path = None
    config.patch_size = 16

    config.patches.grid = (16, 16) 
    config.resnet = ConfigDict()
    config.resnet.num_layers = (3, 4, 9) 
    config.resnet.width_factor = 1

    config.decoder_channels = (256, 128, 64, 16) 
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.n_skip = 3 
    config.activation = 'softmax'

    return config

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)

def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)

def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)

class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block."""

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False) 
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x) 
        features.append(x) 
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x) 

        e1 = self.body[0](x) 
        e2 = self.body[1](e1) 
        e3 = self.body[2](e2) 
        return e3, [e3, e2, e1, x] 

class DA_PAM_Module(nn.Module):
    """ Position attention module (for DA_Transformer)"""
    def __init__(self, in_dim, attention_resolution=(64, 64)): 
        super(DA_PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.attention_resolution = attention_resolution 

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1) 
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()

        q_feats = self.query_conv(x)
        k_feats = self.key_conv(x)
        v_feats = self.value_conv(x)

        q_attn = F.adaptive_avg_pool2d(q_feats, self.attention_resolution)
        k_attn = F.adaptive_avg_pool2d(k_feats, self.attention_resolution)
        v_attn = F.adaptive_avg_pool2d(v_feats, self.attention_resolution)

        proj_query = q_attn.view(m_batchsize, -1, self.attention_resolution[0]*self.attention_resolution[1]).permute(0, 2, 1)
        proj_key = k_attn.view(m_batchsize, -1, self.attention_resolution[0]*self.attention_resolution[1])
        proj_value = v_attn.view(m_batchsize, -1, self.attention_resolution[0]*self.attention_resolution[1])

        energy = torch.bmm(proj_query, proj_key) 
        attention = self.softmax(energy)

        out_attn_low_res = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out_attn_low_res = out_attn_low_res.view(m_batchsize, C, self.attention_resolution[0], self.attention_resolution[1])

        out = F.interpolate(out_attn_low_res, size=(height, width), mode='bilinear', align_corners=True)

        out = self.gamma * out + x
        return out

class DA_CAM_Module(nn.Module):
    """ Channel attention module (for DA_Transformer)"""
    def __init__(self, in_dim):
        super(DA_CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

def norm(planes, mode='bn', groups=16):
    if mode == 'bn':
        return nn.BatchNorm2d(planes, momentum=0.95, eps=1e-03)
    elif mode == 'gn':
        return nn.GroupNorm(groups, planes)
    else:
        return nn.Sequential()

class DANetHead(nn.Module):
    """
    Original DANetHead for classification. Not directly used as feature processor in my DA_Transformer
    but provided in the snippet.
    """
    def __init__(self, in_channels, out_channels):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 16

        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        self.sa = DA_PAM_Module(inter_channels)
        self.sc = DA_CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.05, False), nn.Conv2d(inter_channels, out_channels, 1),
                                   nn.ReLU())
        self.conv7 = nn.Sequential(nn.Dropout2d(0.05, False), nn.Conv2d(inter_channels, out_channels, 1),
                                   nn.ReLU())

        self.conv8 = nn.Sequential(nn.Dropout2d(0.05, False), nn.Conv2d(inter_channels, out_channels, 1),
                                   nn.ReLU())
        

    def forward(self, x):
        
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)
        
        return sasc_output
    
class UpSample_DA(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, 
            in_channels // 2, 
            kernel_size=2, 
            stride=2
        )
        self.skip_conv = nn.Conv2d(
            skip_channels, 
            in_channels // 2, 
            kernel_size=1
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.skip_conv(x2) 
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2
        ])
        
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

class DA_Transformer(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, config: ConfigDict, **kwargs):
        super().__init__()
        self.resnet = ResNetV2(
            block_units=config.resnet.num_layers,
            width_factor=config.resnet.width_factor
        )
        self.bottleneck = DoubleConv(1024, 1024)

        self.up_block1 = UpSample_DA(1024, 512, skip_channels=1024)
        self.pam1 = DA_PAM_Module(512, attention_resolution=(64, 64)) 
        self.cam1 = DA_CAM_Module(512)
        
        self.up_block2 = UpSample_DA(512, 256, skip_channels=512)
        self.pam2 = DA_PAM_Module(256, attention_resolution=(64, 64)) 
        self.cam2 = DA_CAM_Module(256)
        
        self.up_block3 = UpSample_DA(256, 128, skip_channels=256)
        self.pam3 = DA_PAM_Module(128, attention_resolution=(32, 32)) 
        self.cam3 = DA_CAM_Module(128)
        
        self.up_block4 = UpSample_DA(128, 64, skip_channels=64)
        
        self.up_block5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up_block6 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.final_upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)
        self.outc = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_deepest, skips = self.resnet(x)
        bottleneck_out = self.bottleneck(x_deepest)
        
        up1 = self.up_block1(bottleneck_out, skips[0])
        up1 = self.pam1(up1)
        up1 = self.cam1(up1)
        
        up2 = self.up_block2(up1, skips[1])
        up2 = self.pam2(up2)
        up2 = self.cam2(up2)
        
        up3 = self.up_block3(up2, skips[2])
        up3 = self.pam3(up3)
        up3 = self.cam3(up3)
        
        up4 = self.up_block4(up3, skips[3])
        up5 = self.up_block5(up4)  
        up6 = self.up_block6(up5)  
        
        up7 = self.final_upsample(up6) 
        return self.outc(up7)