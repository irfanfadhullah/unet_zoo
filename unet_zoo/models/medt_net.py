import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Union, Tuple

class qkv_transform(nn.Module):
    """
    Proper implementation of qkv_transform for MedT's axial attention.
    This performs a 1D convolution to generate Q, K, V from input features.
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, bias):
        super(qkv_transform, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        
    def forward(self, x):
        return self.conv(x)
    
    @property
    def weight(self):
        return self.conv.weight
    
    @property
    def bias(self):
        return self.conv.bias

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def create_qkv_transform(in_channels, out_channels):
    """Create a working qkv_transform layer."""
    return nn.Sequential(
        nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels * 3,  
            kernel_size=1,
            bias=False
        ),
        nn.BatchNorm1d(out_channels * 3)
    )

class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), 
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(
            self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(
            all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
        
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        
        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        similarity = F.softmax(stacked_similarity, dim=3)
        
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))

class AxialAttention_dynamic(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention_dynamic, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        self.f_qr = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.f_kr = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.f_sve = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.f_sv = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), 
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(
            self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(
            all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
        
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        qr = torch.mul(qr, self.f_qr)
        kr = torch.mul(kr, self.f_kr)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        similarity = F.softmax(stacked_similarity, dim=3)
        
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)

        sv = torch.mul(sv, self.f_sv)
        sve = torch.mul(sve, self.f_sve)

        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))

class AxialAttention_wopos(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention_wopos, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups)
        self.bn_output = nn.BatchNorm1d(out_planes * 1)

        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), 
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        stacked_similarity = self.bn_similarity(qk).reshape(N * W, 1, self.groups, H, H).sum(dim=1).contiguous()
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)

        sv = sv.reshape(N*W, self.out_planes * 1, H).contiguous()
        output = self.bn_output(sv).reshape(N, W, self.out_planes, 1, H).sum(dim=-2).contiguous()

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))

class AxialBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class AxialBlock_dynamic(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock_dynamic, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention_dynamic(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention_dynamic(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class AxialBlock_wopos(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock_wopos, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        
        self.conv_down = conv1x1(inplanes, width)
        self.conv1 = nn.Conv2d(width, width, kernel_size=1)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention_wopos(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention_wopos(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResAxialAttentionUNet(nn.Module):
    def __init__(self, block: nn.Module, layers: List[int], num_classes: int = 1, zero_init_residual: bool = True,
                 groups: int = 8, width_per_group: int = 64, replace_stride_with_dilation: List[bool] = None,
                 norm_layer: nn.Module = None, s: float = 0.125, img_size: int = 128, in_channels: int = 3):
        super(ResAxialAttentionUNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(64 * s)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        self.s = s
        self.img_size = img_size
        
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv2 = nn.Conv2d(self.inplanes, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(128)
        self.bn3 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size=(img_size//2))
        self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=(img_size//2),
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=(img_size//4),
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, int(1024 * s), layers[3], stride=2, kernel_size=(img_size//8),
                                       dilate=replace_stride_with_dilation[2])
        
        self.decoder1 = nn.Conv2d(int(1024 * block.expansion * s), int(512 * block.expansion * s), kernel_size=3, padding=1)
        self.decoder2 = nn.Conv2d(int(512 * block.expansion * s), int(256 * block.expansion * s), kernel_size=3, padding=1)
        self.decoder3 = nn.Conv2d(int(256 * block.expansion * s), int(128 * block.expansion * s), kernel_size=3, padding=1)
        self.decoder4 = nn.Conv2d(int(128 * block.expansion * s), int(64 * block.expansion * s), kernel_size=3, padding=1)
        
        self.final_conv = nn.Conv2d(int(64 * block.expansion * s), num_classes, kernel_size=1)

    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, 
                            norm_layer=norm_layer, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)  
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x1 = self.layer1(x)  
        x2 = self.layer2(x1)
        x3 = self.layer3(x2) 
        x4 = self.layer4(x3) 

        up1 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True) 
        up1 = self.decoder1(up1) 
        up1 = torch.add(up1, x3) 
        up1 = F.relu(up1)

        up2 = F.interpolate(up1, scale_factor=2, mode='bilinear', align_corners=True) 
        up2 = self.decoder2(up2) 
        up2 = torch.add(up2, x2) 
        up2 = F.relu(up2)

        up3 = F.interpolate(up2, scale_factor=2, mode='bilinear', align_corners=True) 
        up3 = self.decoder3(up3) 
        up3 = torch.add(up3, x1)
        up3 = F.relu(up3)

        up4 = F.interpolate(up3, scale_factor=2, mode='bilinear', align_corners=True)  
        up4 = self.decoder4(up4)
        up4 = F.relu(up4)

        output = self.final_conv(up4)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

class medt_net(nn.Module):
    def __init__(self, block: nn.Module, block_2: nn.Module, layers: List[int], num_classes: int = 1, zero_init_residual: bool = True,
                 groups: int = 8, width_per_group: int = 64, replace_stride_with_dilation: List[bool] = None,
                 norm_layer: nn.Module = None, s: float = 0.125, img_size: int = 128, in_channels: int = 3):
        super(medt_net, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(64 * s)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        self.s = s
        self.img_size = img_size
        self.img_size_p = img_size // 4
        
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv2 = nn.Conv2d(self.inplanes, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(128)
        self.bn3 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size=(img_size//2))
        self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=(img_size//2),
                                       dilate=replace_stride_with_dilation[0])
        
        self.decoder4 = nn.Conv2d(int(256 * block.expansion * s), int(128 * block.expansion * s), kernel_size=3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(int(128 * block.expansion * s), int(64 * block.expansion * s), kernel_size=3, stride=1, padding=1)
        
        self.conv1_p = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv2_p = nn.Conv2d(self.inplanes, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_p = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_p = norm_layer(self.inplanes)
        self.bn2_p = norm_layer(128)
        self.bn3_p = norm_layer(self.inplanes)
        self.relu_p = nn.ReLU(inplace=True)

        self.layer1_p = self._make_layer(block_2, int(128 * s), layers[0], kernel_size=(self.img_size_p//2))
        self.layer2_p = self._make_layer(block_2, int(256 * s), layers[1], stride=2, kernel_size=(self.img_size_p//2),
                                       dilate=replace_stride_with_dilation[0])
        self.layer3_p = self._make_layer(block_2, int(512 * s), layers[2], stride=2, kernel_size=(self.img_size_p//4),
                                       dilate=replace_stride_with_dilation[1])
        self.layer4_p = self._make_layer(block_2, int(1024 * s), layers[3], stride=2, kernel_size=(self.img_size_p//8),
                                       dilate=replace_stride_with_dilation[2])
        
        self.decoder1_p = nn.Conv2d(int(1024 * block_2.expansion * s), int(1024 * block_2.expansion * s), kernel_size=3, stride=2, padding=1)
        self.decoder2_p = nn.Conv2d(int(1024 * block_2.expansion * s), int(512 * block_2.expansion * s), kernel_size=3, stride=1, padding=1)
        self.decoder3_p = nn.Conv2d(int(512 * block_2.expansion * s), int(256 * block_2.expansion * s), kernel_size=3, stride=1, padding=1)
        self.decoder4_p = nn.Conv2d(int(256 * block_2.expansion * s), int(128 * block_2.expansion * s), kernel_size=3, stride=1, padding=1)
        self.decoder5_p = nn.Conv2d(int(128 * block_2.expansion * s), int(64 * block_2.expansion * s), kernel_size=3, stride=1, padding=1)

        self.decoderf = nn.Conv2d(int(64 * block.expansion * s), int(64 * block.expansion * s), kernel_size=3, stride=1, padding=1)
        self.adjust = nn.Conv2d(int(64 * block.expansion * s), num_classes, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, 
                            norm_layer=norm_layer, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        xin = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x1_full = self.layer1(x)
        x2_full = self.layer2(x1_full)

        x_full_decode = F.relu(F.interpolate(self.decoder4(x2_full), scale_factor=(2,2), mode='bilinear', align_corners=True))
        x_full_decode = torch.add(x_full_decode, x1_full)
        x_full_decode = F.relu(F.interpolate(self.decoder5(x_full_decode), scale_factor=(2,2), mode='bilinear', align_corners=True))
        
        x_loc = torch.zeros_like(x_full_decode)
        
        num_patches_h = xin.shape[2] // 32
        num_patches_w = xin.shape[3] // 32

        for i in range(num_patches_h):
            for j in range(num_patches_w):
                x_p = xin[:,:,32*i:32*(i+1),32*j:32*(j+1)]
                
                x_p = self.conv1_p(x_p)
                x_p = self.bn1_p(x_p)
                x_p = self.relu_p(x_p)
                x_p = self.conv2_p(x_p)
                x_p = self.bn2_p(x_p)
                x_p = self.relu_p(x_p)
                x_p = self.conv3_p(x_p)
                x_p = self.bn3_p(x_p)
                x_p = self.relu_p(x_p)

                x1_p = self.layer1_p(x_p)
                x2_p = self.layer2_p(x1_p)
                x3_p = self.layer3_p(x2_p)
                x4_p = self.layer4_p(x3_p)
                
                x_p = F.relu(F.interpolate(self.decoder1_p(x4_p), scale_factor=(2,2), mode='bilinear', align_corners=True))
                x_p = torch.add(x_p, x4_p)
                x_p = F.relu(F.interpolate(self.decoder2_p(x_p), scale_factor=(2,2), mode='bilinear', align_corners=True))
                x_p = torch.add(x_p, x3_p)
                x_p = F.relu(F.interpolate(self.decoder3_p(x_p), scale_factor=(2,2), mode='bilinear', align_corners=True))
                x_p = torch.add(x_p, x2_p)
                x_p = F.relu(F.interpolate(self.decoder4_p(x_p), scale_factor=(2,2), mode='bilinear', align_corners=True))
                x_p = torch.add(x_p, x1_p)
                x_p = F.relu(F.interpolate(self.decoder5_p(x_p), scale_factor=(2,2), mode='bilinear', align_corners=True))
                
                x_loc[:,:,32*i:32*(i+1),32*j:32*(j+1)] = x_p

        x_fused = torch.add(x_full_decode, x_loc)
        x_fused = F.relu(self.decoderf(x_fused))
        
        logits = self.adjust(F.relu(x_fused))
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

def axialunet(pretrained=False, **kwargs):
    kwargs.pop('s', None)
    num_classes = kwargs.pop('num_classes', 1)
    img_size = kwargs.pop('img_size', 128)
    in_channels = kwargs.pop('in_channels', 3)
    
    for param in ['layers', 'groups', 'width_per_group', 'norm_layer', 
                  'zero_init_residual', 'replace_stride_with_dilation']:
        kwargs.pop(param, None)
    
    model = ResAxialAttentionUNet(
        AxialBlock, 
        [1, 2, 4, 1], 
        s=0.125,
        num_classes=num_classes,
        img_size=img_size,
        in_channels=in_channels,
        **kwargs
    )
    
    if pretrained:
        pass
    
    return model

def gated(pretrained=False, **kwargs):
    kwargs.pop('s', None)
    num_classes = kwargs.pop('num_classes', 1)
    img_size = kwargs.pop('img_size', 128)
    in_channels = kwargs.pop('in_channels', 3)
    
    for param in ['layers', 'groups', 'width_per_group']:
        kwargs.pop(param, None)
    
    model = ResAxialAttentionUNet(
        AxialBlock_dynamic, 
        [1, 2, 4, 1], 
        s=0.125,
        num_classes=num_classes,
        img_size=img_size,
        in_channels=in_channels,
        **kwargs
    )
    return model

def MedT(pretrained=False, **kwargs):
    kwargs.pop('s', None)
    num_classes = kwargs.pop('num_classes', 1)
    img_size = kwargs.pop('img_size', 128)
    in_channels = kwargs.pop('in_channels', 3)
    
    for param in ['layers', 'groups', 'width_per_group']:
        kwargs.pop(param, None)
    
    model = ResAxialAttentionUNet(
        AxialBlock_wopos, 
        [1, 2, 4, 1], 
        s=0.125,
        num_classes=num_classes,
        img_size=img_size,
        in_channels=in_channels,
        **kwargs
    )
    return model

def logo(pretrained=False, **kwargs):
    kwargs.pop('s', None)
    num_classes = kwargs.pop('num_classes', 1)
    img_size = kwargs.pop('img_size', 128)
    in_channels = kwargs.pop('in_channels', 3)
    
    for param in ['layers', 'groups', 'width_per_group']:
        kwargs.pop(param, None)
    
    model = ResAxialAttentionUNet(
        AxialBlock_dynamic, 
        [1, 2, 4, 1], 
        s=0.125,
        num_classes=num_classes,
        img_size=img_size,
        in_channels=in_channels,
        **kwargs
    )
    return model
