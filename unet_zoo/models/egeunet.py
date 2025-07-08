import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange 

from timm.models.layers import trunc_normal_ 
import math
from typing import List, Dict, Union, Tuple

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, 
                      stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))

class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
    

class group_aggregation_bridge(nn.Module):
    def __init__(self, dim_xh, dim_xl, k_size=3, d_list=[1,2,5,7]):
        super().__init__()
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        
        group_size = dim_xl // 4 
        input_channels = 2 * group_size + 1 
        
        self.g0 = nn.Sequential(
            LayerNorm(normalized_shape=input_channels, data_format='channels_first'), 
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, 
                      padding=(k_size+(k_size-1)*(d_list[0]-1))//2, 
                      dilation=d_list[0], groups=input_channels)
        )
        self.g1 = nn.Sequential(
            LayerNorm(normalized_shape=input_channels, data_format='channels_first'), 
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, 
                      padding=(k_size+(k_size-1)*(d_list[1]-1))//2, 
                      dilation=d_list[1], groups=input_channels)
        )
        self.g2 = nn.Sequential(
            LayerNorm(normalized_shape=input_channels, data_format='channels_first'), 
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, 
                      padding=(k_size+(k_size-1)*(d_list[2]-1))//2, 
                      dilation=d_list[2], groups=input_channels)
        )
        self.g3 = nn.Sequential(
            LayerNorm(normalized_shape=input_channels, data_format='channels_first'),
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, 
                      padding=(k_size+(k_size-1)*(d_list[3]-1))//2, 
                      dilation=d_list[3], groups=input_channels)
        )
        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=4 * input_channels, data_format='channels_first'),
            nn.Conv2d(4 * input_channels, dim_xl, 1)
        )

    def forward(self, xh, xl, mask):
        xh = self.pre_project(xh)
        xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode='bilinear', align_corners=True)
        
        xh_chunks = torch.chunk(xh, 4, dim=1) 
        xl_chunks = torch.chunk(xl, 4, dim=1)
        
        x0 = self.g0(torch.cat((xh_chunks[0], xl_chunks[0], mask), dim=1)) 
        x1 = self.g1(torch.cat((xh_chunks[1], xl_chunks[1], mask), dim=1))
        x2 = self.g2(torch.cat((xh_chunks[2], xl_chunks[2], mask), dim=1))
        x3 = self.g3(torch.cat((xh_chunks[3], xl_chunks[3], mask), dim=1))
        
        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.tail_conv(x)
        return x

class Grouped_multi_axis_Hadamard_Product_Attention(nn.Module):
    def __init__(self, dim_in, dim_out, x_res=8, y_res=8):
        
        c_dim_in = dim_in//4 
        k_size=3
        pad=(k_size-1) // 2
        
        self.params_xy = nn.Parameter(torch.Tensor(1, c_dim_in, x_res, y_res), requires_grad=True)
        nn.init.ones_(self.params_xy)
        self.conv_xy = nn.Sequential(nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv2d(c_dim_in, c_dim_in, 1))

        self.params_zx = nn.Parameter(torch.Tensor(1, 1, c_dim_in, x_res), requires_grad=True) 
        nn.init.ones_(self.params_zx)
        self.conv_zx = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.params_zy = nn.Parameter(torch.Tensor(1, 1, c_dim_in, y_res), requires_grad=True)
        nn.init.ones_(self.params_zy)
        self.conv_zy = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.dw = nn.Sequential(
                nn.Conv2d(c_dim_in, c_dim_in, 1),
                nn.GELU(),
                nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in) 
        )
        
        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        
        self.ldw = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
                nn.GELU(),
                nn.Conv2d(dim_in, dim_out, 1),
        )
        
    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        B, C_chunk, H, W = x1.size()

        params_xy_interpolated = F.interpolate(self.params_xy, size=(H, W), mode='bilinear', align_corners=True)
        x1 = x1 * self.conv_xy(params_xy_interpolated)
        
        x2_permuted = x2.permute(0, 3, 1, 2)
        params_zx_interpolated = F.interpolate(self.params_zx, size=(C_chunk, H), mode='bilinear', align_corners=True).squeeze(0)
        x2_attended = x2_permuted * self.conv_zx(params_zx_interpolated).unsqueeze(0)
        x2 = x2_attended.permute(0, 2, 3, 1) 
        
        x3_permuted = x3.permute(0, 2, 1, 3)
        params_zy_interpolated = F.interpolate(self.params_zy, size=(C_chunk, W), mode='bilinear', align_corners=True).squeeze(0)
        x3_attended = x3_permuted * self.conv_zy(params_zy_interpolated).unsqueeze(0)
        x3 = x3_attended.permute(0, 2, 1, 3)
        
        x4 = self.dw(x4)
        
        x = torch.cat([x1,x2,x3,x4],dim=1)
        
        x = self.norm2(x)
        x = self.ldw(x)
        return x

class EGEUNet(nn.Module):
    """
    EGEUNet model for image segmentation.
    Supports deep supervision via `gt_ds` parameter.
    Outputs raw logits, not sigmoid probabilities.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1, c_list: List[int] = None, bridge: bool = True, gt_ds: bool = True, image_size: int = 512):
        super().__init__()

        self.bridge = bridge
        self.gt_ds = gt_ds
        
        if c_list is None:
            c_list = [8,16,24,32,48,64]

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 =nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        ) 
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        
        _h_div_8 = image_size // 8 
        _h_div_16 = image_size // 16
        _h_div_32 = image_size // 32
        
        self.encoder4 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[2], c_list[3], x_res=_h_div_16, y_res=_h_div_16),
        )
        self.encoder5 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[3], c_list[4], x_res=_h_div_32, y_res=_h_div_32),
        )
        self.encoder6 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[4], c_list[5], x_res=_h_div_32, y_res=_h_div_32),
        )

        if bridge: 
            self.GAB1 = group_aggregation_bridge(c_list[1], c_list[0])
            self.GAB2 = group_aggregation_bridge(c_list[2], c_list[1])
            self.GAB3 = group_aggregation_bridge(c_list[3], c_list[2])
            self.GAB4 = group_aggregation_bridge(c_list[4], c_list[3])
            self.GAB5 = group_aggregation_bridge(c_list[5], c_list[4])
        
        if gt_ds:
            self.gt_conv1 = nn.Sequential(nn.Conv2d(c_list[4], 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(c_list[3], 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(c_list[2], 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(c_list[1], 1, 1))
            self.gt_conv5 = nn.Sequential(nn.Conv2d(c_list[0], 1, 1))
        
        self.decoder1 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[5], c_list[4], x_res=_h_div_32, y_res=_h_div_32),
        ) 
        self.decoder2 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[4], c_list[3], x_res=_h_div_16, y_res=_h_div_16),
        ) 
        self.decoder3 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[3], c_list[2], x_res=_h_div_8, y_res=_h_div_8),
        )  
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )  
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )  
        
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[List[torch.Tensor], torch.Tensor]]:
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out 

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out 

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out 
        
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)),2,2))
        t4 = out 
        
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)),2,2))
        t5 = out 
        
        out = F.gelu(self.encoder6(out)) 
        t6 = out
        

        out5 = F.gelu(self.dbn1(self.decoder1(out))) 
        if self.gt_ds: 
            gt_pre5 = self.gt_conv1(out5)
            t5 = self.GAB5(t6, t5, F.interpolate(gt_pre5, size=t5.shape[2:], mode='bilinear', align_corners=True))
        else: t5 = self.GAB5(t6, t5, torch.ones_like(gt_pre5))
        out5 = torch.add(out5, t5) 
        
        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) 
        if self.gt_ds: 
            gt_pre4 = self.gt_conv2(out4)
            t4 = self.GAB4(t5, t4, F.interpolate(gt_pre4, size=t4.shape[2:], mode='bilinear', align_corners=True))
        else: t4 = self.GAB4(t5, t4, torch.ones_like(gt_pre4))
        out4 = torch.add(out4, t4) 
        
        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) 
        if self.gt_ds: 
            gt_pre3 = self.gt_conv3(out3)
            t3 = self.GAB3(t4, t3, F.interpolate(gt_pre3, size=t3.shape[2:], mode='bilinear', align_corners=True)) 
        else: t3 = self.GAB3(t4, t3, torch.ones_like(gt_pre3)) 
        out3 = torch.add(out3, t3) 
        
        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) 
        if self.gt_ds: 
            gt_pre2 = self.gt_conv4(out2)
            t2 = self.GAB2(t3, t2, F.interpolate(gt_pre2, size=t2.shape[2:], mode='bilinear', align_corners=True))
        else: t2 = self.GAB2(t3, t2, torch.ones_like(gt_pre2))
        out2 = torch.add(out2, t2) 
        
        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) 
        if self.gt_ds: 
            gt_pre1 = self.gt_conv5(out1) 
            t1 = self.GAB1(t2, t1, F.interpolate(gt_pre1, size=t1.shape[2:], mode='bilinear', align_corners=True)) 
        else: t1 = self.GAB1(t2, t1, torch.ones_like(gt_pre1)) 
        out1 = torch.add(out1, t1) 
        
        out0 = F.interpolate(self.final(out1),scale_factor=(2,2),mode ='bilinear',align_corners=True) 
        
        if self.gt_ds:
            final_gt_pres = [
                F.interpolate(gt_pre5, scale_factor=32, mode='bilinear', align_corners=True),
                F.interpolate(gt_pre4, scale_factor=16, mode='bilinear', align_corners=True),
                F.interpolate(gt_pre3, scale_factor=8, mode='bilinear', align_corners=True),
                F.interpolate(gt_pre2, scale_factor=4, mode='bilinear', align_corners=True),
                F.interpolate(gt_pre1, scale_factor=2, mode='bilinear', align_corners=True)
            ]
            return {
                "out": out0, 
                "side5": final_gt_pres[0],
                "side4": final_gt_pres[1],
                "side3": final_gt_pres[2],
                "side2": final_gt_pres[3],
                "side1": final_gt_pres[4],
            }
        else:
            return out0 