import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple

from .common_layers import DoubleConv, Down, OutConv

class MultiHeadDense(nn.Module):
    def __init__(self, d, bias=False):
        super(MultiHeadDense, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(d, d))
        if bias:
            raise NotImplementedError()
            self.bias = nn.Parameter(torch.Tensor(d, d))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):

        b, wh, d = x.size()
        x = torch.bmm(x, self.weight.repeat(b, 1, 1))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

    def positional_encoding_2d(self, d_model, height, width):
        """
        reference: wzlxjtu/PositionalEncoding2D
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)

        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        return pe

    def forward(self, x):
        raise NotImplementedError()

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        channels = int(np.ceil(channels / 2))
        self.channels = channels
        inv_freq = 1. / (10000**(torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x,
                             device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y,
                             device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()),
                          dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2),
                          device=tensor.device).type(tensor.type())
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels:2 * self.channels] = emb_y

        return emb[None, :, :, :orig_ch].repeat(batch_size, 1, 1, 1)

class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)        
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

class MultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self, channel):
        super(MultiHeadSelfAttention, self).__init__()
        self.query = MultiHeadDense(channel, bias=False)
        self.key = MultiHeadDense(channel, bias=False)
        self.value = MultiHeadDense(channel, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.pe = PositionalEncodingPermute2D(channel)

    def forward(self, x):
        b, c, h, w = x.size()
        pe = self.pe(x)
        x = x + pe
        x = x.reshape(b, c, h * w).permute(0, 2, 1)
        Q = self.query(x)
        K = self.key(x)
        A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) /
                         math.sqrt(c))
        V = self.value(x)
        x = torch.bmm(A, V).permute(0, 2, 1).reshape(b, c, h, w)
        return x

class MultiHeadCrossAttention(MultiHeadAttention):
    def __init__(self, channelY, channelS, common_attn_res_for_QK_V=(64, 64)):
        """
        Args:
            channelY (int): Input channels for Y (upsampled path).
            channelS (int): Input channels for S (skip connection).
            common_attn_res_for_QK_V (tuple): Target spatial resolution (H, W) for Q, K, V
                                             before attention calculation to reduce memory.
                                             The attention matrix will be (H*W, H*W).
        """
        super(MultiHeadCrossAttention, self).__init__()
        self.common_attn_channels = channelS
        self.common_attn_res_for_QK_V = common_attn_res_for_QK_V

        self.Sconv_process = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(channelS, self.common_attn_channels, kernel_size=1),
            nn.BatchNorm2d(self.common_attn_channels),
            nn.ReLU(inplace=True)
        )
        self.Yconv_process = nn.Sequential(
            nn.Conv2d(channelY, self.common_attn_channels, kernel_size=1),
            nn.BatchNorm2d(self.common_attn_channels),
            nn.ReLU(inplace=True)
        )
        
        self.query = MultiHeadDense(self.common_attn_channels, bias=False)
        self.key = MultiHeadDense(self.common_attn_channels, bias=False)
        self.value = MultiHeadDense(self.common_attn_channels, bias=False)
        

        self.conv_after_attention = nn.Sequential(
            nn.Conv2d(self.common_attn_channels, self.common_attn_channels, kernel_size=1),
            nn.BatchNorm2d(self.common_attn_channels),
            nn.ReLU(inplace=True)
        )
        

        self.Yconv2_process = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(channelY, channelY, kernel_size=3, padding=1),
            nn.Conv2d(channelY, self.common_attn_channels, kernel_size=1),
            nn.BatchNorm2d(self.common_attn_channels),
            nn.ReLU(inplace=True)
        )
        
        self.softmax = nn.Softmax(dim=1)
        self.Spe = PositionalEncodingPermute2D(channelS)
        self.Ype = PositionalEncodingPermute2D(channelY)

    def forward(self, Y, S):

        Yb, Yc, Yh, Yw = Y.size()
        Sb, Sc, Sh, Sw = S.size()

        S_pe = S + self.Spe(S)
        S_processed = self.Sconv_process(S_pe)

        Y_pe = Y + self.Ype(Y)
        Y_processed = self.Yconv_process(Y_pe)

        H_attn, W_attn = self.common_attn_res_for_QK_V

        Q_source_for_attn = F.adaptive_avg_pool2d(Y_processed, (H_attn, W_attn))
        K_source_for_attn = F.adaptive_avg_pool2d(Y_processed, (H_attn, W_attn))
        V_source_for_attn = F.adaptive_avg_pool2d(S_processed, (H_attn, W_attn))

        Q_flat = Q_source_for_attn.flatten(2).permute(0, 2, 1)
        K_flat = K_source_for_attn.flatten(2).permute(0, 2, 1)
        V_flat = V_source_for_attn.flatten(2).permute(0, 2, 1)

        Q = self.query(Q_flat)
        K = self.key(K_flat)
        V = self.value(V_flat)

        A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(self.common_attn_channels)) 

        attn_out_flat = torch.bmm(A, V) 
        attn_out_spatial_low_res = attn_out_flat.permute(0, 2, 1).reshape(Yb, self.common_attn_channels, H_attn, W_attn)

        target_output_H = Yh * 2
        target_output_W = Yw * 2

        Z_attn = F.interpolate(attn_out_spatial_low_res, size=(target_output_H, target_output_W), mode='bilinear', align_corners=True)
        Z_attn = self.conv_after_attention(Z_attn)

        Y2_processed = self.Yconv2_process(Y_pe)

        final_output = torch.cat([Z_attn, Y2_processed], dim=1)
        return final_output

class TransformerUp(nn.Module):
    def __init__(self, Ychannels, Schannels, common_attn_res_for_QK_V=(64, 64)):
        super(TransformerUp, self).__init__()
        self.MHCA = MultiHeadCrossAttention(Ychannels, Schannels, common_attn_res_for_QK_V)
        self.conv = nn.Sequential(
            nn.Conv2d(Schannels * 2, 
                      Schannels,     
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.BatchNorm2d(Schannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(Schannels,
                      Schannels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.BatchNorm2d(Schannels),
            nn.ReLU(inplace=True))

    def forward(self, Y, S):
        x = self.MHCA(Y, S)
        x = self.conv(x)
        return x

class U_Transformer(nn.Module):
    def __init__(self, in_channels, num_classes, bilinear=True, common_attn_res_for_QK_V=(64, 64), **kwargs):
        super(U_Transformer, self).__init__()
        self.in_channels = in_channels
        self.classes = num_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.MHSA = MultiHeadSelfAttention(512) 
        
        self.up1 = TransformerUp(512, 256, common_attn_res_for_QK_V) 
        self.up2 = TransformerUp(256, 128, common_attn_res_for_QK_V)
        self.up3 = TransformerUp(128, 64, common_attn_res_for_QK_V)
        
        self.outc = OutConv(64, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.MHSA(x4) 
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits