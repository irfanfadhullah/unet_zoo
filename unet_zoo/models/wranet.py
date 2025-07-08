from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class BasicConv(nn.Sequential):
    """
    Basic Conv Block : conv2d - batchnorm - act
    """
    def __init__(self, in_channels:int, out_channels:int, kernel_size, stride:int=1, padding:int=0, dilation=1, groups=1, bias=True, norm='instance', act:nn.Module=nn.ReLU()):
        modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)]
        if norm=='instance':
            modules += [nn.InstanceNorm2d(out_channels)]
        if norm=='batch':
            modules += [nn.BatchNorm2d(out_channels)]
        if act is not None:
            modules += [act]
        super().__init__(*modules)
        
class ResidualBlock(nn.Module):
    """
    ResBlock : { (Conv2d - BN - ReLu) - Conv2d - BN } - sum - ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, act=nn.ReLU(), norm='instance'):
        super().__init__()
        self.convs = nn.Sequential(
            BasicConv(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=True, norm=norm, act=act),
            BasicConv(out_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=True, norm=norm, act=None)
        )
        self.act = act
    def forward(self, x):
        conv_out = self.convs(x)
        output = x + conv_out
        output = self.act(output)
        return output

class DeformableConv2d(nn.Module):
    '''Deformable convolution block'''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, dilation=1):
        super(DeformableConv2d, self).__init__()
        assert type(kernel_size) in (int, tuple), "type of kernel_size must be int or tuple"
        kernel_size = (kernel_size, kernel_size) if type(kernel_size)==int else kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        

        self.offset_conv = nn.Conv2d(in_channels, 2*kernel_size[0]*kernel_size[1], kernel_size=kernel_size, stride=stride, padding=(kernel_size[0]-1)//2, bias=True)

        self.modulator_conv = nn.Conv2d(in_channels, kernel_size[0]*kernel_size[1], kernel_size=kernel_size, stride=stride, padding=(kernel_size[0]-1)//2, bias=True)

        self.zero_init([self.offset_conv, self.modulator_conv])
        

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)
        
    def zero_init(self, convlayer):
        if type(convlayer) == list:
            for c in convlayer:
                nn.init.constant_(c.weight, 0.)
                nn.init.constant_(c.bias, 0.)
        else:
            nn.init.constant_(convlayer.weight, 0.)
            nn.init.constant_(convlayer.bias, 0.)
    
    def forward(self, x):
        offset = self.offset_conv(x)
        modulator =  torch.sigmoid(self.modulator_conv(x))

        output = torchvision.ops.deform_conv2d(input=x, 
                               offset=offset,
                               weight=self.conv.weight,
                               bias=self.conv.bias,
                               stride=self.stride,
                               padding=self.padding,
                               dilation=self.dilation,
                               mask=modulator)
        return output

class Deformable_Resblock(nn.Module):
    def __init__(self, in_channels, deformable_out_channels:int, kernel_size, stride:int=1, padding:int=0, dilation=1, bias=True, act:nn.Module=nn.ReLU()):
        super().__init__()
        self.convs = nn.Sequential(DeformableConv2d(in_channels, deformable_out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation),
                                   act)
        self.last_conv = nn.Conv2d(deformable_out_channels, in_channels, kernel_size=3, padding=1, bias=True)
    
    def forward(self, x):
        convs_out = self.convs(x)
        last_conv_out = self.last_conv(convs_out)
        return x + last_conv_out
    

class ModifiedDSCB(nn.Module):
    ''' with Depthwise conv. 1x1 conv - dw conv - IN - relu'''
    def __init__(self, in_channels, out_channels, kernel_size, norm='instance', act=nn.ReLU()):
        super().__init__()
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=1, bias=False, groups=in_channels)
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        if norm=='instance':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm=='batch':
            self.norm = nn.BatchNorm2d(out_channels)
        self.act = act
    def forward(self,x):
        output = self.dw_conv(x)
        output = self.conv_1x1(output)
        output = self.norm(output)
        output = self.act(output)
        return output

class LiteWRARB(nn.Module):
    def __init__(self, in_channels, ft_desc:int=1, num_blocks_list:List[int]=[1, 2, 3, 4], act=nn.ReLU(), norm='instance'):
        super().__init__()
        self.ft_desc = False
        if ft_desc != 1:
            self.ft_desc = True
            self.first_1x1 = nn.Conv2d(in_channels, in_channels//ft_desc, kernel_size=1, bias=False)
            
        streams = []
        for num_blocks in num_blocks_list:
            l =  [ModifiedDSCB(in_channels//ft_desc, in_channels//ft_desc, kernel_size=3, act=act, norm=norm)]*num_blocks
            streams += [nn.Sequential(*l)]
        
        self.streams = nn.ModuleList(streams)
        self.project = BasicConv((in_channels//ft_desc)*len(num_blocks_list), in_channels, kernel_size=1, bias=False, norm=norm, act=act)
        self.ag = nn.Sequential(nn.Conv2d(in_channels, in_channels//16, 1, bias=True), 
                                act,
                                nn.Conv2d(in_channels//16, in_channels, kernel_size=1, bias=True),
                                nn.Sigmoid())

        self.alpha = nn.Parameter(torch.zeros(1, in_channels, 1, 1), requires_grad=True)
        
    def forward(self, x):
        if self.ft_desc:
            features = self.first_1x1(x)    
        else: 
            features = x
        stream_outs = []
        for stream in self.streams:
            stream_outs += [stream(features)]
        stream_out = torch.cat(tuple(stream_outs), dim=1)
        project = self.project(stream_out)

        ag_out = self.ag(project) 
        output = project * ag_out 
        return (self.alpha * x) + output

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, ft_desc=1, norm='instance', act=nn.ReLU(), num_blocks_list:List[int]=[1, 2, 3, 4]):
        super().__init__()
        self.lite_wragb = LiteWRARB(in_channels, ft_desc=ft_desc, act=act, norm=norm, num_blocks_list=num_blocks_list)
        self.conv_3x3 = BasicConv(in_channels, in_channels, kernel_size=3, padding=1, bias=True, norm=norm, act=act)
    
    def forward(self, x):
        wr_featrures = self.lite_wragb(x)
        output = self.conv_3x3(wr_featrures)
        return output
    
class Decoder(nn.Module):
    def __init__(self, in_channels, norm='batch', act=nn.ReLU()):
        super().__init__()
        self.pixelshuffle_block = nn.Sequential(nn.Conv2d(in_channels, in_channels*4, 3, padding=1, bias=False),
                                        nn.PixelShuffle(2))
        self.conv_3x3_last = BasicConv(2*in_channels, in_channels, 3, padding=1, bias=True, norm=norm, act=act)
        self.rdb = Deformable_Resblock(in_channels, in_channels//4, kernel_size=3, padding=1, bias=True, act=act)
        
    def forward(self, x_s, x_l):
        '''x_s : small input, x_l: large input'''
        upsample = self.pixelshuffle_block(x_s)
        concat_features = torch.cat((upsample, x_l), dim=1)
        output = self.conv_3x3_last(concat_features)
        output = self.rdb(output)
        return output
    
class WRANet(nn.Module):
    """
    WRANet model, originally for image deblurring.
    Adapted for segmentation by changing output layer to predict num_classes and removing final tanh.
    The residual connection `+x` at the very end is also removed, as it's not typical for segmentation masks.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1, feature_channels: int = 128, mode: str = 'segmentation'):
        super().__init__()
        
        self.mode = mode

        self.convblock_1 = nn.Sequential(nn.Conv2d(in_channels, feature_channels//2, kernel_size=3, padding=1, bias=True),
                                                   nn.Conv2d(feature_channels//2, feature_channels, kernel_size=3, padding=1, bias=True),)
        
        self.encoder_block_1 = EncoderBlock(feature_channels)
        self.down1 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=2, padding=1, bias=False)
        
        self.encoder_block_2 = EncoderBlock(feature_channels)
        self.down2 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=2, padding=1, bias=False)
        
        self.encoder_block_3 = EncoderBlock(feature_channels)
        
        self.decoder_lv2 = Decoder(feature_channels)
        self.decoder_lv1 = Decoder(feature_channels)
        

        self.last_conv = nn.Sequential(nn.Conv2d(feature_channels, feature_channels//2, kernel_size=3, padding=1, bias=True),
                                        nn.Conv2d(feature_channels//2, feature_channels//4, kernel_size=3, padding=1, bias=True),
                                        nn.Conv2d(feature_channels//4, num_classes, kernel_size=3, padding=1, bias=True)
                                       )

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        aspp_features = self.convblock_1(x)
        lv1_features = self.encoder_block_1(aspp_features)
        
        lv2_features = self.down1(lv1_features)
        lv2_features = self.encoder_block_2(lv2_features)
        
        lv3_features = self.down2(lv2_features)
        lv3_features = self.encoder_block_3(lv3_features)
        

        lv2_decout = self.decoder_lv2(lv3_features, lv2_features)

        lv1_decout = self.decoder_lv1(lv2_decout, lv1_features)
        
        last_conv_out = self.last_conv(lv1_decout)
        

        return last_conv_out
  
