import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
from typing import List, Optional

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):

        B, N, C = x.shape
        

        if H * W != N:

            total_spatial = N
            H = W = int(total_spatial ** 0.5)
            if H * W != total_spatial:

                import math
                H = int(math.sqrt(total_spatial))
                W = total_spatial // H
                if H * W != total_spatial:

                    for i in range(int(math.sqrt(total_spatial)), 0, -1):
                        if total_spatial % i == 0:
                            H, W = i, total_spatial // i
                            break
        
        x = self.fc1(x)
        

        xn = x.transpose(1, 2).view(B, -1, H, W).contiguous()
        xn = self.dwconv(xn)
        xn = xn.flatten(2).transpose(1, 2).contiguous()
        
        x = self.act(xn)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        

        if H * W != N:
            total_spatial = N
            H = W = int(total_spatial ** 0.5)
            if H * W != total_spatial:
                import math
                for i in range(int(math.sqrt(total_spatial)), 0, -1):
                    if total_spatial % i == 0:
                        H, W = i, total_spatial // i
                        break
        
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):

        B, N, C = x.shape
        
        if H * W != N:

            import math
            spatial_size = N
            H = W = int(math.sqrt(spatial_size))
            if H * W != spatial_size:

                for i in range(int(math.sqrt(spatial_size)), 0, -1):
                    if spatial_size % i == 0:
                        H, W = i, spatial_size // i
                        break
        
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class UNext(nn.Module):
    def __init__(self, input_channels=3, num_classes=1, img_size=224, embed_dims=None, 
                 num_heads=None, mlp_ratios=None, qkv_bias=False, qk_scale=None, 
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, 
                 norm_layer=nn.LayerNorm, depths=None, sr_ratios=None, **kwargs):
        super().__init__()
        

        if embed_dims is None:
            embed_dims = [128, 160, 256]
        if num_heads is None:
            num_heads = [1, 2, 4, 8]
        if mlp_ratios is None:
            mlp_ratios = [4, 4, 4, 4]
        if depths is None:
            depths = [3, 4, 6, 3]
        if sr_ratios is None:
            sr_ratios = [8, 4, 2, 1]

        self.num_classes = num_classes
        self.depths = depths

        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, 
                                            in_chans=input_channels, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, 
                                            in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, 
                                            in_chans=embed_dims[1], embed_dim=embed_dims[2])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        self.decoder_level1 = nn.Conv2d(embed_dims[2], embed_dims[1], 3, padding=1)
        self.decoder_level2 = nn.Conv2d(embed_dims[1], embed_dims[0], 3, padding=1)
        self.decoder_level3 = nn.Conv2d(embed_dims[0], embed_dims[0], 3, padding=1)
        

        self.final_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(embed_dims[0], num_classes, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights for UNext model components."""
        if isinstance(m, nn.Linear):

            if hasattr(nn.init, 'trunc_normal_'):
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
        elif isinstance(m, nn.LayerNorm):

            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
        elif isinstance(m, nn.Conv2d):

            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):

            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B = x.shape[0]
        

        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x1 = x.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()

        x, H, W = self.patch_embed2(x1)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x2 = x.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()

        x, H, W = self.patch_embed3(x2)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x3 = x.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()

        x = F.interpolate(x3, size=x2.shape[-2:], mode='bilinear', align_corners=True)
        x = self.decoder_level1(x)
        x = x + x2
        
        x = F.interpolate(x, size=x1.shape[-2:], mode='bilinear', align_corners=True)
        x = self.decoder_level2(x)
        x = x + x1
        
        x = self.decoder_level3(x)
        

        x = self.final_up(x)
        x = self.final_conv(x)
        
        return x

class UNext_S(UNext):
    """
    UNext-S: Smaller variant of UNext designed for efficiency.
    """
    
    def __init__(self, input_channels=3, num_classes=1, img_size=224, **kwargs):

        embed_dims = [64, 128, 160]      
        num_heads = [1, 2, 4]           
        depths = [2, 2, 2]              
        sr_ratios = [8, 4, 2]           
        mlp_ratios = [4, 4, 4]          
        

        kwargs.pop('embed_dims', None)
        kwargs.pop('num_heads', None)
        kwargs.pop('depths', None)
        kwargs.pop('sr_ratios', None)
        kwargs.pop('mlp_ratios', None)
        
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            img_size=img_size,
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            depths=depths,
            sr_ratios=sr_ratios,
            **kwargs
        )
