import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, List, Dict, Union
from einops import rearrange

class EfficientSelfAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio 
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim*2, bias=True)
        self.proj = nn.Linear(dim, dim)

        if reduction_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, reduction_ratio, reduction_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.reduction_ratio > 1:
            p_x = x.clone().permute(0, 2, 1).reshape(B, C, H, W)
            sp_x = self.sr(p_x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(sp_x)
            
        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)

        return out

class SelfAtten(nn.Module): 
    def __init__(self, dim, head):
        super().__init__()
        self.head = head
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim*2, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)
            
        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)

        return out

class Scale_reduce(nn.Module):
    def __init__(self, dim, reduction_ratios: List[int], patch_resolutions: List[Tuple[int, int]], mi_t_dims: List[int]):
        super().__init__()
        self.dim = dim 
        self.reduction_ratios = reduction_ratios
        self.patch_resolutions = patch_resolutions
        self.mi_t_dims = mi_t_dims

        self.sr_convs = nn.ModuleList()
        for i in range(len(reduction_ratios)):
            if reduction_ratios[i] > 1:
                self.sr_convs.append(nn.Conv2d(self.dim, self.dim, reduction_ratios[i], reduction_ratios[i]))
            else:
                self.sr_convs.append(nn.Identity()) 
        self.norm = nn.LayerNorm(dim) 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N_total, C_common = x.shape 

        num_patches = [res[0]*res[1] for res in self.patch_resolutions]
        slice_indices = [0] + [sum(num_patches[:i+1]) for i in range(len(num_patches))]

        sliced_features = []
        for i in range(len(self.patch_resolutions)):
            H, W = self.patch_resolutions[i]
            current_slice = x[:, slice_indices[i]:slice_indices[i+1], :]
            
            current_slice_spatial = current_slice.permute(0, 2, 1).reshape(B, C_common, H, W)
            
            reduced_slice = self.sr_convs[i](current_slice_spatial)
            
            sliced_features.append(reduced_slice.flatten(2).permute(0, 2, 1))
        
        reduce_out = self.norm(torch.cat(sliced_features, -2))
        
        return reduce_out

class M_EfficientSelfAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratios: List[int], patch_resolutions: List[Tuple[int, int]], mi_t_dims: List[int]):
        super().__init__()
        self.head = head
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim*2, bias=True)
        self.proj = nn.Linear(dim, dim)
        
        self.scale_reduce = Scale_reduce(dim, reduction_ratios, patch_resolutions, mi_t_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape 
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        x_reduced = self.scale_reduce(x)
            
        kv = self.kv(x_reduced).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)

        return out

class LocalEnhance_EfficientSelfAtten(nn.Module): 
    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio 
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim*2, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.local_pos = DWConv(dim) 

        if reduction_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, reduction_ratio, reduction_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.reduction_ratio > 1:
            p_x = x.clone().permute(0, 2, 1).reshape(B, C, H, W)
            sp_x = self.sr(p_x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(sp_x)
            
        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)
        
        local_v = v.permute(0, 2, 1, 3).reshape(B, N, C)
        local_pos_out = self.local_pos(local_v, H, W).reshape(B, -1, self.head, C//self.head).permute(0, 2, 1, 3) 
        
        x_atten = ((attn_score @ v) + local_pos_out).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)

        return out

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)

class MixFFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        ax = self.act(self.dwconv(self.fc1(x), H, W))
        out = self.fc2(ax)
        return out

class MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2) 
        self.norm2 = nn.LayerNorm(c2) 
        self.norm3 = nn.LayerNorm(c2)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        fc1_out = self.fc1(x)
        dwconv_out = self.dwconv(fc1_out, H, W)
        ax = self.act(self.norm1(dwconv_out + fc1_out))
        out = self.fc2(ax)
        return out

class MLP_FFN(nn.Module): 
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class MixD_FFN(nn.Module): 
    def __init__(self, c1, c2, fuse_mode = "add"):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2) 
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1) if fuse_mode=="add" else nn.Linear(c2*2, c1)
        self.fuse_mode = fuse_mode

    def forward(self, x, H, W): 
        ax = self.dwconv(self.fc1(x), H, W)
        fuse = self.act(ax+self.fc1(x)) if self.fuse_mode=="add" else self.act(torch.cat([ax, self.fc1(x)],2))
        out = self.fc2(ax) 
        return out

class OverlapPatchEmbeddings(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, padding=1, in_ch=3, dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2 
        self.proj = nn.Conv2d(in_ch, dim, patch_size, stride, padding)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        px = self.proj(x)
        _, _, H, W = px.shape
        fx = px.flatten(2).transpose(1, 2) 
        nfx = self.norm(fx)
        return nfx, H, W

class TransformerBlock(nn.Module):
    def __init__(self, dim, head, reduction_ratio=1, token_mlp='mix'):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAtten(dim, head, reduction_ratio)
        self.norm2 = nn.LayerNorm(dim)
        if token_mlp=='mix':
            self.mlp = MixFFN(dim, int(dim*4))  
        elif token_mlp=='mix_skip':
            self.mlp = MixFFN_skip(dim, int(dim*4)) 
        else:
            self.mlp = MLP_FFN(dim, int(dim*4))

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        tx = x + self.attn(self.norm1(x), H, W)
        mx = tx + self.mlp(self.norm2(tx), H, W)
        return mx

class FuseTransformerBlock(nn.Module): 
    def __init__(self, dim, head, reduction_ratio=1, fuse_mode = "add"):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAtten(dim, head, reduction_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixD_FFN(dim, int(dim*4), fuse_mode)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        tx = x + self.attn(self.norm1(x), H, W)
        mx = tx + self.mlp(self.norm2(tx), H, W)
        return mx

class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2) 
        return self.proj(x)

class ConvModule(nn.Module):
    def __init__(self, c1, c2, k):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.activate = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))

class MiT(nn.Module):
    def __init__(self, image_size: int, dims: List[int], layers: List[int], in_ch: int=3, token_mlp: str='mix_skip'):
        super().__init__()
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]
        reduction_ratios = [8, 4, 2, 1]
        heads = [1, 2, 5, 8]

        self.patch_embed1 = OverlapPatchEmbeddings(image_size, patch_sizes[0], strides[0], padding_sizes[0], in_ch, dims[0])
        self.patch_embed2 = OverlapPatchEmbeddings(image_size//strides[0], patch_sizes[1], strides[1],  padding_sizes[1],dims[0], dims[1])
        self.patch_embed3 = OverlapPatchEmbeddings(image_size//(strides[0]*strides[1]), patch_sizes[2], strides[2],  padding_sizes[2],dims[1], dims[2])
        self.patch_embed4 = OverlapPatchEmbeddings(image_size//(strides[0]*strides[1]*strides[2]), patch_sizes[3], strides[3],  padding_sizes[3],dims[2], dims[3])
        
        self.block1 = nn.ModuleList([
            TransformerBlock(dims[0], heads[0], reduction_ratios[0],token_mlp)
        for _ in range(layers[0])])
        self.norm1 = nn.LayerNorm(dims[0])

        self.block2 = nn.ModuleList([
            TransformerBlock(dims[1], heads[1], reduction_ratios[1],token_mlp)
        for _ in range(layers[1])])
        self.norm2 = nn.LayerNorm(dims[1])

        self.block3 = nn.ModuleList([
            TransformerBlock(dims[2], heads[2], reduction_ratios[2], token_mlp)
        for _ in range(layers[2])])
        self.norm3 = nn.LayerNorm(dims[2])

        self.block4 = nn.ModuleList([
            TransformerBlock(dims[3], heads[3], reduction_ratios[3], token_mlp)
        for _ in range(layers[3])])
        self.norm4 = nn.LayerNorm(dims[3])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        B = x.shape[0]
        outs = []

        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() 
        outs.append(x)

        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

class FuseMiT(nn.Module): 
    def __init__(self, image_size, dims, layers, fuse_mode='add'):
        super().__init__()
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]
        reduction_ratios = [8, 4, 2, 1]
        heads = [1, 2, 5, 8]

        self.patch_embed1 = OverlapPatchEmbeddings(image_size, patch_sizes[0], strides[0], padding_sizes[0], 3, dims[0])
        self.patch_embed2 = OverlapPatchEmbeddings(image_size//4, patch_sizes[1], strides[1],  padding_sizes[1],dims[0], dims[1])
        self.patch_embed3 = OverlapPatchEmbeddings(image_size//8, patch_sizes[2], strides[2],  padding_sizes[2],dims[1], dims[2])
        self.patch_embed4 = OverlapPatchEmbeddings(image_size//16, patch_sizes[3], strides[3],  padding_sizes[3],dims[2], dims[3])
        
        self.block1 = nn.ModuleList([
            FuseTransformerBlock(dims[0], heads[0], reduction_ratios[0],fuse_mode)
        for _ in range(layers[0])])
        self.norm1 = nn.LayerNorm(dims[0])

        self.block2 = nn.ModuleList([
            FuseTransformerBlock(dims[1], heads[1], reduction_ratios[1],fuse_mode)
        for _ in range(layers[1])])
        self.norm2 = nn.LayerNorm(dims[1])

        self.block3 = nn.ModuleList([
            FuseTransformerBlock(dims[2], heads[2], reduction_ratios[2], fuse_mode)
        for _ in range(layers[2])])
        self.norm3 = nn.LayerNorm(dims[2])

        self.block4 = nn.ModuleList([
            FuseTransformerBlock(dims[3], heads[3], reduction_ratios[3], fuse_mode)
        for _ in range(layers[3])])
        self.norm4 = nn.LayerNorm(dims[3])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        B = x.shape[0]
        outs = []

        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

class Decoder(nn.Module): 
    def __init__(self, dims: List[int], embed_dim: int, num_classes: int):
        super().__init__()

        self.linear_c1 = MLP(dims[0], embed_dim)
        self.linear_c2 = MLP(dims[1], embed_dim)
        self.linear_c3 = MLP(dims[2], embed_dim)
        self.linear_c4 = MLP(dims[3], embed_dim)

        self.linear_fuse = ConvModule(embed_dim*4, embed_dim, 1)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)

        self.dropout = nn.Dropout2d(0.1)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        c1, c2, c3, c4 = inputs 
        n = c1.shape[0]
        
        c1f = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        
        c2f = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        c2f = F.interpolate(c2f, size=c1.shape[2:], mode='bilinear', align_corners=False)

        c3f = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        c3f = F.interpolate(c3f, size=c1.shape[2:], mode='bilinear', align_corners=False)

        c4f = self.linear_c4(c4).permute(0, 2, 3, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        c4f = F.interpolate(c4f, size=c1.shape[2:], mode='bilinear', align_corners=False)

        c = self.linear_fuse(torch.cat([c4f, c3f, c2f, c1f], dim=1))
        c = self.dropout(c)
        return self.linear_pred(c)

segformer_settings: Dict[str, List] = {
    'B0': [[32, 64, 160, 256], [2, 2, 2, 2], 256],  
    'B1': [[64, 128, 320, 512], [2, 2, 2, 2], 256],
    'B2': [[64, 128, 320, 512], [3, 4, 6, 3], 768],
    'B3': [[64, 128, 320, 512], [3, 4, 18, 3], 768],
    'B4': [[64, 128, 320, 512], [3, 8, 27, 3], 768],
    'B5': [[64, 128, 320, 512], [3, 6, 40, 3], 768]
}

class SegFormer(nn.Module): 
    def __init__(self, model_name: str = 'B0', num_classes: int = 19, image_size: int = 224) -> None:
        super().__init__()
        assert model_name in segformer_settings.keys(), f"SegFormer model name should be in {list(segformer_settings.keys())}"
        dims, layers, embed_dim = segformer_settings[model_name]

        self.backbone = MiT(image_size, dims, layers)
        self.decode_head = Decoder(dims, embed_dim, num_classes)

    def init_weights(self, pretrained: str = None) -> None:
        if pretrained:
            self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        encoder_outs = self.backbone(x)
        return self.decode_head(encoder_outs)

class PatchExpand(nn.Module):
    def __init__(self, input_resolution: Tuple[int, int], dim: int, dim_scale: int=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim * dim_scale**2, bias=False)
        self.norm = norm_layer(dim) 
        self.output_dim = dim 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: B, H*W, C_in (C_in = dim or dim*2 from previous stage)
        """
        H, W = self.input_resolution
        x = self.expand(x) 
        
        B, L, C_expanded = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C_expanded)
        x = rearrange(x, 'b h w (p1 p2 c_out)-> b (h p1) (w p2) c_out', p1=self.dim_scale, p2=self.dim_scale, c_out=self.output_dim)
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x.clone())

        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution: Tuple[int, int], dim: int, dim_scale: int=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale 
        self.expand = nn.Linear(dim, dim * dim_scale**2, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: B, H*W, C_in
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C_expanded = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C_expanded)
        x = rearrange(x, 'b h w (p1 p2 c_out)-> b (h p1) (w p2) c_out', p1=self.dim_scale, p2=self.dim_scale, c_out=C_expanded//(self.dim_scale**2))

        x = x.view(B,-1,self.output_dim)
        x= self.norm(x.clone())

        return x

class SegU_decoder(nn.Module):
    def __init__(self, input_resolution: Tuple[int, int], in_out_chan: List[int], heads: int, reduction_ratios: int, token_mlp_mode: str, n_class: int=9, norm_layer=nn.LayerNorm, is_last: bool=False): 
        super().__init__()
        self.input_resolution = input_resolution
        
        dims = in_out_chan[0] 
        out_dim = in_out_chan[1] 

        if not is_last:
            self.concat_linear = nn.Linear(dims, out_dim) 
            self.layer_up = PatchExpand(input_resolution=input_resolution, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.concat_linear = nn.Linear(dims, out_dim)
            self.layer_up = FinalPatchExpand_X4(input_resolution=input_resolution, dim=out_dim, dim_scale=4, norm_layer=norm_layer)
            self.last_layer = nn.Conv2d(out_dim, n_class,1)

        self.layer_former_1 = TransformerBlock(out_dim, heads, reduction_ratios, token_mlp=token_mlp_mode) 
        self.layer_former_2 = TransformerBlock(out_dim, heads, reduction_ratios, token_mlp=token_mlp_mode)
       
        self.init_weights()

    def init_weights(self): 
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
      
    def forward(self, x1: torch.Tensor, x2: torch.Tensor=None) -> torch.Tensor:
        current_spatial_H, current_spatial_W = self.input_resolution 

        if x2 is not None:
            if len(x2.shape) == 4:
                B, C_skip, H_skip, W_skip = x2.shape
                x2_flat = x2.permute(0, 2, 3, 1).view(B, -1, C_skip)
                current_spatial_H, current_spatial_W = H_skip, W_skip 
            else:
                B, N_skip, C_skip = x2.shape
                x2_flat = x2
            
            cat_x = torch.cat([x1, x2_flat], dim=-1)
            cat_linear_x = self.concat_linear(cat_x)

            tran_layer_1 = self.layer_former_1(cat_linear_x, current_spatial_H, current_spatial_W)
            tran_layer_2 = self.layer_former_2(tran_layer_1, current_spatial_H, current_spatial_W)
            
            if self.last_layer:
                expanded_features = self.layer_up(tran_layer_2)
                B_exp, N_exp, C_exp = expanded_features.shape
                H_exp = int(torch.sqrt(torch.tensor(N_exp)).item())
                W_exp = H_exp
                out = self.last_layer(expanded_features.view(B_exp, H_exp, W_exp, C_exp).permute(0,3,1,2)) 
            else:
                out = self.layer_up(tran_layer_2)
        else:
            tran_layer_1 = self.layer_former_1(x1, current_spatial_H, current_spatial_W)
            tran_layer_2 = self.layer_former_2(tran_layer_1, current_spatial_H, current_spatial_W)
            out = self.layer_up(tran_layer_2)
            
        return out

class BridgeLayer_4(nn.Module):
    def __init__(self, mi_t_dims: List[int], head: int, reduction_ratios: List[int], image_size: int):
        super().__init__()
        self.mi_t_dims = mi_t_dims
        self.common_bridge_dim = mi_t_dims[0]

        self.proj_c1 = nn.Linear(mi_t_dims[0], self.common_bridge_dim) 
        self.proj_c2 = nn.Linear(mi_t_dims[1], self.common_bridge_dim)
        self.proj_c3 = nn.Linear(mi_t_dims[2], self.common_bridge_dim)
        self.proj_c4 = nn.Linear(mi_t_dims[3], self.common_bridge_dim)

        self.norm1 = nn.LayerNorm(self.common_bridge_dim)
        self.attn = M_EfficientSelfAtten(self.common_bridge_dim, head, reduction_ratios, self._get_patch_resolutions(image_size), mi_t_dims)
        self.norm2 = nn.LayerNorm(self.common_bridge_dim)

        self.mixffn1 = MixFFN_skip(self.common_bridge_dim, self.common_bridge_dim*4)
        self.mixffn2 = MixFFN_skip(self.common_bridge_dim, self.common_bridge_dim*4)
        self.mixffn3 = MixFFN_skip(self.common_bridge_dim, self.common_bridge_dim*4)
        self.mixffn4 = MixFFN_skip(self.common_bridge_dim, self.common_bridge_dim*4)

        self.patch_resolutions = self._get_patch_resolutions(image_size)

    def _get_patch_resolutions(self, image_size: int) -> List[Tuple[int, int]]:
        return [
            (image_size // 4, image_size // 4),
            (image_size // 8, image_size // 8),
            (image_size // 16, image_size // 16),
            (image_size // 32, image_size // 32)
        ]

    def forward(self, inputs: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        B = inputs[0].shape[0] if isinstance(inputs, list) else inputs.shape[0]

        if isinstance(inputs, list):
            c1, c2, c3, c4 = inputs
            
            c1f = self.proj_c1(c1.permute(0, 2, 3, 1)).reshape(B, -1, self.common_bridge_dim)
            c2f = self.proj_c2(c2.permute(0, 2, 3, 1)).reshape(B, -1, self.common_bridge_dim)
            c3f = self.proj_c3(c3.permute(0, 2, 3, 1)).reshape(B, -1, self.common_bridge_dim)
            c4f = self.proj_c4(c4.permute(0, 2, 3, 1)).reshape(B, -1, self.common_bridge_dim)
            
            cat_inputs = torch.cat([c1f, c2f, c3f, c4f], -2)
        else:
            cat_inputs = inputs
        
        tx1 = cat_inputs + self.attn(self.norm1(cat_inputs))
        tx = self.norm2(tx1)

        num_patches_per_stage = [res[0]*res[1] for res in self.patch_resolutions]
        
        cumulative_patch_indices = [0]
        for i in range(len(num_patches_per_stage)):
            cumulative_patch_indices.append(cumulative_patch_indices[-1] + num_patches_per_stage[i])

        tem1 = tx[:, cumulative_patch_indices[0]:cumulative_patch_indices[1], :]
        tem2 = tx[:, cumulative_patch_indices[1]:cumulative_patch_indices[2], :]
        tem3 = tx[:, cumulative_patch_indices[2]:cumulative_patch_indices[3], :]
        tem4 = tx[:, cumulative_patch_indices[3]:cumulative_patch_indices[4], :]

        m1f = self.mixffn1(tem1, self.patch_resolutions[0][0], self.patch_resolutions[0][1])
        m2f = self.mixffn2(tem2, self.patch_resolutions[1][0], self.patch_resolutions[1][1])
        m3f = self.mixffn3(tem3, self.patch_resolutions[2][0], self.patch_resolutions[2][1])
        m4f = self.mixffn4(tem4, self.patch_resolutions[3][0], self.patch_resolutions[3][1])

        t1_ffn = torch.cat([m1f, m2f, m3f, m4f], -2)
        tx2 = tx1 + t1_ffn

        return tx2

class BridgeLayer_3(nn.Module):
    def __init__(self, mi_t_dims: List[int], head: int, reduction_ratios: List[int], image_size: int):
        super().__init__()
        self.mi_t_dims = mi_t_dims
        self.common_bridge_dim = mi_t_dims[0] 

        self.proj_c2 = nn.Linear(mi_t_dims[1], self.common_bridge_dim) 
        self.proj_c3 = nn.Linear(mi_t_dims[2], self.common_bridge_dim)
        self.proj_c4 = nn.Linear(mi_t_dims[3], self.common_bridge_dim)

        self.norm1 = nn.LayerNorm(self.common_bridge_dim)
        self.attn = M_EfficientSelfAtten(self.common_bridge_dim, head, reduction_ratios[1:], self._get_patch_resolutions(image_size)[1:], mi_t_dims[1:])
        self.norm2 = nn.LayerNorm(self.common_bridge_dim)
        
        self.mixffn2 = MixFFN_skip(self.common_bridge_dim, self.common_bridge_dim*4)
        self.mixffn3 = MixFFN_skip(self.common_bridge_dim, self.common_bridge_dim*4)
        self.mixffn4 = MixFFN_skip(self.common_bridge_dim, self.common_bridge_dim*4)

        self.patch_resolutions = self._get_patch_resolutions(image_size) 

    def _get_patch_resolutions(self, image_size: int) -> List[Tuple[int, int]]:
        return [
            (image_size // 8, image_size // 8),
            (image_size // 16, image_size // 16),
            (image_size // 32, image_size // 32)
        ]

    def forward(self, inputs: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        B = inputs[0].shape[0] if isinstance(inputs, list) else inputs.shape[0]

        if isinstance(inputs, list): 
            c2, c3, c4 = inputs
            
            c2f = self.proj_c2(c2.permute(0, 2, 3, 1)).reshape(B, -1, self.common_bridge_dim)
            c3f = self.proj_c3(c3.permute(0, 2, 3, 1)).reshape(B, -1, self.common_bridge_dim)
            c4f = self.proj_c4(c4.permute(0, 2, 3, 1)).reshape(B, -1, self.common_bridge_dim)
            cat_inputs = torch.cat([c2f, c3f, c4f], -2)
        else:
            cat_inputs = inputs
        
        tx1 = cat_inputs + self.attn(self.norm1(cat_inputs))
        tx = self.norm2(tx1)

        num_patches_per_stage = [res[0]*res[1] for res in self.patch_resolutions]
        cumulative_patch_indices = [0]
        for i in range(len(num_patches_per_stage)):
            cumulative_patch_indices.append(cumulative_patch_indices[-1] + num_patches_per_stage[i])

        tem2 = tx[:, cumulative_patch_indices[0]:cumulative_patch_indices[1], :]
        tem3 = tx[:, cumulative_patch_indices[1]:cumulative_patch_indices[2], :]
        tem4 = tx[:, cumulative_patch_indices[2]:cumulative_patch_indices[3], :]

        m2f = self.mixffn2(tem2, self.patch_resolutions[0][0], self.patch_resolutions[0][1])
        m3f = self.mixffn3(tem3, self.patch_resolutions[1][0], self.patch_resolutions[1][1])
        m4f = self.mixffn4(tem4, self.patch_resolutions[2][0], self.patch_resolutions[2][1])

        t1_ffn = torch.cat([m2f, m3f, m4f], -2)
        tx2 = tx1 + t1_ffn

        return tx2

class BridegeBlock_4(nn.Module):
    def __init__(self, mi_t_dims: List[int], head: int, reduction_ratios: List[int], image_size: int):
        super().__init__()
        self.image_size = image_size
        self.mi_t_dims = mi_t_dims 
        self.common_bridge_dim = mi_t_dims[0] 

        self.bridge_layer1 = BridgeLayer_4(self.mi_t_dims, head, reduction_ratios, image_size)
        self.bridge_layer2 = BridgeLayer_4(self.mi_t_dims, head, reduction_ratios, image_size)
        self.bridge_layer3 = BridgeLayer_4(self.mi_t_dims, head, reduction_ratios, image_size)
        self.bridge_layer4 = BridgeLayer_4(self.mi_t_dims, head, reduction_ratios, image_size)

        self.patch_resolutions = [ 
            (image_size // 4, image_size // 4), 
            (image_size // 8, image_size // 8),
            (image_size // 16, image_size // 16),
            (image_size // 32, image_size // 32)
        ]
        
        self.proj_back_c1 = nn.Linear(self.common_bridge_dim, mi_t_dims[0])
        self.proj_back_c2 = nn.Linear(self.common_bridge_dim, mi_t_dims[1])
        self.proj_back_c3 = nn.Linear(self.common_bridge_dim, mi_t_dims[2])
        self.proj_back_c4 = nn.Linear(self.common_bridge_dim, mi_t_dims[3])

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]: 
        
        bridge1_out = self.bridge_layer1(x) 
        bridge2_out = self.bridge_layer2(bridge1_out) 
        bridge3_out = self.bridge_layer3(bridge2_out) 
        bridge4_out = self.bridge_layer4(bridge3_out) 

        B = bridge4_out.shape[0]

        num_patches_per_stage = [res[0]*res[1] for res in self.patch_resolutions]
        cumulative_patch_indices = [0]
        for i in range(len(num_patches_per_stage)):
            cumulative_patch_indices.append(cumulative_patch_indices[-1] + num_patches_per_stage[i])

        sk1_flat = self.proj_back_c1(bridge4_out[:, cumulative_patch_indices[0]:cumulative_patch_indices[1], :])
        sk2_flat = self.proj_back_c2(bridge4_out[:, cumulative_patch_indices[1]:cumulative_patch_indices[2], :])
        sk3_flat = self.proj_back_c3(bridge4_out[:, cumulative_patch_indices[2]:cumulative_patch_indices[3], :])
        sk4_flat = self.proj_back_c4(bridge4_out[:, cumulative_patch_indices[3]:cumulative_patch_indices[4], :])

        sk1 = sk1_flat.reshape(B, self.patch_resolutions[0][0], self.patch_resolutions[0][1], self.mi_t_dims[0]).permute(0,3,1,2)
        sk2 = sk2_flat.reshape(B, self.patch_resolutions[1][0], self.patch_resolutions[1][1], self.mi_t_dims[1]).permute(0,3,1,2)
        sk3 = sk3_flat.reshape(B, self.patch_resolutions[2][0], self.patch_resolutions[2][1], self.mi_t_dims[2]).permute(0,3,1,2)
        sk4 = sk4_flat.reshape(B, self.patch_resolutions[3][0], self.patch_resolutions[3][1], self.mi_t_dims[3]).permute(0,3,1,2)

        return [sk1, sk2, sk3, sk4]

class BridegeBlock_3(nn.Module):
    def __init__(self, mi_t_dims: List[int], head: int, reduction_ratios: List[int], image_size: int):
        super().__init__()
        self.image_size = image_size

        self.target_mi_t_dims = mi_t_dims[1:] 
        self.common_bridge_dim = self.target_mi_t_dims[0] 

        self.bridge_layer1 = BridgeLayer_3(mi_t_dims, head, reduction_ratios, image_size)
        self.bridge_layer2 = BridgeLayer_3(mi_t_dims, head, reduction_ratios, image_size)
        self.bridge_layer3 = BridgeLayer_3(mi_t_dims, head, reduction_ratios, image_size)
        self.bridge_layer4 = BridgeLayer_3(mi_t_dims, head, reduction_ratios, image_size)

        self.patch_resolutions = [ 
            (image_size // 8, image_size // 8), 
            (image_size // 16, image_size // 16), 
            (image_size // 32, image_size // 32) 
        ]
        
        self.proj_back_c2 = nn.Linear(self.common_bridge_dim, self.target_mi_t_dims[0]) 
        self.proj_back_c3 = nn.Linear(self.common_bridge_dim, self.target_mi_t_dims[1]) 
        self.proj_back_c4 = nn.Linear(self.common_bridge_dim, self.target_mi_t_dims[2]) 

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        mi_t_c1 = x[0] 
        mi_t_c2_to_process = x[1]
        mi_t_c3_to_process = x[2]
        mi_t_c4_to_process = x[3]
        
        bridge1_out = self.bridge_layer1([mi_t_c2_to_process, mi_t_c3_to_process, mi_t_c4_to_process])
        bridge2_out = self.bridge_layer2(bridge1_out)
        bridge3_out = self.bridge_layer3(bridge2_out)
        bridge4_out = self.bridge_layer4(bridge3_out)

        B = bridge4_out.shape[0]

        num_patches_per_stage = [res[0]*res[1] for res in self.patch_resolutions]
        cumulative_patch_indices = [0]
        for i in range(len(num_patches_per_stage)):
            cumulative_patch_indices.append(cumulative_patch_indices[-1] + num_patches_per_stage[i])

        sk2_flat = self.proj_back_c2(bridge4_out[:, cumulative_patch_indices[0]:cumulative_patch_indices[1], :])
        sk3_flat = self.proj_back_c3(bridge4_out[:, cumulative_patch_indices[1]:cumulative_patch_indices[2], :])
        sk4_flat = self.proj_back_c4(bridge4_out[:, cumulative_patch_indices[2]:cumulative_patch_indices[3], :])

        sk2 = sk2_flat.reshape(B, self.patch_resolutions[0][0], self.patch_resolutions[0][1], self.target_mi_t_dims[0]).permute(0,3,1,2)
        sk3 = sk3_flat.reshape(B, self.patch_resolutions[1][0], self.patch_resolutions[1][1], self.target_mi_t_dims[1]).permute(0,3,1,2)
        sk4 = sk4_flat.reshape(B, self.patch_resolutions[2][0], self.patch_resolutions[2][1], self.target_mi_t_dims[2]).permute(0,3,1,2)

        return [mi_t_c1, sk2, sk3, sk4]

class MISSFormer(nn.Module):
    def __init__(self, num_classes: int=1, in_channels: int=3, token_mlp_mode: str="mix_skip", 
                 encoder_pretrained: bool=True, image_size: int=512, **kwargs):
        super().__init__()
    
        model_name = 'B1' 
        dims, layers, embed_dim = segformer_settings[model_name] 
        self.backbone = MiT(image_size, dims, layers, in_channels, token_mlp_mode)

        reduction_ratios = [8, 4, 2, 1] 
        heads = [1, 2, 5, 8] 

        d_base_feat_size = image_size // 32 

        self.bridge = BridegeBlock_4(dims, heads[0], reduction_ratios, image_size) 

        self.decoder_3 = SegU_decoder(
            input_resolution=(d_base_feat_size,d_base_feat_size),
            in_out_chan=[dims[3], dims[3]],
            heads=heads[3], reduction_ratios=reduction_ratios[3], token_mlp_mode=token_mlp_mode, n_class=num_classes, is_last=False
        )

        self.decoder_2 = SegU_decoder(
            input_resolution=(d_base_feat_size*2, d_base_feat_size*2), 
            in_out_chan=[dims[3] + dims[2], dims[2]],
            heads=heads[2], reduction_ratios=reduction_ratios[2], token_mlp_mode=token_mlp_mode, n_class=num_classes, is_last=False
        )
        
        self.decoder_1 = SegU_decoder(
            input_resolution=(d_base_feat_size*4, d_base_feat_size*4), 
            in_out_chan=[dims[2] + dims[1], dims[1]],
            heads=heads[1], reduction_ratios=reduction_ratios[1], token_mlp_mode=token_mlp_mode, n_class=num_classes, is_last=False
        )
        
        self.decoder_0 = SegU_decoder(
            input_resolution=(d_base_feat_size*8,d_base_feat_size*8), 
            in_out_chan=[dims[1] + dims[0], dims[0]], 
            heads=heads[0], reduction_ratios=reduction_ratios[0], token_mlp_mode=token_mlp_mode, n_class=num_classes, is_last=True
        )

        if not encoder_pretrained:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        encoder_outs = self.backbone(x)

        bridge_outs = self.bridge(encoder_outs) 
        
        b_val, c4_val, h4_val, w4_val = bridge_outs[3].shape
        tmp_3 = self.decoder_3(bridge_outs[3].permute(0,2,3,1).view(b_val, -1, c4_val), x2=None) 

        tmp_2 = self.decoder_2(tmp_3, bridge_outs[2])

        tmp_1 = self.decoder_1(tmp_2, bridge_outs[1])

        tmp_0 = self.decoder_0(tmp_1, bridge_outs[0])

        return tmp_0 