import torch.nn as nn
from typing import Dict, Any, List, Union 
import torch

from .unet import UNet
from .attention_unet import AttentionUNet
from .transatt_unet import TransAttUNet
from .raunet import RAUNet
from .da_transformer import DA_Transformer, get_da_transformer_config
from .unet_transformer import U_Transformer
from .uctransnet import UCTransNet, get_uctransnet_config
from .multiresunet import MultiResUnet
from .nested_unet import NestedUNet
from .missformer import MISSFormer

from .vnet import VNet
from .u2net import U2NET, U2NETP
from .swin_unet_v2 import SwinTransformerSys 
from .resunet import ResUnet

from .wranet import WRANet
from .egeunet import EGEUNet
from .unext import UNext, UNext_S
from .mmunet import MMUNet
from .medt_net import ResAxialAttentionUNet, axialunet, gated, MedT, logo

_model_entries: Dict[str, Union[type(nn.Module), Any]] = {
    'unet': UNet,
    'attention_unet': AttentionUNet,
    'transatt_unet': TransAttUNet,
    'raunet': RAUNet,
    'da_transformer': DA_Transformer,
    'unet_transformer': U_Transformer,
    'uctransnet': UCTransNet,
    'multiresunet': MultiResUnet,
    'nested_unet': NestedUNet,
    'missformer': MISSFormer,
    'vnet': VNet,
    'u2net': U2NET,
    'u2netp': U2NETP,
    'swin_unet_v2': SwinTransformerSys,
    'resunet': ResUnet,
    'wranet': WRANet,
    'egeunet': EGEUNet,
    'unext': UNext,
    'unext_s': UNext_S, 
    'mmunet': MMUNet,
    'axialunet': axialunet,
    'gated': gated, 
    'medt': MedT, 
    'logo': logo, 
}

_config_functions = {
    'da_transformer': get_da_transformer_config,
    'uctransnet': get_uctransnet_config,
}

def list_models() -> List[str]:
    """Returns a list of all available model names."""
    return sorted(list(_model_entries.keys()))

def get_model_config(model_name: str, **kwargs) -> Dict[str, Any]:
    """
    Gets the default configuration for a model if available.
    
    Args:
        model_name (str): Name of the model.
        **kwargs: Additional arguments for the config function.
    
    Returns:
        dict: Model configuration or empty dict if no config function exists.
    """
    if model_name in _config_functions:
        return _config_functions[model_name](**kwargs)
    return {}

def create_model(model_name: str, pretrained: bool = False, **kwargs) -> nn.Module:
    """
    Instantiates a UNet variant model, handling model-specific constructor arguments.
    """
    _model_name_lower = model_name.lower()
    if _model_name_lower not in _model_entries:
        raise ValueError(f"Unknown model: '{model_name}'. Available models: {list_models()}")

    model_factory_or_class = _model_entries[_model_name_lower]
    
    in_channels = kwargs.pop('in_channels', 3)
    num_classes = kwargs.pop('num_classes', 1)
    image_size = kwargs.pop('image_size', None)
    depth = kwargs.pop('depth', 5) 
    model_args = {}

    if _model_name_lower in _config_functions:
        model_args.update(get_model_config(_model_name_lower))

    if _model_name_lower == 'unet':
        model_args['in_channels'] = in_channels
        model_args['num_classes'] = num_classes
        
    elif _model_name_lower == 'attention_unet':
        model_args['in_channels'] = in_channels
        model_args['num_classes'] = num_classes
        model_args['depth'] = depth
        
    elif _model_name_lower == 'transatt_unet':
        model_args['in_channels'] = in_channels
        model_args['num_classes'] = num_classes
        model_args['depth'] = depth
        
    elif _model_name_lower == 'raunet':
        model_args['in_channels'] = in_channels
        model_args['num_classes'] = num_classes
        model_args['depth'] = depth

    elif _model_name_lower == 'da_transformer':
        config = get_da_transformer_config()
        model = model_factory_or_class(in_channels, num_classes, config, **kwargs)
        if pretrained:
            print(f"Warning: Pre-trained weights for {model_name} are not yet implemented.")
        return model 

    elif _model_name_lower == 'uctransnet':
        config = get_uctransnet_config()
        if image_size is None:
            raise ValueError(f"Model '{model_name}' requires 'image_size' parameter in config.")
        
        model_args['config'] = config
        model_args['in_channels'] = in_channels
        model_args['num_classes'] = num_classes
        model_args['img_size'] = image_size 
        model_args['vis'] = kwargs.pop('vis', False)
        
    elif _model_name_lower == 'multiresunet':
        model_args['in_channels'] = in_channels
        model_args['num_classes'] = num_classes
        model_args['depth'] = depth
        
    elif _model_name_lower == 'nested_unet':
        model_args['in_channels'] = in_channels
        model_args['num_classes'] = num_classes
        model_args['depth'] = depth
        model_args['deep_supervision'] = kwargs.pop('deep_supervision', False)
        
    elif _model_name_lower == 'missformer':
        model_args['in_channels'] = in_channels
        model_args['num_classes'] = num_classes
        model_args['depth'] = depth
        
    elif _model_name_lower == 'vnet':
        model_args['elu'] = kwargs.pop('elu', True)
        model_args['nll'] = kwargs.pop('nll', False) 
        model_args['in_channels'] = in_channels
        model_args['num_classes'] = num_classes
        
    elif _model_name_lower in ['u2net', 'u2netp']:
        model_args['in_ch'] = in_channels
        model_args['out_ch'] = num_classes
        
    elif _model_name_lower == 'swin_unet_v2':
        if image_size is None:
            raise ValueError(f"Model '{model_name}' requires 'image_size' parameter in config.")
        model_args['img_size'] = image_size
        model_args['in_chans'] = in_channels
        model_args['num_classes'] = num_classes
        
    elif _model_name_lower == 'resunet':
        model_args['in_channels'] = in_channels
        model_args['num_classes'] = num_classes
        model_args['filters'] = kwargs.pop('filters', [64, 128, 256, 512])
        
    elif _model_name_lower == 'wranet':
        model_args['in_channels'] = in_channels
        model_args['num_classes'] = num_classes
        model_args['feature_channels'] = kwargs.pop('feature_channels', 128)
        
    elif _model_name_lower == 'egeunet':
        model_args['in_channels'] = in_channels
        model_args['num_classes'] = num_classes
        model_args['c_list'] = kwargs.pop('c_list', None) 
        model_args['bridge'] = kwargs.pop('bridge', True)
        model_args['gt_ds'] = kwargs.pop('gt_ds', True) 
        model_args['image_size'] = image_size 
        
    elif _model_name_lower in ['unext', 'unext_s']:
        model_args['input_channels'] = in_channels
        model_args['num_classes'] = num_classes
        model_args['img_size'] = image_size if image_size is not None else 224 
        model_args['embed_dims'] = kwargs.pop('embed_dims', None)
        model_args['num_heads'] = kwargs.pop('num_heads', None)
        model_args['mlp_ratios'] = kwargs.pop('mlp_ratios', None)
        model_args['qkv_bias'] = kwargs.pop('qkv_bias', False)
        model_args['qk_scale'] = kwargs.pop('qk_scale', None)
        model_args['drop_rate'] = kwargs.pop('drop_rate', 0.0)
        model_args['attn_drop_rate'] = kwargs.pop('attn_drop_rate', 0.0)
        model_args['drop_path_rate'] = kwargs.pop('drop_path_rate', 0.0)
        model_args['norm_layer'] = kwargs.pop('norm_layer', nn.LayerNorm) 
        model_args['depths'] = kwargs.pop('depths', None)
        model_args['sr_ratios'] = kwargs.pop('sr_ratios', None)
        
    elif _model_name_lower == 'mmunet':
        model_args['in_channels'] = in_channels
        model_args['num_classes'] = num_classes
        model_args['base_channels'] = kwargs.pop('base_channels', 96)
        model_args['bilinear'] = kwargs.pop('bilinear', True)
        model_args['layer_scale_init_value'] = kwargs.pop('layer_scale_init_value', 1e-6)
        model_args['se_ratio'] = kwargs.pop('se_ratio', 0.25)
        
    elif _model_name_lower in ['axialunet', 'gated', 'medt', 'logo']:
        model_args['num_classes'] = num_classes
        model_args['img_size'] = image_size if image_size is not None else 128 
        model_args['in_channels'] = in_channels 
        model_args['layers'] = kwargs.pop('layers', [1, 2, 4, 1])
        model_args['s'] = kwargs.pop('s', 0.125)
        model_args['groups'] = kwargs.pop('groups', 8)
        model_args['width_per_group'] = kwargs.pop('width_per_group', 64)
        model_args['norm_layer'] = kwargs.pop('norm_layer', nn.BatchNorm2d)
        model_args['zero_init_residual'] = kwargs.pop('zero_init_residual', True)
        model_args['replace_stride_with_dilation'] = kwargs.pop('replace_stride_with_dilation', None)
        
    else:
        model_args['in_channels'] = in_channels
        model_args['num_classes'] = num_classes

    if _model_name_lower == 'uctransnet':
        model_args['vis'] = kwargs.pop('vis', False)

    model_args.update(kwargs)

    if _model_name_lower in ['axialunet', 'gated', 'medt', 'logo']:
        model = model_factory_or_class(pretrained=pretrained, **model_args)
    else:
        model = model_factory_or_class(**model_args)

    if pretrained:
        print(f"Warning: Pre-trained weights for {model_name} are not yet implemented.")

    return model

__all__ = [
    'UNet', 'AttentionUNet', 'TransAttUNet', 'RAUNet', 'DA_Transformer',
    'U_Transformer', 'UCTransNet', 'MultiResUnet', 'NestedUNet', 'MISSFormer',
    'VNet', 'U2NET', 'U2NETP', 'SwinTransformerSys', 'ResUnet',
    'WRANet', 'EGEUNet', 'UNext', 'UNext_S', 'MMUNet',
    'axialunet', 'gated', 'MedT', 'logo', 
    'get_da_transformer_config', 'get_uctransnet_config',
    'list_models', 'get_model_config', 'create_model'
]
