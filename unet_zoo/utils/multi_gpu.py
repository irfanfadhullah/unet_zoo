# unet_zoo/utils/multi_gpu.py
import torch
import torch.nn as nn
import os
import copy
from unet_zoo.config import Config 

class MultiGPUManager:
    """Manager class for handling multi-GPU operations with single-GPU visualization support"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def _get_unwrapped_model(self, model: nn.Module) -> nn.Module:
        """Returns the underlying model if it's wrapped by DataParallel/DistributedDataParallel."""
        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return model.module
        return model

    def setup_model_for_gpu(self, model: nn.Module) -> nn.Module:
        """
        Setup model for single or multi-GPU training.
        Supports nn.DataParallel. For nn.DistributedDataParallel, a more complex setup
        (e.g., torch.distributed.launch, init_process_group) is required outside this manager.
        """
        model = model.to(self.config.DEVICE)
        
        if self.config.USE_MULTI_GPU and len(self.config.GPU_IDS) > 1:
            if self.config.MULTI_GPU_STRATEGY == "DataParallel":
                model = nn.DataParallel(model, device_ids=self.config.GPU_IDS)
                print(f"Model wrapped with DataParallel on GPUs: {self.config.GPU_IDS}")
            else:
                print(f"WARNING: Multi-GPU strategy '{self.config.MULTI_GPU_STRATEGY}' not supported by this manager. Falling back to single GPU.")
        else:
            print(f"Model running on single GPU: {self.config.DEVICE}")
            
        return model
    
    def save_model_state(self, model: nn.Module, path: str):
        """Save model state dict, handling multi-GPU wrapper"""
        unwrapped_model = self._get_unwrapped_model(model)
        torch.save(unwrapped_model.state_dict(), path)
    
    def _strip_module_prefix(self, state_dict: dict) -> dict:
        """Removes 'module.' prefix from state dict keys if present."""
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_key = k[len('module.'):]
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        return new_state_dict
    
    def load_model_state(self, model: nn.Module, path: str, device: torch.device):
        """
        Load model state dict, robustly handling 'module.' prefix.
        
        Args:
            model: Target model to load weights into. Can be wrapped or unwrapped.
            path: Path to checkpoint file.
            device: Target device to map loaded weights to.
        """
        if not os.path.exists(path):
            print(f"Warning: Checkpoint file not found at {path}. Model weights not loaded.")
            return model

        state_dict = torch.load(path, map_location=device)
        
        state_dict = self._strip_module_prefix(state_dict)

        unwrapped_model = self._get_unwrapped_model(model)
        try:
            unwrapped_model.load_state_dict(state_dict)
            print(f"Model weights loaded successfully from {path} onto {device}.")
        except RuntimeError as e:
            print(f"Error loading state_dict: {e}")
            print("This often happens due to a mismatch in model architecture (e.g., different layers or shapes) or leftover 'module.' prefixes.")
            print("Attempting to load with `strict=False` (might load partial weights).")
            try:
                unwrapped_model.load_state_dict(state_dict, strict=False)
                print("Model weights loaded with strict=False (partial match).")
            except Exception as e_non_strict:
                print(f"Failed to load even with strict=False: {e_non_strict}")
                print("Model state_dict could not be loaded. Model will use randomized weights.")
        
        return model
    
    def get_model_for_inference(self, model: nn.Module, device: torch.device) -> nn.Module:
        """
        Get the underlying model for single-GPU inference, ensuring proper device placement.
        Sets model to evaluation mode.
        """
        unwrapped_model = self._get_unwrapped_model(model)
        unwrapped_model = unwrapped_model.to(device)
        unwrapped_model.eval()
        return unwrapped_model