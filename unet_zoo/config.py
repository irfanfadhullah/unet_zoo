# unet_zoo/config.py
import torch
import os

class Config:
    """
    Centralized configuration object for UNet Zoo training and evaluation.
    This class is initialized with the parsed YAML configuration dictionary.
    """
    def __init__(self, overall_config_dict: dict):
        self.PROJECT_NAME = overall_config_dict['general']['project_name']
        self.WORKING_DIR = overall_config_dict['general']['working_dir']
        
        self.DATASET_DIR = overall_config_dict['data']['dataset_dir']
        self.NUM_WORKERS = overall_config_dict['data']['num_workers']
        self.IMAGE_SIZE = overall_config_dict['data'].get('image_size', 512)

        self.EPOCHS = overall_config_dict['training']['epochs']
        self.BATCH_SIZE = overall_config_dict['training']['batch_size']
        self.LEARNING_RATE = overall_config_dict['training']['learning_rate']
        self.EARLY_STOPPING_PATIENCE = overall_config_dict['training']['early_stopping_patience']
        self.LR_SCHEDULER_PATIENCE = overall_config_dict['training']['lr_scheduler_patience']
        self.LR_SCHEDULER_FACTOR = overall_config_dict['training']['lr_scheduler_factor']
        self.MIN_LR = overall_config_dict['training']['min_lr']
        self.NUM_CLASSES = overall_config_dict['training']['num_classes']

        self.USE_MULTI_GPU = overall_config_dict['gpu']['use_multi_gpu']
        self.GPU_IDS = overall_config_dict['gpu']['gpu_ids']
        self.SINGLE_GPU_ID = overall_config_dict['gpu']['single_gpu_id']
        self.MULTI_GPU_STRATEGY = overall_config_dict['gpu'].get('multi_gpu_strategy', 'DataParallel') 
        
        if torch.cuda.is_available():
            if self.USE_MULTI_GPU and len(self.GPU_IDS) > 0:
                self.DEVICE = torch.device(f"cuda:{self.GPU_IDS[0]}")
            elif self.SINGLE_GPU_ID is not None and torch.cuda.device_count() > self.SINGLE_GPU_ID:
                self.DEVICE = torch.device(f"cuda:{self.SINGLE_GPU_ID}")
            else:
                self.DEVICE = torch.device("cuda:0") 
        else:
            self.DEVICE = torch.device("cpu")

        self.RUN_TIMESTAMP = overall_config_dict.get('run_timestamp', datetime.datetime.now().strftime("%Y%m%d-%H%M%S_fallback"))
        
        self.BASE_RUN_DIR = os.path.join(self.WORKING_DIR, f"overall_runs_{self.RUN_TIMESTAMP}")
        self.OVERALL_LOG_DIR = os.path.join(self.BASE_RUN_DIR, "overall_logs")
        self.TENSORBOARD_BASE_DIR = os.path.join(self.BASE_RUN_DIR, 'tensorboard_logs')

        os.makedirs(self.OVERALL_LOG_DIR, exist_ok=True)
        os.makedirs(self.TENSORBOARD_BASE_DIR, exist_ok=True)

    def get_device_info(self) -> str:
        """Returns a string describing the active device."""
        if self.DEVICE.type == 'cuda':
            return f"CUDA ({torch.cuda.get_device_name(self.DEVICE)})"
        return "CPU"

import datetime