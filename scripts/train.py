import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import datetime

from unet_zoo.models import create_model, list_models 
from unet_zoo.data.datasets import BoneDataset 
from unet_zoo.utils.logger import Logger
from unet_zoo.utils.multi_gpu import MultiGPUManager 
from unet_zoo.utils.metrics import check_dataset_integrity 
from unet_zoo.utils.training_loop import train_model
from unet_zoo.utils.visualize import plot_training_comparison, save_all_test_results
from unet_zoo.config import Config as TrainingConfig 


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train UNet variants for image segmentation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config', type=str, default='configs/default_train_config.yaml', 
                        help='Path to the YAML configuration file.')
    return parser.parse_args()

def setup_paths(working_dir, model_name, timestamp, base_run_dir):
    """Dynamically set up output directories and paths within the main run directory."""
    model_run_dir = os.path.join(base_run_dir, model_name) 

    checkpoint_dir = os.path.join(model_run_dir, 'checkpoints')
    log_dir = os.path.join(model_run_dir, 'logs')
    results_dir = os.path.join(model_run_dir, 'results')

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    return {
        'run_dir': model_run_dir,
        'checkpoint_dir': checkpoint_dir,
        'log_dir': log_dir,
        'results_dir': results_dir,
        'training_log_path': os.path.join(log_dir, 'training_log.txt'),
        'test_results_path': os.path.join(results_dir, 'test_results.csv'),
        'model_checkpoint_paths': {
            'best': os.path.join(checkpoint_dir, f'{model_name}_best.pth'),
            'last': os.path.join(checkpoint_dir, f'{model_name}_last.pth'),
        }
    }

if __name__ == "__main__":
    args = parse_arguments()

    with open(args.config, 'r') as f:
        overall_config = yaml.safe_load(f)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    overall_config['run_timestamp'] = timestamp 
    
    config = TrainingConfig(overall_config) 

    dataset_dir = config.DATASET_DIR
    working_dir = config.WORKING_DIR 
    batch_size = config.BATCH_SIZE
    learning_rate = config.LEARNING_RATE
    epochs = config.EPOCHS
    num_workers = config.NUM_WORKERS 
    early_stopping_patience = config.EARLY_STOPPING_PATIENCE
    lr_scheduler_patience = config.LR_SCHEDULER_PATIENCE
    lr_scheduler_factor = config.LR_SCHEDULER_FACTOR
    min_lr = config.MIN_LR
    
    use_multi_gpu = config.USE_MULTI_GPU 
    gpu_ids = config.GPU_IDS
    single_gpu_id = config.SINGLE_GPU_ID
    
    models_to_train = overall_config['models']['names'] 
    input_image_size = config.IMAGE_SIZE
    num_classes = config.NUM_CLASSES 

    
    gpu_manager = MultiGPUManager(config=config) 
    device = config.DEVICE

    overall_logger = Logger(os.path.join(config.OVERALL_LOG_DIR, f"overall_training_{config.RUN_TIMESTAMP}.txt")) 

    overall_logger.log_both(f"Configuration loaded from: {args.config}")
    overall_logger.log_both(f"Starting UNet Zoo training run: {config.RUN_TIMESTAMP}")
    overall_logger.log_both(f"  Project Name: {config.PROJECT_NAME}")
    overall_logger.log_both(f"  Base Run Directory: {config.BASE_RUN_DIR}")
    overall_logger.log_both(f"  Dataset directory: {dataset_dir}")
    overall_logger.log_both(f"  Batch size: {batch_size}")
    overall_logger.log_both(f"  Learning rate: {learning_rate}")
    overall_logger.log_both(f"  Epochs: {epochs}")
    overall_logger.log_both(f"  Models to train: {models_to_train}")
    overall_logger.log_both(f"  Device Configuration: {config.get_device_info()}") 
    overall_logger.log_both(f"  Multi-GPU enabled: {use_multi_gpu}")
    if use_multi_gpu:
        overall_logger.log_both(f"  GPU IDs: {gpu_ids}")
    overall_logger.log_both(f"  Early Stopping Patience: {early_stopping_patience} epochs")
    overall_logger.log_both(f"  LR Scheduler Patience: {lr_scheduler_patience} epochs")
    overall_logger.log_both(f"  Input Image Size: {input_image_size}x{input_image_size}")
    overall_logger.log_both(f"  Number of Classes: {num_classes}")
    
    check_dataset_integrity(dataset_dir, overall_logger)

    train_dataset = BoneDataset(dataset_dir, split='train')
    val_dataset = BoneDataset(dataset_dir, split='valid')

    overall_logger.log_both(f"Train dataset size: {len(train_dataset)}")
    overall_logger.log_both(f"Validation dataset size: {len(val_dataset)}")
    
    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True,
                                drop_last=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True,
                                drop_last=False)

    models_dict = {}
    optimizers_dict = {}
    metrics_history = {} 
    early_stopped_flags = {} 
    
    criterion = nn.BCEWithLogitsLoss()

    overall_logger.log_both("\n" + "="*80)
    overall_logger.log_both("STARTING MULTI-MODEL COMPARISON TRAINING")
    overall_logger.log_both("="*80)
    
    for model_name_key in models_to_train:
        overall_logger.log_both(f"\n🚀 Training {model_name_key.upper()}...")
        
        paths = setup_paths(working_dir, model_name_key, config.RUN_TIMESTAMP, config.BASE_RUN_DIR) 
        
        model_params = overall_config['models'].get('params', {}).get(model_name_key, {})
        
        model_params.setdefault('in_channels', 3)
        model_params.setdefault('num_classes', num_classes) 
        model_params.setdefault('image_size', input_image_size) 
        
        model = create_model(model_name_key, **model_params)
        model = gpu_manager.setup_model_for_gpu(model)
        models_dict[model_name_key] = model 

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        optimizers_dict[model_name_key] = optimizer

        def count_parameters(model_instance):
            if isinstance(model_instance, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                return sum(p.numel() for p in model_instance.module.parameters())
            return sum(p.numel() for p in model_instance.parameters())
        overall_logger.log_both(f"{model_name_key.upper()} parameters: {count_parameters(model):,}")

        model_logger = Logger(paths['training_log_path'])

        train_losses, train_dcs, val_losses, val_dcs, early_stopped = train_model(
            model=model, 
            train_dataloader=train_dataloader, 
            val_dataloader=val_dataloader, 
            optimizer=optimizer, 
            criterion=criterion, 
            config=config, 
            model_name=model_name_key, 
            best_checkpoint_path=paths['model_checkpoint_paths']['best'], 
            last_checkpoint_path=paths['model_checkpoint_paths']['last'], 
            logger=model_logger, 
            gpu_manager=gpu_manager 
        )
        
        metrics_history[model_name_key] = (train_losses, train_dcs, val_losses, val_dcs)
        early_stopped_flags[model_name_key] = early_stopped
        
        model_logger.close() 

    if len(metrics_history) > 0:
        overall_logger.log_both("\n📊 Plotting training comparison...")
        plot_training_comparison(
            epochs, 
            metrics_history,  
            early_stopped_flags,  
            overall_logger, 
            os.path.join(config.BASE_RUN_DIR, "overall_results")
        )

    overall_logger.log_both("\n✅ Multi-model training completed! Check overall logs for details.")
    overall_logger.close()