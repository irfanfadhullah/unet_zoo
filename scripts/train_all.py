import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import datetime
import random

from unet_zoo.models import create_model, list_models 
from unet_zoo.data.datasets import BoneDataset 
from unet_zoo.utils.logger import Logger
from unet_zoo.utils.multi_gpu import MultiGPUManager
from unet_zoo.utils.metrics import check_dataset_integrity, dice_coefficient
from unet_zoo.utils.training_loop import train_model
from unet_zoo.config import Config as TrainingConfig

from unet_zoo.utils.visualize import (
    plot_training_comparison, 
    save_all_test_results,
    visualize_inference_comparison
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train UNet variants for image segmentation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config', type=str, default='configs/default_train_config.yaml', 
                        help='Path to the YAML configuration file.')
    parser.add_argument('--skip-training', action='store_true', 
                        help='Skip training phase and only run evaluation.')
    parser.add_argument('--skip-evaluation', action='store_true', 
                        help='Skip evaluation phase and only run training.')
    parser.add_argument('--visualization-samples', type=int, default=5,
                        help='Number of samples to use for visualization comparison.')
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
        'evaluation_log_path': os.path.join(log_dir, 'evaluation_log.txt'),
        'test_results_path': os.path.join(results_dir, 'test_results.csv'),
        'model_checkpoint_paths': {
            'best': os.path.join(checkpoint_dir, f'{model_name}_best.pth'),
            'last': os.path.join(checkpoint_dir, f'{model_name}_last.pth'),
        }
    }

def evaluate_model(model, test_dataloader, criterion, device, logger, gpu_manager):
    """Evaluate a single model on test set."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (images, masks, _) in enumerate(test_dataloader):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            if isinstance(outputs, dict):
                if 'main' in outputs: 
                    main_outputs = outputs['main']
                elif 'out' in outputs: 
                    main_outputs = outputs['out']
                else:
                    main_outputs = list(outputs.values())[0]
            elif isinstance(outputs, (list, tuple)):
                main_outputs = outputs[-1] if isinstance(outputs, list) else outputs[0]
            else:
                main_outputs = outputs
            
            loss = criterion(main_outputs, masks)
            total_loss += loss.item()
            
            dice_score = dice_coefficient(main_outputs, masks)
            total_dice += dice_score.item()
            
            num_batches += 1
            
            if batch_idx % 20 == 0:
                logger.log_both(f"  Batch {batch_idx}/{len(test_dataloader)}: "
                               f"Loss={loss.item():.6f}, DICE={dice_score.item():.6f}")
    
    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches
    
    return avg_loss, avg_dice

def run_evaluation_phase(models_to_evaluate, overall_config, config, gpu_manager, 
                        test_dataloader, criterion, overall_logger):
    """Run evaluation phase for all trained models."""
    overall_logger.log_both("\n" + "="*80)
    overall_logger.log_both("STARTING EVALUATION PHASE")
    overall_logger.log_both("="*80)
    
    all_test_results = {}
    model_configs_for_visualization = []
    
    for model_name in models_to_evaluate:
        overall_logger.log_both(f"\n🔍 Evaluating {model_name.upper()}...")
        
        paths = setup_paths(config.WORKING_DIR, model_name, config.RUN_TIMESTAMP, config.BASE_RUN_DIR)
        
        best_checkpoint_path = paths['model_checkpoint_paths']['best']
        if not os.path.exists(best_checkpoint_path):
            overall_logger.log_both(f"❌ Best checkpoint not found for {model_name}: {best_checkpoint_path}")
            continue
            
        model_params = overall_config['models'].get('params', {}).get(model_name, {})
        model_params.setdefault('in_channels', 3)
        model_params.setdefault('num_classes', config.NUM_CLASSES)
        model_params.setdefault('image_size', config.IMAGE_SIZE)
        
        try:
            model = create_model(model_name, **model_params)
            model = gpu_manager.load_model_state(model, best_checkpoint_path, config.DEVICE)
            model = gpu_manager.setup_model_for_gpu(model)
            
            eval_logger = Logger(paths['evaluation_log_path'])
            eval_logger.log_both(f"Starting evaluation for {model_name}")
            eval_logger.log_both(f"Loading checkpoint: {best_checkpoint_path}")
            
            test_loss, test_dice = evaluate_model(
                model, test_dataloader, criterion, config.DEVICE, eval_logger, gpu_manager
            )
            
            all_test_results[model_name] = (test_loss, test_dice)
            
            model_configs_for_visualization.append({
                'name': model_name,
                'checkpoint': best_checkpoint_path,
                'params': model_params
            })
            
            overall_logger.log_both(f"✅ {model_name.upper()} Test Results:")
            overall_logger.log_both(f"   Test Loss: {test_loss:.6f}")
            overall_logger.log_both(f"   Test DICE: {test_dice:.6f}")
            
            eval_logger.log_both(f"Final Results - Loss: {test_loss:.6f}, DICE: {test_dice:.6f}")
            eval_logger.close()
            
        except Exception as e:
            overall_logger.log_both(f"❌ Error evaluating {model_name}: {str(e)}")
            continue
    
    return all_test_results, model_configs_for_visualization

def run_visualization_phase(model_configs_for_visualization, test_dataset, val_dataset, 
                           config, gpu_manager, overall_logger, num_samples=5):
    """Run visualization comparison phase."""
    if not model_configs_for_visualization:
        overall_logger.log_both("⚠️ No models available for visualization.")
        return
        
    overall_logger.log_both("\n" + "="*80)
    overall_logger.log_both("STARTING VISUALIZATION PHASE")
    overall_logger.log_both("="*80)
    
    vis_dir = os.path.join(config.BASE_RUN_DIR, "visualization_results")
    os.makedirs(vis_dir, exist_ok=True)
    
    overall_logger.log_both(f"\n📊 Creating visualization comparisons on test set...")
    visualize_inference_comparison(
        dataset=test_dataset,
        model_configs=model_configs_for_visualization,
        num_samples=num_samples,
        config_device=config.DEVICE,
        logger=overall_logger,
        save_dir=os.path.join(vis_dir, "test_set"),
        gpu_manager=gpu_manager,
        general_image_size=config.IMAGE_SIZE,
        general_in_channels=3,
        general_num_classes=config.NUM_CLASSES
    )
    
    overall_logger.log_both(f"\n📊 Creating visualization comparisons on validation set...")
    visualize_inference_comparison(
        dataset=val_dataset,
        model_configs=model_configs_for_visualization,
        num_samples=num_samples,
        config_device=config.DEVICE,
        logger=overall_logger,
        save_dir=os.path.join(vis_dir, "validation_set"),
        gpu_manager=gpu_manager,
        general_image_size=config.IMAGE_SIZE,
        general_in_channels=3,
        general_num_classes=config.NUM_CLASSES
    )

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
    overall_logger.log_both(f"  Models to train: {models_to_train}")
    overall_logger.log_both(f"  Device Configuration: {config.get_device_info()}")
    overall_logger.log_both(f"  Skip Training: {args.skip_training}")
    overall_logger.log_both(f"  Skip Evaluation: {args.skip_evaluation}")
    overall_logger.log_both(f"  Visualization Samples: {args.visualization_samples}")
    
    check_dataset_integrity(dataset_dir, overall_logger)

    train_dataset = BoneDataset(dataset_dir, split='train')
    val_dataset = BoneDataset(dataset_dir, split='valid')
    test_dataset = BoneDataset(dataset_dir, split='test')

    overall_logger.log_both(f"Train dataset size: {len(train_dataset)}")
    overall_logger.log_both(f"Validation dataset size: {len(val_dataset)}")
    overall_logger.log_both(f"Test dataset size: {len(test_dataset)}")
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    models_dict = {}
    optimizers_dict = {}
    metrics_history = {} 
    early_stopped_flags = {} 
    
    criterion = nn.BCEWithLogitsLoss()

    if not args.skip_training:
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
    
    if not args.skip_evaluation:
        all_test_results, model_configs_for_visualization = run_evaluation_phase(
            models_to_train, overall_config, config, gpu_manager, 
            test_dataloader, criterion, overall_logger
        )
        
        if all_test_results:
            overall_results_dir = os.path.join(config.BASE_RUN_DIR, "overall_results")
            os.makedirs(overall_results_dir, exist_ok=True)
            
            save_all_test_results(
                all_test_results,
                os.path.join(overall_results_dir, "all_test_results.txt"),
                overall_logger
            )
        
        run_visualization_phase(
            model_configs_for_visualization, 
            test_dataset, 
            val_dataset,
            config, 
            gpu_manager, 
            overall_logger, 
            num_samples=args.visualization_samples
        )
    
    overall_logger.log_both("\n" + "="*80)
    overall_logger.log_both("🎉 COMPLETE PIPELINE FINISHED!")
    overall_logger.log_both("="*80)
    overall_logger.log_both(f"📁 All results saved to: {config.BASE_RUN_DIR}")
    overall_logger.log_both(f"📊 Training plots: {os.path.join(config.BASE_RUN_DIR, 'overall_results')}")
    overall_logger.log_both(f"🔍 Evaluation results: {os.path.join(config.BASE_RUN_DIR, 'overall_results')}")
    overall_logger.log_both(f"📸 Visualization results: {os.path.join(config.BASE_RUN_DIR, 'visualization_results')}")
    
    overall_logger.close()
