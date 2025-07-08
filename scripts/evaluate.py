
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import datetime

from unet_zoo.models import create_model
from unet_zoo.data.datasets import BoneDataset 
from unet_zoo.utils.logger import Logger
from unet_zoo.utils.multi_gpu import MultiGPUManager
from unet_zoo.utils.training_loop import evaluate_model
from unet_zoo.utils.visualize import visualize_inference_comparison, save_all_test_results

def parse_arguments():
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description='Evaluate UNet variants on the test set and visualize inference.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the YAML configuration file for evaluation.')
    return parser.parse_args()

def setup_eval_paths(output_base_dir, timestamp):
    """Dynamically set up output directories for evaluation results."""
    eval_run_dir = os.path.join(output_base_dir, f"evaluation_{timestamp}")
    os.makedirs(eval_run_dir, exist_ok=True)
    
    results_dir = os.path.join(eval_run_dir, 'results')
    log_dir = os.path.join(eval_run_dir, 'logs')

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    return {
        'results_dir': results_dir,
        'log_path': os.path.join(log_dir, 'evaluation_log.txt'),
        'test_results_path': os.path.join(results_dir, 'final_test_metrics.csv')
    }

if __name__ == "__main__":
    args = parse_arguments()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    dataset_dir = config['data']['dataset_dir']
    output_base_dir = config['evaluation']['output_base_dir']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    num_inference_samples = config['evaluation']['num_inference_samples']
    models_to_evaluate = config['models_to_evaluate'] 
    input_image_size = config['data'].get('image_size', 512) 

    eval_paths = setup_eval_paths(output_base_dir, timestamp)
    
    use_multi_gpu_eval = config['gpu'].get('use_multi_gpu', False)
    gpu_ids_eval = config['gpu'].get('gpu_ids', [0])
    single_gpu_id_eval = config['gpu'].get('single_gpu_id', 0)

    gpu_manager = MultiGPUManager(
        use_multi_gpu=use_multi_gpu_eval,
        gpu_ids=gpu_ids_eval,
        single_gpu_id=single_gpu_id_eval
    )
    device = gpu_manager.get_device()
    
    logger = Logger(eval_paths['log_path'])

    logger.log_both(f"Configuration loaded from: {args.config}")
    logger.log_both(f"Starting UNet Zoo evaluation run: {timestamp}")
    logger.log_both(f"  Dataset directory: {dataset_dir}")
    logger.log_both(f"  Batch size: {batch_size}")
    logger.log_both(f"  Models to evaluate: {models_to_evaluate}")
    logger.log_both(f"  Device Configuration: {device}")
    logger.log_both(f"  Input Image Size: {input_image_size}x{input_image_size}")
    
    test_dataset = BoneDataset(dataset_dir, split='test')
    logger.log_both(f"Test dataset size: {len(test_dataset)}")
    
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=False)
    
    criterion = nn.BCEWithLogitsLoss() 

    test_results_summary = {}
    models_for_visualization = {} 
    
    logger.log_both("\n" + "="*50)
    logger.log_both("LOADING MODELS AND PERFORMING EVALUATION")
    logger.log_both("="*50)

    for model_info in models_to_evaluate:
        model_name = model_info['name']
        checkpoint_path = model_info['checkpoint']
        
        logger.log_both(f"\n📈 Evaluating {model_name.upper()} from checkpoint: {checkpoint_path}...")

        model_params = config['models'].get('params', {}).get(model_name, {})
        model_params.setdefault('in_channels', 3)
        model_params.setdefault('num_classes', 1)
        model_params.setdefault('image_size', input_image_size)

        model = create_model(model_name, **model_params)
        
        model = gpu_manager.load_model_state(model, checkpoint_path, device)
        
        test_loss, test_dc = evaluate_model(
            model, test_dataloader, criterion, device, model_name.upper(), logger
        )
        test_results_summary[model_name] = (test_loss, test_dc)
        models_for_visualization[model_name] = checkpoint_path 

    if len(test_results_summary) > 0:
        logger.log_both(f"\nSaving final test results to: {eval_paths['test_results_path']}")
        save_all_test_results(
            test_results_summary,  
            eval_paths['test_results_path'], 
            logger
        )

    if len(models_for_visualization) > 0: 
        logger.log_both("\n🔍 Visualizing inference comparison for selected models...")
        visualize_inference_comparison(
            test_dataset, 
            models_for_visualization,
            num_inference_samples, 
            device,
            logger,
            eval_paths['results_dir'], 
            gpu_manager,
            image_size=input_image_size
        )
    
    logger.log_both("\n✅ Evaluation completed!")
    logger.log_both(f"All outputs saved to: {eval_paths['results_dir']}")
    logger.close()