import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
plt.ioff() 
from PIL import Image
from typing import Dict, List, Tuple, Union, Any 

from unet_zoo.config import Config 
from unet_zoo.utils.metrics import dice_coefficient
from unet_zoo.utils.multi_gpu import MultiGPUManager

from unet_zoo.models import create_model

from unet_zoo.models import UNet, VNet, U2NET, U2NETP, SwinTransformerSys, ResUnet
from unet_zoo.models import WRANet, EGEUNet, UNext, UNext_S, MMUNet, ResAxialAttentionUNet, medt_net
from unet_zoo.models import UCTransNet, NestedUNet


def _get_main_prediction_logits(
    outputs: Union[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, Any], List[torch.Tensor]],
    model_type: type
) -> torch.Tensor:
    """
    Extracts the main prediction logits from model outputs, handling various output formats.
    """
    if issubclass(model_type, (U2NET, U2NETP)):
        return outputs['main']
    elif issubclass(model_type, EGEUNet):
        if isinstance(outputs, dict):
            return outputs['out']
        else:
            return outputs
    elif issubclass(model_type, MMUNet):
        return outputs['out']
    elif issubclass(model_type, UCTransNet):
        if isinstance(outputs, tuple) and len(outputs) == 2:
            return outputs[0] 
        return outputs 
    elif issubclass(model_type, NestedUNet):
        if isinstance(outputs, list):
            return outputs[-1] 
        return outputs 
    else:
        return outputs

def visualize_inference_comparison(
    dataset, 
    model_configs: List[Dict[str, Union[str, Dict]]], 
    num_samples: int, 
    config_device: torch.device, 
    logger,
    save_dir: str,
    gpu_manager: MultiGPUManager,
    general_image_size: int = 512,
    general_in_channels: int = 3,
    general_num_classes: int = 1
):
    """
    Compare predictions from selected models - SINGLE GPU INFERENCE
    """
    inference_device = torch.device(f"cuda:{config_device.index}" if config_device.type == 'cuda' and config_device.index is not None else 'cpu')
    logger.log_both(f"Visualization running on: {inference_device}")

    loaded_models = {}
    model_unwrapped_types = {} 

    for model_entry in model_configs:
        model_name = model_entry['name']
        path = model_entry['checkpoint']
        model_params = model_entry.get('params', {}) 

        if not os.path.exists(path):
            logger.log_both(f"Warning: Checkpoint for {model_name} not found at {path}. Skipping visualization for this model.")
            continue
        
        model_create_params = {
            'in_channels': general_in_channels,
            'num_classes': general_num_classes,
            'image_size': general_image_size,
            **model_params 
        }

        try:
            model = create_model(model_name, **model_create_params)
            
            model = gpu_manager.load_model_state(model, path, inference_device)
            
            unwrapped_model = gpu_manager.get_model_for_inference(model, inference_device)
            unwrapped_model.eval()

            loaded_models[model_name] = unwrapped_model
            model_unwrapped_types[model_name] = type(unwrapped_model)
            logger.log_both(f"{model_name.replace('_', ' ').title()} loaded for inference on {next(unwrapped_model.parameters()).device}")
        except Exception as e:
            logger.log_both(f"Error loading or setting up {model_name} for visualization from {path}: {e}. Skipping this model.")
            continue 

    if not loaded_models:
        logger.log_both("No models loaded for visualization. Skipping inference comparison plots.")
        return

    logger.log_both(f"\nComparing trained models on {dataset.split} set ({num_samples} samples):")

    os.makedirs(save_dir, exist_ok=True)
    vis_results_path = os.path.join(save_dir, f'visual_comparison_{dataset.split}.txt')
    with open(vis_results_path, 'w') as f:
        f.write(f"Visual Comparison Results - {dataset.split} Set (Single GPU Inference)\n")
        f.write("="*60 + "\n")
        
        for i in range(num_samples):
            random_index = random.randint(0, len(dataset) - 1)
            current_image_tensor, current_mask_tensor, image_path_str = dataset[random_index]
            
            current_image_tensor = current_image_tensor.to(inference_device)
            current_mask_tensor = current_mask_tensor.to(inference_device)
            
            img_batch = current_image_tensor.unsqueeze(0) 
            
            sample_dices = {}
            predicted_logits_for_plotting = {} 
            
            with torch.no_grad():
                for model_name, model_instance in loaded_models.items():
                    outputs = model_instance(img_batch)
                    
                    main_pred_logits = _get_main_prediction_logits(outputs, model_unwrapped_types[model_name])
                    
                    if main_pred_logits.shape[1] > 1 and general_num_classes == 1:
                        print(f"Warning: Model {model_name} output {main_pred_logits.shape[1]} channels, but general_num_classes is 1. "
                              "Taking the first channel for Dice/plotting. Adjust if different logic is needed.")
                        main_pred_logits = main_pred_logits[:, 0:1, :, :]
                    
                    dc = dice_coefficient(main_pred_logits, current_mask_tensor.unsqueeze(0))
                    sample_dices[model_name] = dc.item()
                    predicted_logits_for_plotting[model_name] = main_pred_logits
            
            result_text = f"Image: {os.path.basename(image_path_str)}\n"
            for model_name, dc_score in sample_dices.items():
                result_text += f"  {model_name.replace('_', ' ').title()} DICE: {dc_score:.5f}\n"
            
            if sample_dices:
                winner_name = max(sample_dices, key=sample_dices.get)
                winner_score = sample_dices[winner_name]
                result_text += f"  🏆 Winner: {winner_name.replace('_', ' ').title()} (Dice: {winner_score:.5f})\n\n"
            else:
                result_text += "  (No models trained or loaded for comparison)\n\n"

            logger.log_both(result_text)
            f.write(result_text)
            
            img_display = current_image_tensor.cpu().detach()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_display = img_display * std + mean 
            img_display = torch.clamp(img_display, 0, 1).permute(1, 2, 0) 
            
            mask_display = current_mask_tensor.cpu().detach().squeeze(0) 
            
            num_cols = 2 + len(loaded_models)
            plt.figure(figsize=(num_cols * 5, 5)) 
            
            plt.subplot(1, num_cols, 1)
            plt.imshow(img_display)
            plt.title("Original Image")
            plt.axis('off')
            
            plot_idx = 2
            for model_name, logits_output in predicted_logits_for_plotting.items(): 
                pred_display = (torch.sigmoid(logits_output).squeeze().cpu().detach() > 0.5).float()
                if pred_display.dim() == 3 and pred_display.shape[0] == 1:
                    pred_display = pred_display.squeeze(0)
                elif pred_display.dim() == 4 and pred_display.shape[1] == 1:
                    pred_display = pred_display.squeeze(0).squeeze(0)

                plt.subplot(1, num_cols, plot_idx)
                plt.imshow(pred_display, cmap="gray", vmin=0, vmax=1)
                plt.title(f"{model_name.replace('_', ' ').title()}\n(Dice: {sample_dices[model_name]:.4f})")
                plt.axis('off')
                plot_idx += 1
            
            plt.subplot(1, num_cols, plot_idx)
            plt.imshow(mask_display, cmap="gray", vmin=0, vmax=1)
            plt.title("Ground Truth")
            plt.axis('off')
            
            plt.suptitle(f"Model Comparison: {os.path.basename(image_path_str)}")
            plt.tight_layout()
            
            vis_save_path = os.path.join(save_dir, f'comparison_{i+1}_{os.path.basename(image_path_str)}.png')
            plt.savefig(vis_save_path, dpi=150, bbox_inches='tight')
            logger.log_both(f"Comparison plot {i+1} saved to: {vis_save_path}")
            plt.close()

def plot_training_comparison(
    epochs: int,
    all_models_metrics: Dict[str, Tuple[List[float], List[float], List[float], List[float]]], 
    all_early_stopping_info: Dict[str, bool], 
    logger,
    save_dir: str
):
    """
    Plot comparison between all trained models - SAVE ONLY, NO DISPLAY
    """
    os.makedirs(save_dir, exist_ok=True)
    model_names = list(all_models_metrics.keys())
    
    markers = ['o', 's', '^', 'D', 'x', 'P', '*', 'h', 'v', 'X', '>', '<', 'p', 'H', '+', '|', '_', '.', ','] 
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'brown', 'magenta', 'lime', 'gold', 
              'teal', 'darkblue', 'darkgreen', 'darkred', 'darkorange', 'indigo', 'maroon', 'olive', 'pink']

    fig, axes = plt.subplots(2, 2, figsize=(18, 12)) 
    
    axes_flat = axes.flatten()

    plot_titles = [
        'Training Loss Comparison', 'Validation Loss Comparison',
        'Training DICE Comparison', 'Validation DICE Comparison'
    ]
    y_labels = ['Loss', 'Loss', 'DICE Score', 'DICE Score']
    
    for i, model_name in enumerate(model_names):
        metrics = all_models_metrics[model_name]
        train_losses, train_dcs, val_losses, val_dcs = metrics
        early_stopped = all_early_stopping_info.get(model_name, False)
        
        current_epochs = list(range(1, len(train_losses) + 1))
        
        label_suffix = "*" if early_stopped else ""
        plot_label = f'{model_name.replace("_", " ").title()}{label_suffix}'
        
        axes_flat[0].plot(current_epochs, train_losses, label=plot_label, 
                         marker=markers[i % len(markers)], color=colors[i % len(colors)], alpha=0.7)
        
        axes_flat[1].plot(current_epochs, val_losses, label=plot_label, 
                         marker=markers[i % len(markers)], color=colors[i % len(colors)], alpha=0.7)
        
        axes_flat[2].plot(current_epochs, train_dcs, label=plot_label, 
                         marker=markers[i % len(markers)], color=colors[i % len(colors)], alpha=0.7)
        
        axes_flat[3].plot(current_epochs, val_dcs, label=plot_label, 
                         marker=markers[i % len(markers)], color=colors[i % len(colors)], alpha=0.7)

    for ax, title, ylabel in zip(axes_flat, plot_titles, y_labels):
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.figtext(0.02, 0.02, '* indicates early stopping', fontsize=10, style='italic')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.suptitle('Model Training Metrics Comparison', fontsize=16, y=0.98) 
    
    plot_save_path = os.path.join(save_dir, 'training_comparison_plots.png')
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    logger.log_both(f"Training comparison plot saved to: {plot_save_path}")
    plt.close()
    
    summary_log = "\n" + "="*70
    summary_log += "\nFINAL MODEL COMPARISON SUMMARY (Multi-GPU with Early Stopping)"
    summary_log += "\n" + "="*70
    
    best_overall_val_dice = -1.0
    overall_winner_name = "N/A"
    
    for model_name in model_names:
        metrics = all_models_metrics[model_name]
        train_losses, train_dcs, val_losses, val_dcs = metrics
        early_stopped = all_early_stopping_info.get(model_name, False)
        
        if val_dcs:
            best_val_dice_model = max(val_dcs)
            if best_val_dice_model > best_overall_val_dice:
                best_overall_val_dice = best_val_dice_model
                overall_winner_name = model_name.replace("_", " ").title()

            summary_log += f"\n\n{model_name.replace('_', ' ').upper()} - Training Epochs: {len(train_losses)} {'(Early Stopped)' if early_stopped else ''}"
            summary_log += f"\n{model_name.replace('_', ' ').upper()} - Best Training DICE: {max(train_dcs):.4f}"
            summary_log += f"\n{model_name.replace('_', ' ').upper()} - Best Validation DICE: {best_val_dice_model:.4f}"
            summary_log += f"\n{model_name.replace('_', ' ').upper()} - Final Training Loss: {train_losses[-1]:.4f}"
            summary_log += f"\n{model_name.replace('_', ' ').upper()} - Final Validation Loss: {val_losses[-1]:.4f}"
        else:
            summary_log += f"\n\n{model_name.replace('_', ' ').upper()} - No training data available."
            
    summary_log += f"\n\n🏆 OVERALL WINNER (based on Validation DICE): {overall_winner_name}"
    summary_log += f"\n📈 Best Validation DICE achieved: {best_overall_val_dice:.4f}"
    
    summary_log += f"\n\n📊 EARLY STOPPING SUMMARY:"
    for model_name, stopped in all_early_stopping_info.items():
        summary_log += f"\n  {model_name.replace('_', ' ').title()}: {'Triggered' if stopped else 'Not triggered'}"
    
    if len(model_names) > 1 and best_overall_val_dice > 0:
        summary_log += f"\n📊 {overall_winner_name} improvements over other models (based on best Validation DICE):"
        winner_score = best_overall_val_dice
        for name in model_names:
            if name == overall_winner_name.lower().replace(" ", "_"):
                continue
            other_score = max(all_models_metrics[name][3]) if all_models_metrics[name][3] else 0
            if other_score > 0:
                improvement = ((winner_score - other_score) / other_score) * 100
                summary_log += f"\n  vs {name.replace('_', ' ').title()}: {improvement:.2f}%"

    summary_log += "\n" + "="*70
    
    logger.log_both(summary_log)
    
    summary_file_path = os.path.join(save_dir, 'training_summary.txt')
    with open(summary_file_path, 'w') as f:
        f.write(summary_log)

def save_all_test_results(
    all_test_results: Dict[str, Tuple[float, float]], 
    test_results_path: str,
    logger
):
    """Save test results for all trained models to file"""
    
    test_summary = "="*60 + "\n"
    test_summary += "FINAL TEST SET EVALUATION RESULTS (Multi-GPU)\n"
    test_summary += "="*60 + "\n\n"
    
    best_test_dice_overall = -1.0
    overall_test_winner = "N/A"

    for model_name, results in all_test_results.items():
        loss, dc = results
        test_summary += f"{model_name.replace('_', ' ').title()} Test Results:\n"
        test_summary += f"  Test Loss: {loss:.6f}\n"
        test_summary += f"  Test DICE: {dc:.6f}\n\n"
        
        if dc > best_test_dice_overall:
            best_test_dice_overall = dc
            overall_test_winner = model_name.replace('_', ' ').title()

    test_summary += f"🏆 BEST TEST PERFORMANCE: {overall_test_winner}\n"
    test_summary += f"📈 Best Test DICE: {best_test_dice_overall:.6f}\n"
    test_summary += "="*60 + "\n"
    
    with open(test_results_path, 'w') as f:
        f.write(test_summary)
    
    logger.log_both(test_summary)
