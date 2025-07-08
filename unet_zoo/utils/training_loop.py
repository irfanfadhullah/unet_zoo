import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Tuple, List, Union, Dict

from unet_zoo.config import Config 

from .logger import Logger
from .metrics import dice_coefficient 
from .early_stopping import EarlyStopping
from .lr_scheduler import DiceScheduler
from .multi_gpu import MultiGPUManager 
import os

from unet_zoo.models import UNet, VNet, U2NET, U2NETP, SwinTransformerSys, ResUnet
from unet_zoo.models import WRANet, EGEUNet, UNext, UNext_S, MMUNet, ResAxialAttentionUNet, medt_net


U2NET_LOSS_WEIGHTS = {
    'main': 1.0, 
    'side1': 1.0,
    'side2': 1.0,
    'side3': 1.0,
    'side4': 1.0,
    'side5': 1.0,
    'side6': 1.0,
}

EGEUNET_DS_LOSS_WEIGHTS = {
    'out': 1.0, 
    'side1': 0.5, 
    'side2': 0.5,
    'side3': 0.5,
    'side4': 0.5,
    'side5': 0.5, 
}


def _process_model_outputs_for_loss_and_metrics(
    outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
    masks: torch.Tensor,
    criterion: nn.Module,
    model_type: type, 
    logger: Logger,
    is_training: bool 
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
    """
    Calculates the total loss and main output's Dice score based on model type.
    This function expects model outputs to be LOGITS (raw scores), not probabilities.

    Returns: total_loss, main_prediction_logits, dice_score
    """
    total_loss = torch.tensor(0.0, device=masks.device) 
    main_prediction_logits = None 

    if issubclass(model_type, (U2NET, U2NETP)):
        for key, output_tensor in outputs.items():
            resized_mask = F.interpolate(masks, size=output_tensor.shape[2:], mode='bilinear', align_corners=False)
            total_loss += U2NET_LOSS_WEIGHTS.get(key, 0.5) * criterion(output_tensor, resized_mask)
        main_prediction_logits = outputs['main'] 
    elif issubclass(model_type, EGEUNet):
        if isinstance(outputs, dict): 
            for key, output_tensor in outputs.items():
                resized_mask = F.interpolate(masks, size=output_tensor.shape[2:], mode='bilinear', align_corners=False)
                total_loss += EGEUNET_DS_LOSS_WEIGHTS.get(key, 0.5) * criterion(output_tensor, resized_mask)
            main_prediction_logits = outputs['out'] 
        else: 
            total_loss = criterion(outputs, masks)
            main_prediction_logits = outputs
    elif issubclass(model_type, MMUNet):
        total_loss = criterion(outputs['out'], masks)
        main_prediction_logits = outputs['out']
    else:
        total_loss = criterion(outputs, masks)
        main_prediction_logits = outputs

    dice = dice_coefficient(main_prediction_logits, masks)
    
    return total_loss, main_prediction_logits, dice


def train_one_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    optimizer: optim.Optimizer, 
    criterion: nn.Module, 
    device: torch.device, 
    writer: SummaryWriter, 
    epoch: int,
    model_name: str,
    logger: Logger
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_dc = 0.0
    
    max_grad_norm = 1.0
    
    from unet_zoo.models import UNet, VNet, U2NET, U2NETP, SwinTransformerSys, ResUnet
    from unet_zoo.models import WRANet, EGEUNet, UNext, UNext_S, MMUNet, ResAxialAttentionUNet, medt_net

    unwrapped_model_type = type(model.module) if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else type(model)

    for idx, (img, mask, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1} Train ({model_name})", leave=False)):
        img = img.float().to(device)
        mask = mask.float().to(device) 

        optimizer.zero_grad()
        outputs = model(img) 
        
        loss, main_pred_logits, dc = _process_model_outputs_for_loss_and_metrics(
            outputs, mask, criterion, unwrapped_model_type, logger, is_training=True
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        running_loss += loss.item()
        running_dc += dc.item()
        
        if idx % 50 == 0:
            with torch.no_grad():
                pred_sigmoid = torch.sigmoid(main_pred_logits)
                pred_mean = pred_sigmoid.mean().item()
                pred_max = pred_sigmoid.max().item()
                mask_mean = mask.mean().item() 
                
                log_msg = f"{model_name} - Batch {idx}: Loss={loss.item():.4f}, Dice={dc.item():.4f}"
                logger.log_file_only(log_msg)
                log_msg = f"  Pred stats (sigmoid): mean={pred_mean:.4f}, max={pred_max:.4f}, mask_mean={mask_mean:.4f}"
                logger.log_file_only(log_msg)

        if idx % 100 == 0:
            global_step = epoch * len(dataloader) + idx
            writer.add_scalar(f'Batch/{model_name}_Train_Loss', loss.item(), global_step)
            writer.add_scalar(f'Batch/{model_name}_Train_Dice', dc.item(), global_step)

    avg_loss = running_loss / len(dataloader)
    avg_dc = running_dc / len(dataloader)
    return avg_loss, avg_dc

def validate_one_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device,
    model_name: str,
    logger: Logger
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_dc = 0.0
    
    from unet_zoo.models import UNet, VNet, U2NET, U2NETP, SwinTransformerSys, ResUnet
    from unet_zoo.models import WRANet, EGEUNet, UNext, UNext_S, MMUNet, ResAxialAttentionUNet, medt_net

    unwrapped_model_type = type(model.module) if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else type(model)

    with torch.no_grad():
        for idx, (img, mask, _) in enumerate(tqdm(dataloader, desc=f"Validation ({model_name})", leave=False)):
            img = img.float().to(device)
            mask = mask.float().to(device)

            outputs = model(img) 

            loss, _, dc = _process_model_outputs_for_loss_and_metrics(
                outputs, mask, criterion, unwrapped_model_type, logger, is_training=False 
            )
            
            running_loss += loss.item()
            running_dc += dc.item()

    avg_loss = running_loss / len(dataloader)
    avg_dc = running_dc / len(dataloader)
    return avg_loss, avg_dc

def train_model(
    model: nn.Module, 
    train_dataloader: DataLoader, 
    val_dataloader: DataLoader, 
    optimizer: optim.Optimizer, 
    criterion: nn.Module, 
    config: Config, 
    model_name: str,
    best_checkpoint_path: str,
    last_checkpoint_path: str,
    logger: Logger,
    gpu_manager: MultiGPUManager
) -> Tuple[List[float], List[float], List[float], List[float], bool]:
    from unet_zoo.models import VNet 
    unwrapped_model = model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else type(model)
    
    if isinstance(unwrapped_model, VNet):
        logger.log_both(f"ERROR: {model_name} is a 3D model ({type(unwrapped_model).__name__}), but the training script and BoneDataset are configured for 2D. Training will not proceed for this model. Please remove it from your training list or adapt dataset/pipeline.")
        raise ValueError("Dimensionality mismatch: VNet is 3D, training script/dataset is 2D. Please ensure your dataset provides 5D data (NCDHW) and adjust the loss function (e.g., CrossEntropyLoss) and target format accordingly, or remove VNet from the training list for this setup.")


    tensorboard_log_dir = os.path.join(config.TENSORBOARD_BASE_DIR, model_name.replace(" ", "_").lower())
    writer = SummaryWriter(tensorboard_log_dir)
    logger.log_both(f"TensorBoard logs for {model_name} will be saved to: {tensorboard_log_dir}")

    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE, 
        min_delta=0.0, 
        restore_best_weights=True,
        verbose=True,
        mode='max' 
    )
    
    dice_scheduler = DiceScheduler(
        optimizer, 
        patience=config.LR_SCHEDULER_PATIENCE, 
        factor=config.LR_SCHEDULER_FACTOR, 
        min_lr=config.MIN_LR,
        min_delta=0.0, 
        verbose=True,
        mode='max' 
    )

    train_losses = []
    train_dcs = []
    val_losses = []
    val_dcs = []

    early_stopped = False

    logger.log_both(f"\nStarting training for {model_name} - {config.EPOCHS} epochs on {config.get_device_info()}")
    logger.log_both(f"  Early Stopping: patience={config.EARLY_STOPPING_PATIENCE}, mode='max'")
    logger.log_both(f"  LR Scheduler: patience={config.LR_SCHEDULER_PATIENCE}, factor={config.LR_SCHEDULER_FACTOR}, min_lr={config.MIN_LR}, mode='max'")
    
    for epoch in range(config.EPOCHS):
        train_loss, train_dc = train_one_epoch(
            model, train_dataloader, optimizer, criterion, config.DEVICE, writer, epoch, model_name, logger
        )
        train_losses.append(train_loss)
        train_dcs.append(train_dc)

        val_loss, val_dc = validate_one_epoch(
            model, val_dataloader, criterion, config.DEVICE, model_name, logger
        )
        val_losses.append(val_loss)
        val_dcs.append(val_dc)

        dice_scheduler.step(val_dc, epoch + 1)
        early_stopping(val_dc, model, epoch + 1)
        
        gpu_manager.save_model_state(model, last_checkpoint_path)

        writer.add_scalar(f'Epoch/{model_name}_Train_Loss', train_loss, epoch + 1)
        writer.add_scalar(f'Epoch/{model_name}_Train_Dice', train_dc, epoch + 1)
        writer.add_scalar(f'Epoch/{model_name}_Val_Loss', val_loss, epoch + 1)
        writer.add_scalar(f'Epoch/{model_name}_Val_Dice', val_dc, epoch + 1)
        
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar(f'{model_name}_Learning_Rate', current_lr, epoch + 1)

        epoch_log = "-" * 60
        epoch_log += f"\n{model_name} - Epoch {epoch + 1}/{config.EPOCHS}"
        epoch_log += f"\n  Train Loss: {train_loss:.6f} | Train DICE: {train_dc:.6f}"
        epoch_log += f"\n  Val Loss:   {val_loss:.6f} | Val DICE:   {val_dc:.6f}"
        epoch_log += f"\n  Learning Rate: {current_lr:.8f}"
        epoch_log += f"\n  Best Val Dice: {early_stopping.get_best_score():.6f}"
        epoch_log += f"\n{'-' * 60}"
        logger.log_both(epoch_log)

        if early_stopping.early_stop:
            logger.log_both(f"\n🛑 Early stopping triggered for {model_name} at epoch {epoch + 1}")
            logger.log_both(f"Best validation dice: {early_stopping.get_best_score():.6f}")
            logger.log_both(f"Stopped after {early_stopping.stopped_epoch} epochs")
            early_stopped = True
            break

    writer.close()
    
    final_message = f"Training {'stopped early' if early_stopped else 'completed'} for {model_name}"
    final_message += f" after {epoch + 1 if early_stopped else config.EPOCHS} epochs"
    logger.log_both(final_message)
    logger.log_both(f"Best validation Dice coefficient for {model_name}: {early_stopping.get_best_score():.6f}")

    return train_losses, train_dcs, val_losses, val_dcs, early_stopped

def evaluate_model(
    model: nn.Module, 
    test_dataloader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device,
    model_name: str,
    logger: Logger
) -> Tuple[float, float]:
    model.eval()
    test_running_loss = 0.0
    test_running_dc = 0.0

    from unet_zoo.models import UNet, VNet, U2NET, U2NETP, SwinTransformerSys, ResUnet
    from unet_zoo.models import WRANet, EGEUNet, UNext, UNext_S, MMUNet, ResAxialAttentionUNet, medt_net

    unwrapped_model_type = type(model.module) if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else type(model)

    logger.log_both(f"\nEvaluating {model_name} on test set...")
    with torch.no_grad():
        for idx, (img, mask, _) in enumerate(tqdm(test_dataloader, desc=f"{model_name} test evaluation", leave=True)):
            img = img.float().to(device)
            mask = mask.float().to(device)

            outputs = model(img) 

            loss, _, dc = _process_model_outputs_for_loss_and_metrics(
                outputs, mask, criterion, unwrapped_model_type, logger, is_training=False 
            )

            test_running_loss += loss.item()
            test_running_dc += dc.item()

    test_loss = test_running_loss / len(test_dataloader)
    test_dc = test_running_dc / len(test_dataloader)
    
    logger.log_both(f"{model_name} - Final Test Loss: {test_loss:.4f}")
    logger.log_both(f"{model_name} - Final Test DICE: {test_dc:.4f}")
    
    return test_loss, test_dc