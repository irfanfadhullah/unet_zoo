# utils/metrics.py
import torch
import numpy as np
import os
from PIL import Image

def dice_coefficient(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-7, threshold: float = 0.5) -> torch.Tensor:
    """
    Improved Dice Coefficient calculation.
    """
    prediction = torch.sigmoid(prediction)
    prediction = (prediction > threshold).float()
    
    prediction = prediction.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (prediction * target).sum()
    union = prediction.sum() + target.sum()
    
    if union == 0:
        return torch.tensor(1.0, device=prediction.device)
    
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice

def check_dataset_integrity(dataset_path: str, logger):
    """
    Function to check dataset integrity and mask values
    """
    logger.log_both("Checking dataset integrity...")
    for split in ['train', 'test', 'valid']:
        masks_path = os.path.join(dataset_path, split, 'masks')
        if os.path.exists(masks_path):
            mask_files = [f for f in os.listdir(masks_path) if f.endswith(('.png', '.jpg', '.jpeg'))][:3]
            
            for mask_file in mask_files:
                mask = Image.open(os.path.join(masks_path, mask_file)).convert('L')
                mask_array = np.array(mask)
                unique_values = np.unique(mask_array)
                logger.log_both(f"{split}/{mask_file}: unique values = {unique_values}, shape = {mask_array.shape}")