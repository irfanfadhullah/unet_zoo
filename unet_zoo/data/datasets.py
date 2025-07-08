import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import Tuple

class BoneDataset(Dataset):
    """
    Bone Dataset with improved preprocessing
    """
    def __init__(self, root_path: str, split: str = 'train', limit: int = None):
        self.root_path = root_path
        self.split = split
        self.limit = limit
        
        images_path = os.path.join(root_path, split, "images")
        masks_path = os.path.join(root_path, split, "masks")
        
        if not os.path.exists(images_path):
            raise FileNotFoundError(f"Image directory not found: {images_path}")
        if not os.path.exists(masks_path):
            raise FileNotFoundError(f"Mask directory not found: {masks_path}")

        valid_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
        image_files = sorted([f for f in os.listdir(images_path) 
                              if not f.startswith('.') and f.lower().endswith(valid_extensions)])
        mask_files = sorted([f for f in os.listdir(masks_path) 
                             if not f.startswith('.') and f.lower().endswith(valid_extensions)])
        
        self.images = [os.path.join(images_path, i) for i in image_files][:self.limit]
        self.masks = [os.path.join(masks_path, i) for i in mask_files][:self.limit]

        if len(self.images) != len(self.masks):
            print(f"Warning: Number of images ({len(self.images)}) doesn't match number of masks ({len(self.masks)}) for split '{split}'.")


        self.image_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")

        img_tensor = self.image_transform(img)
        mask_tensor = self.mask_transform(mask)
        

        mask_tensor = (mask_tensor > 0.5).float()
        
        return img_tensor, mask_tensor, self.images[index]

    def __len__(self) -> int:
        return len(self.images)