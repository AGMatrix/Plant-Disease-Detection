"""
Dataset module for plant disease detection
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


class PlantDiseaseDataset(Dataset):
    """Dataset class for plant disease images"""
    
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        label = self.data.iloc[idx]['class_id']
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label