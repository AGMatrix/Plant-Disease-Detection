"""
Model architecture for plant disease detection
Uses DINOv2 vision transformer from Meta AI
"""

import torch
import torch.nn as nn


def create_model(num_classes, model_size='base', freeze_backbone=False):
    """
    Create DINOv2-based model for plant disease classification
    
    Args:
        num_classes (int): Number of disease classes
        model_size (str): 'small' (384-dim), 'base' (768-dim), 'large' (1024-dim)
        freeze_backbone (bool): Freeze DINOv2 weights (recommended to start)
        
    Returns:
        DINOv2Classifier: Complete model
    """
    
    print(f"Loading DINOv2-{model_size}...")
    
    # Load DINOv2 from torch hub
    if model_size == 'small':
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        feature_dim = 384
    elif model_size == 'base':
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        feature_dim = 768
    elif model_size == 'large':
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        feature_dim = 1024
    else:
        raise ValueError(f"Invalid model_size: {model_size}")
    
    # Freeze backbone if specified
    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
        print("DINOv2 backbone frozen (only training classification head)")
    else:
        print("DINOv2 backbone trainable (fine-tuning entire model)")
    
    # Create classifier
    model = DINOv2Classifier(backbone, feature_dim, num_classes)
    
    return model


class DINOv2Classifier(nn.Module):
    """DINOv2 with classification head"""
    
    def __init__(self, backbone, feature_dim, num_classes, dropout=0.3):
        super().__init__()
        self.backbone = backbone
        
        # Simple but effective classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_classes)
        )
    
    def forward(self, x):
        # Get DINOv2 features
        features = self.backbone(x)
        
        # Classify
        logits = self.classifier(features)
        
        return logits


def get_model_info(model):
    """Print model statistics"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': total_params - trainable_params
    }
