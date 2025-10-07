"""
Configuration file for plant disease detection project
Uses DINOv2 vision transformer
Purpose: Centralized configuration makes the project modular and easy to modify. 
Instead of hardcoding values across files, I defined them once here.
"""

# Data Configuration
DATA_CONFIG = {
    'train_ratio': 0.7,      # 70% of data for training
    'val_ratio': 0.15,       # 15% for validation (hyperparameter tuning), detects overfitting
    'test_ratio': 0.15,      # 15% for final testing (never seen during training). If test accuracy << validation accuracy, indicates overfitting
    'image_size': (224, 224), # DINOv2 expects 224×224 images (standard for ViT) - DINOv2 was trained on 224×224 images
    'batch_size': 64,        # Process 32 images simultaneously (GPU memory dependent) - Number of images processed together in one forward/backward pass
    'num_workers': 4,        # DataLoader uses 4 CPU threads to load images in parallel
}

# Model Configuration - DINOv2
MODEL_CONFIG = {
    'architecture': 'dinov2',
    'model_size': 'base',  # Options: 'small' (faster), 'base' (balanced), 'large' (best accuracy)
     # 'small' (vits14): 22M parameters, 384-dim features, fastest, 90-93% accuracy
     # 'base' (vitb14): 86M parameters, 768-dim features, balanced, 92-96% accuracy
     # 'large' (vitl14): 304M parameters, 1024-dim features, slowest, 94-97% accuracy
    'freeze_backbone': True,  # Start frozen, then can unfreeze for fine-tuning

    'num_classes': None,  # Auto-set from dataset
}

# Training Configuration
TRAINING_CONFIG = {
    'num_epochs': 15,  # DINOv2 may need more epochs when backbone is frozen 
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
}

# Paths
PATHS = {
    'data_folder': '../archive', 
    'train_csv': 'train.csv',
    'val_csv': 'val.csv',
    'test_csv': 'test.csv',
    'classes_json': 'classes.json',
    'model_save': 'best_model.pth',
    'history_save': 'training_history.json',
}

# Image Normalization (ImageNet statistics - required for DINOv2)
NORMALIZATION = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}
