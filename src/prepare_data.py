"""
Data preparation script for plant disease detection
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import json


def prepare_dataset(data_folder, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Organize dataset into train/val/test splits"""
    data_root = Path(data_folder)
    
    images = []
    labels = []
    class_names = []
    
    for class_folder in data_root.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            
            if class_name not in class_names:
                class_names.append(class_name)
            
            class_id = class_names.index(class_name)
            
            for img_file in class_folder.glob('*.*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    images.append(str(img_file))
                    labels.append(class_id)
    
    df = pd.DataFrame({'image_path': images, 'class_id': labels})
    
    train_val_df, test_df = train_test_split(
        df, test_size=test_ratio, stratify=df['class_id'], random_state=42
    )
    
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size_adjusted, 
        stratify=train_val_df['class_id'], random_state=42
    )
    
    train_df.to_csv('train.csv', index=False)
    val_df.to_csv('val.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    
    with open('classes.json', 'w') as f:
        json.dump(class_names, f, indent=2)
    
    print("=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"Classes: {len(class_names)}")
    print("=" * 60)
    
    return class_names


if __name__ == "__main__":
    DATA_FOLDER = '../archive'
    class_names = prepare_dataset(DATA_FOLDER)