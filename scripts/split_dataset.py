import os
import shutil
import random
from pathlib import Path

def split_dataset(dataset_dir, output_dir, train_ratio=0.8, seed=42):
    random.seed(seed)
    
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    train_dir = output_dir / 'train'
    test_dir = output_dir / 'test'
    
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all class directories
    classes = [d.name for d in dataset_dir.iterdir() if d.is_dir()]
    
    for class_name in classes:
        class_dir = dataset_dir / class_name
        images = list(class_dir.glob('*'))
        
        # Shuffle images
        random.shuffle(images)
        
        # Split images
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # Create class directories in train and test folders
        train_class_dir = train_dir / class_name
        test_class_dir = test_dir / class_name
        
        train_class_dir.mkdir(parents=True, exist_ok=True)
        test_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images to respective directories
        for img in train_images:
            shutil.copy(img, train_class_dir / img.name)
        
        for img in test_images:
            shutil.copy(img, test_class_dir / img.name)
    
    print(f'Dataset split completed. Train set: {train_dir}, Test set: {test_dir}')

# Usage
dataset_dir = 'data/dataset'
output_dir = 'data/dataset_splits'
split_dataset(dataset_dir, output_dir)
