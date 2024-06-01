import json
import torch
import torch.utils
from torch.utils.data import random_split
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import numpy as np
import random

import torch.utils.tensorboard

def reset_dir(dirname):
    dirname = Path(dirname)
    for file in dirname.glob('*'):
        if file.is_file():
            file.unlink()

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def convert_2d_array_to_pe_file(array_2d: np.ndarray, original_size: int, output_file_path: str):
    # Flatten the 2D array
    flattened_array = array_2d.flatten()
    
    # Slice the flattened array to match the original size
    byte_string = flattened_array[:original_size]
    
    # Convert to bytes
    byte_string = byte_string.tobytes()
    
    # Write the byte string to a file
    with open(output_file_path, 'wb') as file:
        file.write(byte_string)
        
def split_dataset(dataset, train_ratio=0.8, seed=None):
    if seed is not None:
        set_seed(seed)
    
    # Calculate the sizes of the train and validation datasets
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    
    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    return train_dataset, val_dataset