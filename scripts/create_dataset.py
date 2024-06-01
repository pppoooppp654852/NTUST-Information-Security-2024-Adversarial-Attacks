import torch
from PIL import Image
import numpy as np
from pathlib import Path
from utils import *

source_dir = Path('data/converted')
output_dir = Path('data/training_dataset')
mask_dir = Path('data/mask')
for path in source_dir.rglob('*.pt'):
    data = torch.load(path)
    print(path, data['length'])
    ensure_dir(output_dir.joinpath(path.parent.name))
    ensure_dir(mask_dir.joinpath(path.parent.name))
    image = Image.fromarray(data['image'])
    image.save(output_dir.joinpath(path.parent.name).joinpath(path.stem + '.png'))

    mask = Image.fromarray(data['mask'] * 255)
    mask.save(mask_dir.joinpath(path.parent.name).joinpath(path.stem + '_mask.png'))

