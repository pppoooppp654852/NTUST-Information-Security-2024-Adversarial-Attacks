from pathlib import Path
import torch
import sys
root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))
from utils import ensure_dir, convert_2d_array_to_pe_file

injected_dir = Path('data/injected_data')


for file in injected_dir.rglob('*.pt'):
    if "Normal_NirSoft" == file.parent.name:
        print(file.parent.name)
        data = torch.load(file)
        exe_file = convert_2d_array_to_pe_file(data["image"], data['length'], f'{file.stem}.exe')
        break