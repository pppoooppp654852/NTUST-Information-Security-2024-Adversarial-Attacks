import os

import numpy as np
from torch import load as pt_load

def convert_2d_array_to_pe_file(array_2d: np.ndarray, original_size: int, output_file_path: str) -> None:
    # Flatten the 2D array
    flattened_array = array_2d.flatten()
    
    # Slice the flattened array to match the original size
    byte_string = flattened_array[:original_size]
    
    # Convert to bytes
    byte_string = byte_string.tobytes()
    
    # Write the byte string to a file
    with open(output_file_path, "wb") as file:
        file.write(byte_string)
        
pt_folder_path = "./injected_data"
dest_folder_path = "./injected_data-converted"
for tag_folder in os.listdir(pt_folder_path):
    now_path = os.path.join(pt_folder_path, tag_folder)
    file_list = os.listdir(now_path)
    
    for file_name in file_list:
        with open(os.path.join(now_path, file_name), "rb") as pt_file:
            target_data: dict[str, np.ndarray | int] = pt_load(pt_file)
            
            pe_file_name = file_name.removesuffix(".pt").removesuffix(".exe") + ".exe"
            pe_file_path = os.path.join(dest_folder_path, tag_folder, pe_file_name)
            
            if (type(target_data["image"]) is np.ndarray) and (type(target_data["length"]) is int):
                if not os.path.exists(os.path.join(dest_folder_path, tag_folder)):
                    os.makedirs(os.path.join(dest_folder_path, tag_folder))
                convert_2d_array_to_pe_file(target_data["image"], target_data["length"], pe_file_path)
                print(f"Converted {file_name}")
