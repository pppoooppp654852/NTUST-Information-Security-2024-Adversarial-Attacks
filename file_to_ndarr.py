from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math

from torch import save as pt_save
import pefile

def get_file_names_in_folder(folder_path):
    file_names = []
    # Iterate over all files in the folder
    for file_path in folder_path.iterdir():
        # Check if the path is a file (not a folder)
        if file_path.is_file():
            file_names.append(file_path.name)
    return file_names

# Specify the folder path
folder_path = Path("D:/test_slack")

# Get the file names in the folder
file_names = get_file_names_in_folder(folder_path)
len(file_names)

def file_to_byte_string(file_path):
    byte_string = b""

    try:
        with open(file_path, "rb") as file:
            byte_string = file.read()

    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print("An error occurred:", str(e))

    return byte_string

def slack_area_to_1d_arr(pe_file: pefile.PE, file_len: int) -> np.ndarray:
    slack_regions = []

    for section in pe_file.sections:
        raw_size = section.SizeOfRawData
        virtual_size = section.Misc_VirtualSize

        if raw_size > virtual_size:
            slack_start = section.PointerToRawData + virtual_size
            slack_end = section.PointerToRawData + raw_size
            slack_regions.append((slack_start, slack_end))

    result = np.zeros(file_len, dtype=np.uint8)
    for i, (start, end) in enumerate(slack_regions):
        result[start:end] = 1
        print(f'Slack Region {i + 1}: Start Offset: {start}, End Offset: {end}, Size: {end - start} bytes')
    
    return result

def byte_string_to_2d_array(byte_string, width):
    num_elements = len(byte_string)
    num_columns = width
    num_rows = -(-num_elements // num_columns)  # Ceiling division to ensure all elements fit
    
    # Convert byte string to numpy array
    array_1d = np.frombuffer(byte_string, dtype=np.uint8)
    
    # Calculate the number of elements to pad
    num_to_pad = num_rows * num_columns - len(array_1d)

    # Pad the array with zeros
    padded_array = np.pad(array_1d, (0, num_to_pad), mode='constant')

    # Reshape the padded array into a 2D array
    array_2d = padded_array.reshape((num_rows, num_columns))
    
    return array_2d

def get_img_width(size):
    img_width = math.ceil((size * 1000) ** 0.5)
    return img_width

for file in file_names:
    try : 
        file_path = folder_path / file
        byte_string = file_to_byte_string(file_path)
        file_size = len(byte_string)
        file_size_kB = file_size / 1024
        
        img_width = get_img_width(file_size_kB)
        array_2d = byte_string_to_2d_array(byte_string, img_width)
        
        pe_file = pefile.PE(file_path)
        slack_arr = slack_area_to_1d_arr(pe_file, file_size)
        slack_2d = byte_string_to_2d_array(slack_arr, img_width)
        
        img_height = -(-file_size // img_width)  # Ceiling division to ensure all elements fit
        result_data: dict[str, np.ndarray | int] \
                    = {"image": array_2d, "mask": slack_2d, "length": len(byte_string)}
        
        # Save the image as a .pt file
        save_path = f"converted/{file}.pt"
        pt_save(result_data, save_path)
        print(f"{file}.pt saved...")
    except Exception as e:
        print(f"\n!> Error: {e}")
        print(f"file name: {file}, size: {file_size_kB:.1f}kB")
        print(f"image shape: {array_2d.shape}\n")
