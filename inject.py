import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import random
from pathlib import Path
from model import CustomResNet18
from utils import *
import warnings

warnings.filterwarnings("ignore")

INJECT_LOSS_THRESHOLD = 0.1
INJECT_EARLY_STOP_THRESHOLD = 0.0001

set_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

inverse_transform = transforms.Compose([
    transforms.Normalize(mean=[-1], std=[2]),  # Inverse of normalization [0.5], [0.5]
    transforms.ToPILImage()
])

target_folder = Path("data/target_data")
output_folder = Path("data/injected_data")
ensure_dir(str(output_folder))
reset_dir(str(output_folder))

classes = read_json("config/VIRUS_classes.json")

model = CustomResNet18(num_classes=len(classes))
model.load_state_dict(torch.load("data/weights/checkpoint.pt"))
model.to(device)
model.eval()

# Function to save the modified ndarray
def save_file(tensor, original_img, mask, length, path):
    
    
    np_array = tensor.cpu().clone()
    np_array = np_array.squeeze(0)  # remove the batch dimension
    np_array = inverse_transform(np_array)
    np_array = np.array(np_array)
    
    # Apply mask to only change values in the masked area
    original_np = np.array(original_img)
    np_array = np.where(mask == 1, np_array, original_np)
    
    result_data = {"image": np_array, "length": length}
    torch.save(result_data, path)


for file in target_folder.rglob('*.pt'):
    label = file.parent.name
    data = torch.load(file)
    img = Image.fromarray(data["image"]).copy()
    # Add a batch dimension to the image 
    img = transform(img).unsqueeze(0).to(device)
    
    # Make the image require gradients
    img.requires_grad = True
    
    # create mask
    mask = Image.fromarray(data["mask"]).copy()
    mask = np.array(mask)
    mask = torch.from_numpy(mask)
    mask = mask.unsqueeze(0).to(device)

    # Define optimizer
    optimizer = torch.optim.Adam([img], lr=0.001)

    loss_val = None
    diff_losses = []
    while True:
        # Forward pass
        logits = model(img)
        
        # Define the target label (Normal software class label)
        target_label = torch.tensor([0])
        target_label = target_label.to(device)

        # Define a loss function that encourages misclassification
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, target_label)
        
        # Early stopping if the loss is not decreasing
        if loss_val:
            diff_losses.append(loss_val - loss.item())
            if len(diff_losses) > 20:
                diff_losses.pop(0)
            # print(loss_val, sum(diff_losses) / len(diff_losses))
            if (sum(diff_losses) / len(diff_losses)) < INJECT_EARLY_STOP_THRESHOLD or loss_val < INJECT_LOSS_THRESHOLD:
                break
        loss_val = loss.item()
        
        # Calculate gradients
        optimizer.zero_grad()
        loss.backward()
        
        # Apply mask to gradients
        img.grad.data *= mask
        
        # update the image
        optimizer.step()

        img.data = torch.clamp(img.data, 0, 1)
        img.grad.zero_()
        
    
    # Save the modified image
    modified_img_path = output_folder / label / file.name
    modified_img_path.parent.mkdir(parents=True, exist_ok=True)
    original_img = Image.fromarray(data["image"])
    save_file(img, original_img, data["mask"], data["length"], modified_img_path)
    print(f'Saved {modified_img_path}')
    
    # Check the prediction of the modified tensor
    print(f'file: {file.name} | Label: {label}')
    print("Modified tensor")
    logits = model(img)
    probilities = torch.softmax(logits, dim=1)
    conf, pred = torch.max(probilities, 1)
    predicted_label = classes[pred.item()]
    print(f'Predicted: {predicted_label} | confidence: {conf.item()} | modified_loss: {loss_val}')
    
    # Load the modified data and check the prediction
    print("Loaded tensor")
    modified_data = torch.load(modified_img_path)
    injected_img = Image.fromarray(modified_data['image'])
    injected_img = transform(injected_img).unsqueeze(0).to(device)
    logits = model(injected_img)
    probilities = torch.softmax(logits, dim=1)
    conf, pred = torch.max(probilities, 1)
    predicted_label = classes[pred.item()]
    print(f"Predicted: {predicted_label} | confidence: {conf.item():.4f}\n")
