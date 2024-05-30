import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import random
from pathlib import Path
import sys
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
from model import CustomResNet18
from utils import *

INJECT_LOSS_THRESHOLD = 0.1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device}")

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor()
])

image_folder = Path("data/normal")
output_folder = Path("injected")
ensure_dir(str(output_folder))
reset_dir(str(output_folder))

images_path = [img_path for img_path in image_folder.glob("*.jpg")]
images_path = random.sample(images_path, 10)

classes = read_json("config/MNIST_classes.json")

model = CustomResNet18(num_classes=len(classes))
model.load_state_dict(torch.load("data/weights/checkpoint.pt"))
model.to(device)
model.eval()

# Function to save the modified image
def save_image(tensor, path):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)  # remove the batch dimension
    image = unloader(image)
    image.save(path)

for img_path in images_path:
    img = Image.open(str(img_path))
    img = transform(img).unsqueeze(0).to(device)

    # Make the image require gradients
    img.requires_grad = True

    # Define optimizer
    optimizer = torch.optim.Adam([img], lr=0.01)

    loss_val = float('inf')
    while loss_val > INJECT_LOSS_THRESHOLD:
        # Forward pass
        logits = model(img)
        
        # Define the target label (an incorrect class)
        target_label = torch.tensor([1])
        target_label = target_label.to(device)

        # Define a loss function that encourages misclassification
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, target_label)
        loss_val = loss.item()
        
        # Calculate gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Ensure the tensor values remain within valid pixel range [0, 1]
        img.data = torch.clamp(img.data, 0, 1)
        
        # Zero out the gradients for the next iteration
        img.grad.zero_()
        
    # Save the modified image
    modified_img_path = output_folder / img_path.name
    save_image(img, modified_img_path)
    print(f'Saved {modified_img_path}')