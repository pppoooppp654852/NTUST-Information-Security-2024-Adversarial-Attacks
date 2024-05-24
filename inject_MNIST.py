import numpy as np
import torch
from torchvision import transforms
from model.custom_resnet import CustomResNet18
from utils import *
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# Define the transformation for ResNet-18
transform = transforms.Compose([
    transforms.Resize(256),              # Resize the shorter side of the image to 256 pixels
    transforms.CenterCrop(224),          # Crop the center of the image to 224x224 pixels
    transforms.ToTensor(),               # Convert the image to a PyTorch tensor
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat the single channel 3 times to create an RGB image
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using the ImageNet mean and standard deviation
                         std=[0.229, 0.224, 0.225])
])

image_folder = Path("data/normal")
output_folder = Path("data/injected")
ensure_dir(str(output_folder))
reset_dir(str(output_folder))

images_path = [img_path for img_path in image_folder.glob("*.jpg")]
images_path = random.sample(images_path, 10)

classes = read_json("config/classes.json")

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
    img_copy = img.copy()
    img = transform(img).unsqueeze(0).to(device)

    # Make the image require gradients
    img.requires_grad = True

    # Define optimizer
    optimizer = torch.optim.Adam([img], lr=0.01)
    
    # Iterative process for modifying the image
    num_iterations = 20

    for iteration in range(num_iterations):
        # Forward pass
        logits = model(img)
        
        # Define the target label (an incorrect class)
        target_label = torch.tensor([1])
        target_label = target_label.to(device)

        # Define a loss function that encourages misclassification
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, target_label)
        
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

    # Display the modified image with the new prediction
    modified_logits = model(img)
    modified_probilities = torch.softmax(modified_logits, dim=1)
    modified_conf, modified_pred = torch.max(modified_probilities, 1)
    modified_predicted_label = classes[modified_pred.item()]

    modified_img = img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    modified_img = (modified_img - modified_img.min()) / (modified_img.max() - modified_img.min())  # Normalize for display

    # plt.imshow(modified_img)
    # plt.title(f"Modified: {modified_predicted_label} | Conf: {modified_conf.item():.4f}")
    # plt.axis('off')
    # plt.show()
