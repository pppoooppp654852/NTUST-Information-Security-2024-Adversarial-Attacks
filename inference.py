import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision import transforms
from model.custom_resnet import CustomResNet18
from utils import *
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# Define the transformation for ResNet-18
transform = transforms.ToTensor()

image_folder = Path("data/injected")
images_path = [str(img_path) for img_path in image_folder.glob("*.jpg")]


classes = read_json("config/classes.json")

model = CustomResNet18(num_classes=len(classes))
model.load_state_dict(torch.load("data/weights/checkpoint.pt"))
model.to(device)
model.eval()

with torch.inference_mode():

    for img_path in images_path:
        img = Image.open(img_path)
        img_copy = img.copy()
        img = transform(img).unsqueeze(0).to(device)
        logits = model(img)
        probilities = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probilities, 1)
        print(pred)
        predicted_label = classes[pred.item()]
        
        # Display the image with the predicted label
        plt.imshow(img_copy)
        plt.title(f"Predicted: {predicted_label} | confidence: {conf.item():.4f}")
        plt.axis('off')
        plt.show()
