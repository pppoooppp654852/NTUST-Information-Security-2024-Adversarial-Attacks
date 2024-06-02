import torch
from torchvision import transforms
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from utils import *
from model import CustomResNet18
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

target_folder = Path("data/injected_data")
# target_folder = Path("data/target_data")

classes = read_json("config/VIRUS_classes.json")

model = CustomResNet18(num_classes=len(classes))
model.load_state_dict(torch.load("data/weights/checkpoint.pt"))
model.to(device)
model.eval()

for file in target_folder.rglob('*.pt'):
    label = file.parent.name
    data = torch.load(file)
    img = Image.fromarray(data["image"])
    
    img_copy = img.copy()
    img = transform(img).unsqueeze(0).to(device)
    logits = model(img)
    probilities = torch.softmax(logits, dim=1)
    conf, pred = torch.max(probilities, 1)
    predicted_label = classes[pred.item()]
    print(f"File: {file.stem}\nLabel: {label} | Predicted: {predicted_label} | confidence: {conf.item():.4f}\n")
    
    # Display the image with the predicted label
    plt.figure(figsize=(5,5))
    plt.imshow(img_copy, cmap='gray')
    plt.title(f"File: {file.stem}\nLabel: {label} | Predicted: {predicted_label} | confidence: {conf.item():.4f}", fontsize=8)
    plt.axis('off')
    plt.show()