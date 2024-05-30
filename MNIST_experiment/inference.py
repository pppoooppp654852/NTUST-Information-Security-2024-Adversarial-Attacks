import torch
from torchvision import transforms
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import sys
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
from utils import *
from model import CustomResNet18



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# Define the transformation for ResNet-18
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor()
])

image_folder = Path("injected")
images_path = [str(img_path) for img_path in image_folder.glob("*.jpg")]


classes = read_json("config/MNIST_classes.json")

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
        plt.imshow(img_copy, cmap='gray')
        plt.title(f"Predicted: {predicted_label} | confidence: {conf.item():.4f}")
        plt.axis('off')
        plt.show()
