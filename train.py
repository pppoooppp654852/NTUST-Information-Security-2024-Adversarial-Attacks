import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision import transforms
from model.custom_resnet import CustomResNet18
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# Define the transformation for ResNet-18
transform = transforms.Compose([
    transforms.Resize(256),              # Resize the shorter side of the image to 256 pixels
    transforms.CenterCrop(224),          # Crop the center of the image to 224x224 pixels
    transforms.ToTensor(),               # Convert the image to a PyTorch tensor
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat the single channel 3 times to create an RGB image
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using the ImageNet mean and standard deviation
                         std=[0.229, 0.224, 0.225])
])

# Get the MNIST train dataset 
train_data = datasets.MNIST(root=".",
                            train=True,
                            download=True,
                            transform=transform) # do we want to transform the data as we download it? 

# Get the MNIST test dataset
test_data = datasets.MNIST(root=".",
                           train=False,
                           download=True,
                           transform=transform) 

class_names = train_data.classes
write_json(class_names, "classes.json")

# Create train dataloader
from torch.utils.data import DataLoader

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=256,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=256,
                             shuffle=False)

model = CustomResNet18(num_classes=len(class_names))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00)
early_stopping = EarlyStopping(patience=5, verbose=True, save_dir='weights/')
num_epochs = 10

for epoch in range(num_epochs):
    train_loss, train_acc, train_preds, train_labels = train_step(
        model=model,
        dataloader=train_dataloader,
        loss_fn=criterion,
        optimizer=optimizer,
        device=device,
    )
    
    val_loss, val_acc, val_preds, val_labels = test_step(
        model=model,
        dataloader=test_dataloader,
        loss_fn=criterion,
        device=device
    )
    
    early_stopping(val_loss, model)
        
    # Print out what's happening
    print(
        f"Epoch: {epoch+1} | "
        f"train_loss: {train_loss:.4f} | "
        f"train_acc: {train_acc:.4f} | "
        f"test_loss: {val_loss:.4f} | "
        f"test_acc: {val_acc:.4f} | "
    )