import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path
import sys
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
from model import CustomResNet18
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device}")

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor()
])

# Get the MNIST train dataset
train_data = datasets.MNIST(root=".",
                            train=True,
                            download=True,
                            transform=transform)

# Get the MNIST test dataset
test_data = datasets.MNIST(root=".",
                           train=False,
                           download=True,
                           transform=transform)

class_names = train_data.classes

# Create train dataloader

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
early_stopping = EarlyStopping(patience=5, verbose=True, save_dir='MNIST_data/weights')
num_epochs = 2

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