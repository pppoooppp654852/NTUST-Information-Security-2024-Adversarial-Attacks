import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((288, 288), transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Create dataset
train_dataset = datasets.ImageFolder(root="data/dataset_splits/train", transform=transform)
val_dataset = datasets.ImageFolder(root="data/dataset_splits/test", transform=transform)
class_names = train_dataset.classes

# Create dataloader
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=64,
                              shuffle=True)

val_dataloader = DataLoader(dataset=val_dataset,
                             batch_size=64,
                             shuffle=False)

# Create model
model = CustomResNet18(num_classes=len(class_names))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00)
early_stopping = EarlyStopping(patience=10, verbose=True, save_dir='data/weights')
num_epochs = 30

writer = SummaryWriter()

for epoch in range(num_epochs):
    print(f"Epoch: {epoch+1}")
    train_loss, train_acc, train_preds, train_labels = train_step(
        model=model,
        dataloader=train_dataloader,
        loss_fn=criterion,
        optimizer=optimizer,
        device=device,
    )
    
    val_loss, val_acc, val_preds, val_labels = test_step(
        model=model,
        dataloader=val_dataloader,
        loss_fn=criterion,
        device=device
    )
        
    # Print out what's happening
    print(
        f"train_loss: {train_loss:.4f} | "
        f"train_acc: {train_acc:.4f} | "
        f"val_loss: {val_loss:.4f} | "
        f"val_acc: {val_acc:.4f} | "
    )
    
    # Update the metrics
    writer.add_scalar('Loss/train', float(train_loss), epoch)
    writer.add_scalar('Accuracy/train', float(train_acc), epoch)
    writer.add_scalar('Loss/val', float(val_loss), epoch)
    writer.add_scalar('Accuracy/val', float(val_acc), epoch)

    
    early_stopping(val_loss, model)
    if early_stopping.early_stop == True:
        break

writer.flush()
writer.close()