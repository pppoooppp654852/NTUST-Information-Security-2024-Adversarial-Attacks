from typing import Dict, List, Tuple
import torch
from tqdm.auto import tqdm
from pathlib import Path

def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()
    train_loss, train_acc = 0, 0
    all_preds, all_labels = [], []
    
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")
    
    for batch, (X, y) in progress_bar:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        all_preds.extend(y_pred_class.detach().cpu().numpy())
        all_labels.extend(y.detach().cpu().numpy())
        
        # Update progress bar description
        progress_bar.set_description(f"Training - Loss: {train_loss/(batch+1):.4f}, Acc: {train_acc/(batch+1):.4f}")

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc, torch.tensor(all_preds), torch.tensor(all_labels)

def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
):
    model.eval()
    test_loss, test_acc = 0, 0
    all_preds, all_labels = [], []
    
    with torch.inference_mode():
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing")
        
        for batch, (X, y) in progress_bar:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            y_pred_class = y_pred.argmax(dim=1)
            test_acc += (y_pred_class == y).sum().item() / len(y_pred_class)

            all_preds.extend(y_pred_class.detach().cpu().numpy())
            all_labels.extend(y.detach().cpu().numpy())
            
            # Update progress bar description
            progress_bar.set_description(f"Testing - Loss: {test_loss/(batch+1):.4f}, Acc: {test_acc/(batch+1):.4f}")

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc, torch.tensor(all_preds), torch.tensor(all_labels)

class EarlyStopping:
    def __init__(self, patience, verbose=False, delta=0, save_dir=None, save=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        if not Path(save_dir).exists():
            Path(save_dir).mkdir()
        self.save_dir = save_dir
        self.save = save
        

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if self.save:
                self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save:
                self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        checkpoint_path = Path(self.save_dir) / 'checkpoint.pt'
        torch.save(model.state_dict(), str(checkpoint_path))
        self.val_loss_min = val_loss