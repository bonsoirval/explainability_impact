import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

def get_lightweight_model(model_name: str, num_classes: int, pretrained: bool = True):
    """
    Instantiates a lightweight model with adjusted classifier head.
    """
    model_name = model_name.lower()
    
    if model_name == "mobilenet_v3":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        
    elif model_name == "shufflenet_v2":
        weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        model = models.shufflenet_v2_x1_0(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")
        
    return model

def train_model(model, train_loader, val_loader, num_epochs=3, lr=0.001, device="cpu", save_dir="phase_4_modeling/models", model_name="model"):
    """
    Standard training/validation loop.

    Returns
    -------
    model         : Trained model (best checkpoint loaded).
    save_path     : Path to the saved best checkpoint.
    history       : Dict with keys train_loss, train_acc, val_loss, val_acc
                    each containing a list of per-epoch float values.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_acc = 0.0
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_best.pth")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        
        # Train
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(float(epoch_acc))
        
        # Val
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        history["val_loss"].append(val_loss)
        history["val_acc"].append(float(val_acc))
        
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved -> {save_path}")
            
    return model, save_path, history
