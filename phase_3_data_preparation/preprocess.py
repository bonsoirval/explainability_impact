import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class PlantDiseaseDataset(Dataset):
    """
    Loads leaf disease images from directory structure.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset directory {root_dir} does not exist. Run Phase 2 first.")
            
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)
            
        return img_tensor, label

def get_dataloaders(root_dir="data/PlantVillage", batch_size=8, val_split=0.2):
    """
    Creates train and validation PyTorch dataloaders.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset_train = PlantDiseaseDataset(root_dir=root_dir, transform=train_transform)
    full_dataset_val = PlantDiseaseDataset(root_dir=root_dir, transform=val_transform)
    
    n_samples = len(full_dataset_train)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val
    
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(n_samples, generator=generator).tolist()
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    train_subset = torch.utils.data.Subset(full_dataset_train, train_idx)
    val_subset = torch.utils.data.Subset(full_dataset_val, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, full_dataset_train.classes
