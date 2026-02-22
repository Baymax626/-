import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

batch_size = 32
lr = 0.001
epochs = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
os.makedirs("./outputs", exist_ok=True)

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ]
)

train_dataset = datasets.ImageFolder(
    root="./data/train",
    transform=transform_train
)

val_dataset = datasets.ImageFolder(
    root="./data/val",
    transform=transform_val
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False
)

print(f"训练集类别: {train_dataset.classes}")
print(f"训练集样本数: {len(train_dataset)}")
print(f"验证集样本数: {len(val_dataset)}")