import os

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

batch_size = 32
lr = 0.001
epochs = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
os.makedirs("./outputs", exist_ok=True)

transform_train = transforms.Compose([
    transforms.Resize((128, 128)),          # 原来是 (224,224)
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

class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classfiler = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128 * 16 * 16, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classfiler(x)
        return x

model = CNNClassifier().to(device)
print(model)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def train_one_epoch(loader , model , criterion, optimizer):
    model.train()
    total_loss = 0
    total = 0
    correct = 0
    for imgs , labs in loader:
        imgs , labs = imgs.to(device), labs.to(device).float().unsqueeze(1)
        outputs = model(imgs)
        loss = criterion(outputs, labs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labs.size(0)
        correct += (predicted == labs).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total    # 改为乘以100
    return avg_loss, accuracy


def validate(loader, model, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

train_losses, train_accs = [], []
val_losses, val_accs = [], []

print("准备开始训练...")

for epoch in range(epochs):
    print(f"Epoch {epoch + 1} 开始")
    train_loss, train_acc = train_one_epoch(train_loader, model, criterion, optimizer)
    val_loss, val_acc = validate(val_loader, model, criterion)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{epochs}: "
          f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | "
          f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

# 保存模型权重
torch.save(model.state_dict(), "./outputs/catdog_cnn.pth")
print("模型已保存至 outputs/catdog_cnn.pth")

# 绘制训练曲线
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.savefig("./outputs/training_curve.png", dpi=300, bbox_inches='tight')
plt.show()