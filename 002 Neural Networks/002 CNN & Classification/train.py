############################################
# Title     : Image classification with CNN
# Author    : Jebeom Chae
# Reference : Jaewoong Han
# Date    : 2024-07-08
# Dataset : CIFAR-10
# Pre-trained Model   : ResNet18
###########################################

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from cnn_network import ResNet18

# Resize and Normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet18(num_classes=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0002)
loss_func = nn.CrossEntropyLoss()

# Train
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    corrects = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model.forward(images)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        corrects += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss /= len(train_loader)
    epoch_acc = corrects / total * 100
    print(f"epoch: [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.5f}, Accuracy: {epoch_acc:.5f}%")

# Test
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Accuracy of Test Data: {100 * correct / total:.5f}%")
