############################################
# Title : Image(MNIST) classification with MLP - Train
# Author: Jebeom Chae
# Referench : Jaewoong Han
# Date:   2024-07-04
###########################################

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

batch_size = 512
num_classes = 10
epochs_1 = 25
epochs_2 = 25

# Dataset load
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Data Load
train_dataset = datasets.MNIST(root='../data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='../data', train=False, transform=transform, download=True)

x_train = train_dataset.data.numpy()
y_train = train_dataset.targets.numpy()
x_test = test_dataset.data.numpy()
y_test = test_dataset.targets.numpy()

# Concatenate
x = np.concatenate((x_train, x_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

# Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=41)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.15/0.85, random_state=41)

# Numpy to Tensor 
x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1) / 255.0
x_valid = torch.tensor(x_valid, dtype=torch.float32).unsqueeze(1) / 255.0
x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1) / 255.0
y_train = torch.tensor(y_train, dtype=torch.long)
y_valid = torch.tensor(y_valid, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Data Loader
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(TensorDataset(x_valid, y_valid), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

# Model 1
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.fc1 = nn.Linear(784, 512) 
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Model 2
class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)  
        self.fc4 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Train & Eval
def train_and_evaluate(model, train_loader, valid_loader, epochs, criterion, optimizer):
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print(f'Epoch [{str(epoch+1).zfill(2)}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        model.eval()
        valid_loss, valid_correct = 0, 0
        with torch.no_grad():
            for images, labels in valid_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                valid_correct += (predicted == labels).sum().item()
        
        valid_accuracy = 100 * valid_correct / len(valid_loader.dataset)
        print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {valid_loss/len(valid_loader)}, Validation Accuracy: {valid_accuracy}%')

    # save model
    torch.save(model.state_dict(), f'model_{model.__class__.__name__}.pth')

if __name__ == '__main__':
    # Model 1 Train
    model_1 = Model1()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model_1.parameters())
    train_and_evaluate(model_1, train_loader, valid_loader, epochs_1, criterion, optimizer)

    # Model 2 Train
    model_2 = Model2()
    optimizer = optim.Adam(model_2.parameters(), lr=0.001)
    train_and_evaluate(model_2, train_loader, valid_loader, epochs_2, criterion, optimizer)
