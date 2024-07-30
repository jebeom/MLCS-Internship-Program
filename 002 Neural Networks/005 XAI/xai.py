############################################
# Title     : Image classification with CAM
# Author    : Jebeom Chae
# Reference : Jaewoong Han
# Date    : 2024-07-09
# Dataset : CIFAR-10
###########################################

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import skimage.transform

num_epochs = 15
batch_size = 64
learning_rate = 0.00001

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load CIFAR-10 dataset
train_val_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

# train dataset split
train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size

train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train dataset size: {len(train_loader.dataset)}")
print(f"Validation dataset size: {len(val_loader.dataset)}")
print(f"Test dataset size: {len(test_loader.dataset)}")

image, label = next(iter(train_loader))
print(f"Image shape: {image.shape}")

base_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
base_model.classifier = nn.Identity()

class VGG16_with_GAP(nn.Module):
    def __init__(self):
        super(VGG16_with_GAP, self).__init__()
        self.base = base_model.features
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 10)

        for param in list(self.base.parameters())[:-4]:
            param.requires_grad = False

    def forward(self, x):
        x = self.base(x)
        features = x.clone()
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, features

model = VGG16_with_GAP()

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

best_accuracy = 0.0
best_model_weights = model.state_dict()

# Train
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset) * 100

    print(f'Epoch {str(epoch+1).zfill(2)}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f} %')

    # Validation
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

            val_outputs, _ = model(val_inputs)
            val_loss = criterion(val_outputs, val_labels)

            _, val_preds = torch.max(val_outputs, 1)
            val_running_loss += val_loss.item() * val_inputs.size(0)
            val_running_corrects += torch.sum(val_preds == val_labels.data)

    val_epoch_loss = val_running_loss / len(val_loader.dataset)
    val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset) * 100

    print(f'Validation Loss: {val_epoch_loss:.4f}, '
          f'Validation Accuracy: {val_epoch_acc:.4f} %')

    # best model save
    if val_epoch_acc > best_accuracy:
        best_accuracy = val_epoch_acc
        best_model_weights = model.state_dict()

# setting to best model
model.load_state_dict(best_model_weights)


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

model.eval()
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
params = list(model.parameters())[-2]

for data in test_loader:
    images, labels = data
    images = images.to(device)
    labels = labels.to(device)
    outputs, features = model(images)
    _, predicted = torch.max(outputs, 1)
    break

# CAM visualization
for num in range(64):
    print(f"Item {str(num+1).zfill(2)}/64 | Ground Truth: {classes[int(labels[num])]} | Prediction: {classes[int(predicted[num])]}")

    overlay = params[int(predicted[num])].matmul(features[num].reshape(512, 49)).reshape(7, 7).cpu().data.numpy()

    overlay = overlay - np.min(overlay)
    overlay = overlay / np.max(overlay)
    overlay_resized = skimage.transform.resize(overlay, [224, 224])

    original_image = images[num].cpu()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    img = original_image.permute(1, 2, 0).numpy()
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    ax[0].imshow(img)
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(img)
    ax[1].imshow(overlay_resized, alpha=0.4, cmap='jet')
    ax[1].set_title("Learned Overlay")
    ax[1].axis('off')

    plt.show()

# Test
model.eval()
test_running_corrects = 0
y_true = []
y_score = []

with torch.no_grad():
    for test_inputs, test_labels in test_loader:
        test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

        test_outputs, _ = model(test_inputs)
        _, test_preds = torch.max(test_outputs, 1)

        test_running_corrects += torch.sum(test_preds == test_labels.data)

        y_true.extend(test_labels.cpu().numpy())
        y_score.extend(torch.softmax(test_outputs, dim=1).cpu().numpy())

test_accuracy = test_running_corrects.double() / len(test_loader.dataset) * 100
print(f'Test Accuracy: {test_accuracy:.4f} %')

# Calculate AUC
y_true = label_binarize(y_true, classes=list(range(10)))
auc = roc_auc_score(y_true, y_score, multi_class='ovo')
print(f'AUC: {auc:.4f}')

