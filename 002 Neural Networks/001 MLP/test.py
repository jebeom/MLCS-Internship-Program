############################################
# Title : Image(MNIST) classification with MLP - Test
# Author: Jebeom Chae
# Reference: Jaewoong Han
# Date:   2024-07-04
############################################

import torch
from train import Model1, Model2, test_loader  
from torch import nn
import matplotlib.pyplot as plt

# Load models
model_1 = Model1()
model_2 = Model2()

# Load the saved model parameters
model_1.load_state_dict(torch.load('model_Model1.pth'))
model_2.load_state_dict(torch.load('model_Model2.pth'))



# Define loss function
criterion = nn.CrossEntropyLoss()

# Evaluate Model 1
model_1.eval()  # Set model to evaluation mode
test_loss_1, test_correct_1 = 0, 0
examples_1 = []
num_examples_1 = 16
with torch.no_grad():  # Disable gradient computation
    for images, labels in test_loader:
        outputs = model_1(images)
        loss = criterion(outputs, labels)
        test_loss_1 += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        test_correct_1 += (predicted == labels).sum().item()

        if len(examples_1) < num_examples_1:
            for i in range(len(labels)):
                if len(examples_1) < num_examples_1:
                    examples_1.append((images[i], labels[i], predicted[i]))
                else:
                    break

fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, (image, label, prediction) in enumerate(examples_1):
    image = image.squeeze().numpy() * 0.5 + 0.5
    ax = axes[i // 4, i % 4]
    ax.imshow(image, cmap='gray')
    ax.set_title(f'True: {label.item()}\nPred: {prediction.item()}', fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.show()

test_accuracy_1 = 100 * test_correct_1 / len(test_loader.dataset)
print(f'Test Loss of Model 1: {test_loss_1 / len(test_loader)}')
print(f'Test Accuracy of Model 1: {test_accuracy_1}%')

# Evaluate Model 2
model_2.eval()  # Set model to evaluation mode
test_loss_2, test_correct_2 = 0, 0
examples_2 = []
num_examples_2 = 16
with torch.no_grad():  # Disable gradient computation
    for images, labels in test_loader:
        outputs = model_2(images)
        loss = criterion(outputs, labels)
        test_loss_2 += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        test_correct_2 += (predicted == labels).sum().item()

        if len(examples_2) < num_examples_2:
                    for i in range(len(labels)):
                        if len(examples_2) < num_examples_2:
                            examples_2.append((images[i], labels[i], predicted[i]))
                        else:
                            break

fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, (image, label, prediction) in enumerate(examples_2):
    image = image.squeeze().numpy() * 0.5 + 0.5
    ax = axes[i // 4, i % 4]
    ax.imshow(image, cmap='gray')
    ax.set_title(f'True: {label.item()}\nPred: {prediction.item()}', fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.show()

test_accuracy_2 = 100 * test_correct_2 / len(test_loader.dataset)

print(f'Test Loss of Model 2: {test_loss_2 / len(test_loader)}')
print(f'Test Accuracy of Model 2: {test_accuracy_2}%')

