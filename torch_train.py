import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define a simple feedforward neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Set the path to your 'data' folder
data_folder = 'data'  # Adjust this path based on your directory structure

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to a consistent size
    transforms.ToTensor(),  # Convert images to PyTorch tensors
])

# Load the image data from the 'data' folder
dataset = ImageFolder(root=data_folder, transform=transform)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(dataset, dataset.targets, test_size=0.2, shuffle=True, stratify=dataset.targets)

# Create data loaders for training and testing
train_loader = DataLoader(x_train, batch_size=64, shuffle=True)
test_loader = DataLoader(x_test, batch_size=64, shuffle=False)

# Define the neural network model
input_size = len(dataset[0][0].view(-1))  # Calculate input size from the first sample
num_classes = len(dataset.classes)
model = SimpleNN(input_size, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images.view(images.size(0), -1))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
predicted_labels = []
true_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.view(images.size(0), -1))
        _, predicted = torch.max(outputs, 1)
        predicted_labels.extend(predicted.numpy())
        true_labels.extend(labels.numpy())

accuracy = accuracy_score(true_labels, predicted_labels)
print('{}% of samples were classified correctly!'.format(accuracy * 100))

# Save the trained model
torch.save(model.state_dict(), 'torch_model1.pth')
