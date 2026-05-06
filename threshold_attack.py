import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# Random seed
torch.manual_seed(10)
np.random.seed(10)

print("Loading CIFAR-10...")
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

full_trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=False, transform=transform
)
full_testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=False, transform=transform
)

# To make model overfit quick, we use a small subset
# (private data)
train_subset = Subset(full_trainset, range(5000))  # The "Train" set is our "Members"
test_subset = Subset(full_testset, range(5000))  # The "Test" set is "Non-Members"

trainloader = DataLoader(train_subset, batch_size=64, shuffle=True)
testloader = DataLoader(test_subset, batch_size=64, shuffle=False)


# Target model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\nTraining Target Model (5 Epochs)...")
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs} | Loss: {running_loss / len(trainloader):.4f}")

print("Training Complete.")
