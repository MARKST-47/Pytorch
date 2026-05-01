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
