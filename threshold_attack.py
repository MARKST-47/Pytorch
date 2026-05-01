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
