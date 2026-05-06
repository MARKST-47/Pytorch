import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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

print("\nTraining Target Model (50 Epochs)...")
epochs = 50
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

print("\nRunning Threshold Membership Inference Attack:")


# Helper function to get maximum confidence score for a dataset
def get_confidence_scores(model, dataloader):
    model.eval()
    confidences = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            outputs = model(inputs)
            # Convert raw logits to probabilities (0 to 1)
            probs = torch.softmax(outputs, dim=1)
            # The model's confidence is the highest probability among the 10 classes
            max_probs, _ = torch.max(probs, dim=1)
            confidences.extend(max_probs.numpy())
    return np.array(confidences)


# Extract scores for Members (Train) and Non-Members (Test)
member_scores = get_confidence_scores(model, trainloader)
non_member_scores = get_confidence_scores(model, testloader)

# Define our attack rule: "If confidence > 90%, we accuse them of being in the training set."
THRESHOLD = 0.90

# Calculate True Positives (Members correctly guessed)
true_positives = np.sum(member_scores >= THRESHOLD)
tpr = (true_positives / len(member_scores)) * 100

# Calculate False Positives (Non-members falsely accused)
false_positives = np.sum(non_member_scores >= THRESHOLD)
fpr = (false_positives / len(non_member_scores)) * 100

print(f"Threshold set to: {THRESHOLD * 100}% Confidence")
print(
    f"True Positive Rate (TPR): {tpr:.2f}% ({true_positives} out of {len(member_scores)} members found)"
)
print(
    f"False Positive Rate (FPR): {fpr:.2f}% ({false_positives} innocent people falsely accused)"
)

print("\nTHE SHADOW MODEL ATTACK")

# 1. Create a "Shadow" Dataset (Data the Target Model never saw)
# We use the next 5000 images for training the shadow model, and 5000 test images for its holdout set.
shadow_train_subset = Subset(full_trainset, range(5000, 10000))
shadow_test_subset = Subset(full_testset, range(5000, 10000))

shadow_trainloader = DataLoader(shadow_train_subset, batch_size=64, shuffle=True)
shadow_testloader = DataLoader(shadow_test_subset, batch_size=64, shuffle=False)

# 2. Train the Shadow Model (Must have the exact same architecture as Target)
print("\nTraining Shadow Model (Mimicking the Target Model):")
shadow_model = CNN()
shadow_optimizer = optim.Adam(shadow_model.parameters(), lr=0.001)

# We train it for the same number of epochs so it learns and overfits in the exact same way
for epoch in range(epochs):
    shadow_model.train()
    for inputs, labels in shadow_trainloader:
        shadow_optimizer.zero_grad()
        outputs = shadow_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        shadow_optimizer.step()
print("Shadow Model Training Complete.")


# 3. Feature Extraction: Get the "Fingerprints"
# Instead of just the top confidence score, we give the ML attack model the top 3 sorted probabilities.
# This gives the attack model much more statistical information about the model's uncertainty.
def extract_attack_features(model, dataloader):
    model.eval()
    features = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            # Sort probabilities descending and take the top 3
            sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
            top_3_probs = sorted_probs[:, :3].numpy()
            features.extend(top_3_probs)
    return np.array(features)


print("\nExtracting Features from Shadow Model:")
# Shadow Train = Members (Label 1)
shadow_member_features = extract_attack_features(shadow_model, shadow_trainloader)
shadow_member_labels = np.ones(len(shadow_member_features))

# Shadow Test = Non-Members (Label 0)
shadow_nonmember_features = extract_attack_features(shadow_model, shadow_testloader)
shadow_nonmember_labels = np.zeros(len(shadow_nonmember_features))

# Combine to create the training dataset for our Attack Model
X_attack_train = np.vstack((shadow_member_features, shadow_nonmember_features))
y_attack_train = np.concatenate((shadow_member_labels, shadow_nonmember_labels))

# 4. Train the Attack Model (A Random Forest Classifier)
print("Training the Attack Model (Random Forest):")
attack_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
attack_model.fit(X_attack_train, y_attack_train)

# 5. Execute the Attack on the REAL Target Model
print("\nExecuting Shadow Model Attack on Target Model:")
# Extract features from the original Target Model
target_member_features = extract_attack_features(model, trainloader)
target_nonmember_features = extract_attack_features(model, testloader)

# Combine for evaluation
X_attack_test = np.vstack((target_member_features, target_nonmember_features))
y_attack_test = np.concatenate(
    (np.ones(len(target_member_features)), np.zeros(len(target_nonmember_features)))
)

# Predict membership
attack_predictions = attack_model.predict(X_attack_test)

# Calculate final TPR and FPR
target_members_mask = y_attack_test == 1
target_nonmembers_mask = y_attack_test == 0

ml_true_positives = np.sum((attack_predictions == 1) & target_members_mask)
ml_tpr = (ml_true_positives / len(target_member_features)) * 100

ml_false_positives = np.sum((attack_predictions == 1) & target_nonmembers_mask)
ml_fpr = (ml_false_positives / len(target_nonmember_features)) * 100

print(f"ML Attack TPR: {ml_tpr:.2f}% ({ml_true_positives} members found)")
print(
    f"ML Attack FPR: {ml_fpr:.2f}% ({ml_false_positives} innocent people falsely accused)"
)
print(
    f"Overall Attack Accuracy: {accuracy_score(y_attack_test, attack_predictions) * 100:.2f}%"
)
