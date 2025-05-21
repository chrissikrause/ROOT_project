import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter

class TemporalCNN(nn.Module):
    def __init__(self, input_channels, num_classes, kernel_size=3, dropout=0.2):
        super(TemporalCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(dropout)

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (batch_size, input_channels, sequence_length)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)

        # Global Average Pooling
        x = torch.mean(x, dim=2)

        x = self.fc(x)
        return x


# CSV-Datei einlesen
df = pd.read_csv('/Users/christinakrause/HIWI_DLR_Forest/Data_Collection/DI_points_timeseries/combined_time_series_wide.csv')

# Fehlende Werte behandeln
df.replace(-2147483648.0, np.nan, inplace=True)
# Lineare Interpolation entlang der Spalten durchführen
df.interpolate(method='linear', axis=0, inplace=True)
df.bfill(axis=0, inplace=True)
df.ffill(axis=0, inplace=True)

time_cols = [col for col in df.columns if col.startswith('di_t')]
df[time_cols] = df[time_cols].interpolate(axis=1).ffill(axis=1).bfill(axis=1)

# Features und Labels extrahieren
X = df[time_cols].values
y = df['class'].values - 1  # Klassenlabels auf 0 und 1 setzen

# Daten normalisieren
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Daten in Trainings- und Testsets aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Daten in Tensoren umwandeln
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # (batch_size, channels, sequence_length)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# DataLoader erstellen
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Modell, Loss-Funktion und Optimierer definieren
model = TemporalCNN(input_channels=1, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Modell in den Evaluierungsmodus versetzen
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")


# Validierung
# -------------------------------
# 1. Klassenverteilung prüfen
# -------------------------------
class_counts = Counter(y)
print("Klassenverteilung:", class_counts)

# -------------------------------
# 2. Confusion Matrix & Klassifikationsbericht
# -------------------------------
# Vorhersagen auf dem Testset
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(y_batch.numpy())

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("/Users/christinakrause/HIWI_DLR_Forest/ROOT_project/confusion_matrix.png", dpi=300)

# Klassifikationsbericht
print("Classification Report:")
print(classification_report(all_labels, all_preds, digits=4))

# -------------------------------
# 3. Cross-Validation (optional)
# -------------------------------
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    print(f"\nFold {fold+1}")
    X_train_fold = torch.tensor(X[train_idx], dtype=torch.float32).unsqueeze(1)
    y_train_fold = torch.tensor(y[train_idx], dtype=torch.long)
    X_val_fold = torch.tensor(X[val_idx], dtype=torch.float32).unsqueeze(1)
    y_val_fold = torch.tensor(y[val_idx], dtype=torch.long)

    train_dataset = TensorDataset(X_train_fold, y_train_fold)
    val_dataset = TensorDataset(X_val_fold, y_val_fold)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = TemporalCNN(input_channels=1, num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):  # Weniger Epochen pro Fold
        model.train()
        for X_batch, y_batch in train_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Bewertung auf Validierungsfold
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    acc = 100 * correct / total
    cv_scores.append(acc)
    print(f"Validation Accuracy: {acc:.2f}%")

print("\nCross-Validation Accuracy Scores:", cv_scores)
print("Mean CV Accuracy:", sum(cv_scores)/len(cv_scores))

# -------------------------------
# 4. Trainings- und Testverlust plotten
# -------------------------------
train_losses = []
test_losses = []

model = TemporalCNN(input_channels=1, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(20):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Test Loss
    model.eval()
    running_test_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            running_test_loss += loss.item()
    avg_test_loss = running_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Test Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("/Users/christinakrause/HIWI_DLR_Forest/ROOT_project/loss_epochs.png", dpi=300)

