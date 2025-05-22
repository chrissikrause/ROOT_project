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
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from collections import Counter
print("Packages imported sucessfully")

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

# Nur für zwei Klassen
df = df[df['class'].isin([1, 2])]

# Features und Labels extrahieren
X = df[time_cols].values
y = df['class'].values - 1  # Klassenlabels auf 0 und 1 setzen

# Daten normalisieren
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Gesamtdaten in Train+Test splitten
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Train+Val in Train und Val aufteilen
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42)


# Tensoren erstellen
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Datasets & Loader
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=32)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32)

# Modell, Loss-Funktion und Optimierer definieren
model = TemporalCNN(input_channels=1, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_losses = []
val_losses = []
# Training
num_epochs = 20
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validierung
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += y_batch.size(0)
            val_correct += (predicted == y_batch).sum().item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    val_accuracy = 100 * val_correct / val_total

    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("/Users/christinakrause/HIWI_DLR_Forest/ROOT_project/loss_epochs_val_test.png", dpi=300)


# Validierung
# Test-Set Bewertung
model.eval()
test_correct = 0
test_total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        test_total += y_batch.size(0)
        test_correct += (predicted == y_batch).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

test_accuracy = 100 * test_correct / test_total
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Klassifikationsbericht und Confusion-Matrix
print(classification_report(all_labels, all_preds))
cm = confusion_matrix(all_labels, all_preds)

# Visualisierung
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Test Set')
plt.tight_layout()
plt.savefig('/Users/christinakrause/HIWI_DLR_Forest/ROOT_project/confusion_matrix_test.png')  # Da du Agg verwendest

