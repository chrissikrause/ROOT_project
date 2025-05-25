import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalCNN(nn.Module):
    def __init__(self, input_channels, num_classes, kernel_size=3, dropout=0.2):
        super(TemporalCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(64, 128, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv1d(128, 256, kernel_size, padding=kernel_size//2)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(dropout)

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        x = torch.mean(x, dim=2)
        return self.fc(x)
