import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalCNN(nn.Module):
    def __init__(
        self,
        input_channels,
        num_classes,
        kernel_size=3,
        dropout=0.2,
        num_filters=64
    ):
        super(TemporalCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, num_filters, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(num_filters, num_filters*2, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(num_filters*2)
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv1d(num_filters*2, num_filters*4, kernel_size, padding=kernel_size // 2)
        self.bn3 = nn.BatchNorm1d(num_filters*4)
        self.dropout3 = nn.Dropout(dropout)

        self.fc = nn.Linear(num_filters*4, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        x = torch.mean(x, dim=2)  # Global Average Pooling
        return self.fc(x)

