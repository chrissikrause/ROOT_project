import torch
import torch.nn as nn
import torch.nn.functional as F


class TempCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=2, kernel_size=5, num_filters=64, num_layers=3, dropout=0.5):
        super(TempCNN, self).__init__()

        layers = []
        for i in range(num_layers):
            in_channels = input_channels if i == 0 else num_filters
            layers.append(nn.Conv1d(in_channels, num_filters, kernel_size, padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(num_filters))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.conv_layers = nn.Sequential(*layers)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)  # Remove the last dimension
        x = self.fc(x)
        return x


# Example usage
if __name__ == "__main__":
    batch_size = 16
    time_steps = 30  # Number of time points in the Disturbance Index time series
    model = TempCNN(input_channels=1, num_classes=2)

    sample_input = torch.randn(batch_size, 1, time_steps)  # [batch, channels, time]
    output = model(sample_input)
    print(output.shape)  # Expected output: [batch_size, num_classes]
