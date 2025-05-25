from models.temporal_cnn import TemporalCNN
from utils.data_loader import load_and_preprocess_data
from train.train_model import train_model
from test.evaluate_model import evaluate_model
import torch.nn as nn
import torch


def main():
    train_loader, val_loader, test_loader, input_length = load_and_preprocess_data("data/combined_time_series_wide.csv")
    model = TemporalCNN(input_channels=1, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model = train_model(model, criterion, optimizer, train_loader, val_loader, input_length)
    evaluate_model(model, test_loader)

if __name__ == "__main__":
    main()
