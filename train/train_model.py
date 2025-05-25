import torch
from torch.utils.tensorboard import SummaryWriter
from utils.early_stopping import EarlyStopping
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def train_model(model, criterion, optimizer, train_loader, val_loader, input_length, num_epochs=30, log_dir="runs"):
    writer = SummaryWriter(log_dir=log_dir)
    input = torch.randn(1, 1, input_length)
    writer.add_graph(model, input)

    early_stopping = EarlyStopping(patience=5, monitor='val_acc')
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            writer.add_scalar("Loss/train", loss.item(), epoch)

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
        val_accuracy = 100 * val_correct / val_total        
        early_stopping(val_accuracy, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

    writer.close()
    return model
