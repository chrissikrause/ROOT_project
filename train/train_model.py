import torch
from torch.utils.tensorboard import SummaryWriter
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.early_stopping import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import optuna


def train_model(model, criterion, optimizer, train_loader, val_loader, input_length, num_epochs, log_dir="runs", trial=None, scheduler=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device) 

    # Trial-spezifische Pfade
    trial_number = trial.number if trial is not None else "manual"
    trial_output_dir = os.path.join("output/weights/6months/trials/", f"trial_{trial_number}")
    os.makedirs(trial_output_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(log_dir, f"trial_{trial_number}"))
    dummy_input = torch.randn(1, 1, input_length).to(device)
    writer.add_graph(model, dummy_input)

    early_stopping = EarlyStopping(patience=5, path=os.path.join(trial_output_dir, f"best_model_trial_{trial_number}.pth"), monitor='val_loss')
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch, idxs in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if scheduler is not None:
            scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)

        

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch, idxs in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        val_accuracy = 100 * val_correct / val_total        

        # Early Stopping
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

        # Optuna Pruning: Bricht trial ab, wenn vorherige Trials an der Epoche besser
        if trial is not None:
            trial.report(avg_val_loss, step=epoch)
            if trial.should_prune():
                print(f"Pruned at epoch {epoch}")
                raise optuna.exceptions.TrialPruned()

        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

    writer.close()

    np.save(os.path.join(trial_output_dir, "train_losses.npy"), train_losses)
    np.save(os.path.join(trial_output_dir, "val_losses.npy"), val_losses)
    
    # Plot train + val loss
    plt.figure(figsize=(8,6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(trial_output_dir, "loss_epochs_val_test.png"))
    plt.close()

    return model
