import optuna
import torch
import torch.nn as nn
import os
from optuna.pruners import MedianPruner
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print("sys.path:", sys.path)

from models.temporal_cnn import TemporalCNN
from utils.data_loader import load_and_preprocess_data
from train.train_model import train_model
from test.evaluate_model import evaluate_model

def objective(trial):
    # Hyperparameter-Suchraum
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    num_filters = trial.suggest_categorical("num_filters", [16, 32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    # Load data
    train_loader, val_loader, _, input_length, weights = load_and_preprocess_data("data/extracted_DI_polygons/all_polygons_time_series_wide_interpolated_6months.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = weights.to(device)

    # Build model
    model = TemporalCNN(input_channels=1, num_classes=3, num_filters=num_filters, dropout=dropout).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)


    # Train model
    trained_model = train_model(model, criterion, optimizer, train_loader, val_loader, input_length, num_epochs=30, trial=trial)

    # Get vall acc and loss
    model.eval()
    val_loss = 0.0
    val_total = 0
    val_correct = 0

    with torch.no_grad():
        for X_batch, y_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += y_batch.size(0)
            val_correct += (predicted == y_batch).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total

    
    return avg_val_loss  # or: return 1.0 - val_acc

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize",
    pruner=MedianPruner(n_warmup_steps=3, n_startup_trials=3))  # or "maximize" für val_acc
    study.optimize(objective, n_trials=30)

    print("Beste Parameter:")
    print(study.best_params)
    print("Beste Validierungs-LOSS:", study.best_value)

    df = study.trials_dataframe()
    df.to_csv("optuna_results.csv", index=False)