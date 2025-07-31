'''
This script uses the Optuna framework for hyperparameter optimization of a Temporal CNN
to classify DI-time series
'''

import optuna
import torch
import torch.nn as nn
import os
from optuna.pruners import MedianPruner
import sys
import json
from sklearn.metrics import precision_score, recall_score, f1_score
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.temporal_cnn import TemporalCNN
from utils.data_loader import load_and_preprocess_data
from train.train_model import train_model
from test.evaluate_model import evaluate_model


def objective(trial, train_loader, val_loader, input_length, weights, output_dir, time_tag):
    # Define hyperparameter search
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    num_filters = trial.suggest_categorical("num_filters", [16, 32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    # Compute on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = weights.to(device)

    # Build model
    model = TemporalCNN(input_channels=1, num_classes=3, num_filters=num_filters, dropout=dropout).to(device)

    # Weighted Cross Entropy Loss
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Test different Optimizer
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)


    num_epochs = 50
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    trial_run_dir = os.path.join(output_dir, time_tag, "trials", f"trial_{trial.number}")
    os.makedirs(trial_run_dir, exist_ok=True)

    # Train model
    trained_model = train_model(
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        input_length,
        num_epochs=num_epochs,
        log_dir=os.path.join(trial_run_dir, "tensorboard"),
        trial=trial,
        scheduler=scheduler,
        output_dir="output/weights/6months" 
    )

    # Get vall acc and loss
    model.eval()
    val_loss = 0.0
    val_total = 0
    val_correct = 0
    all_preds_list = []
    all_labels_list = []

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

            all_preds_list.extend(predicted.cpu().numpy())  # zurück auf CPU fürs Logging
            all_labels_list.extend(y_batch.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total


    # Calculate Precision, Recall, F1
    precision = precision_score(all_labels_list, all_preds_list, average="weighted", zero_division=0)
    recall = recall_score(all_labels_list, all_preds_list, average="weighted", zero_division=0)
    f1 = f1_score(all_labels_list, all_preds_list, average="weighted", zero_division=0)

    # Write to json summary
    summary = {
        "trial_number": trial.number,
        "value": avg_val_loss,
        "accuracy": val_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "params": trial.params,
    }
    
    with open(os.path.join(trial_run_dir, f"trial_{trial.number}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return avg_val_loss  # or: return 1.0 - val_acc




if __name__ == "__main__":
    time_tag = "6months"
    input_csv_path = f"data/extracted_DI_polygons/all_polygons_time_series_wide_interpolated_{time_tag}.csv"
    output_dir = "output/weights6"

    # Data loading and preprocessing
    train_loader, val_loader, test_loader, input_length, weights, split_indices = load_and_preprocess_data(input_csv_path)

    study = optuna.create_study(
        direction="minimize",
        pruner=MedianPruner(n_warmup_steps=5, n_startup_trials=3)
    )
    
    study.optimize(
    lambda trial: objective(
        trial,
        train_loader=train_loader,
        val_loader=val_loader,
        input_length=input_length,
        weights=weights,
        output_dir=output_dir,
        time_tag=time_tag
    ),
    n_trials=50
)

    best_trial = study.best_trial
    best_trial_number = best_trial.number
    best_params = best_trial.params

    trial_dir = os.path.join(output_dir, time_tag, "trials")
    os.makedirs(trial_dir, exist_ok=True)

    with open(os.path.join(trial_dir, "best_trial_number.txt"), "w") as f:
        f.write(str(best_trial_number))

    df = study.trials_dataframe()
    df.to_csv(os.path.join(trial_dir, f"optuna_results_{time_tag}.csv"), index=False)


    # Reconstruct best model
    model = TemporalCNN(
        input_channels=1,
        num_classes=3,
        num_filters=best_params["num_filters"],
        dropout=best_params["dropout"]
    )

    best_model_path = os.path.join(trial_dir, f"trial_{best_trial_number}", f"best_model_trial_{best_trial_number}.pth")
    model.load_state_dict(torch.load(best_model_path, map_location="cpu"))

    evaluate_model(model, test_loader, trial_number=best_trial_number, output_dir=os.path.join(output_dir, time_tag))