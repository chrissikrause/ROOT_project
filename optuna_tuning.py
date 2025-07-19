import optuna
import torch
import torch.nn as nn
import os
from optuna.pruners import MedianPruner
import sys
import json
from sklearn.metrics import precision_score, recall_score, f1_score
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


    num_epochs = 50
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    trained_model = train_model(
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        input_length,
        num_epochs=num_epochs,
        log_dir=f"runs/trial_{trial.number}",
        trial=trial,
        scheduler=scheduler
    )

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

    # Zusätzliche Metriken berechnen
    all_preds_list = []
    all_labels_list = []

    with torch.no_grad():
        for X_batch, y_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            all_preds_list.extend(predicted.cpu().numpy())
            all_labels_list.extend(y_batch.cpu().numpy())

    # Berechne Precision, Recall, F1
    precision = precision_score(all_labels_list, all_preds_list, average="weighted", zero_division=0)
    recall = recall_score(all_labels_list, all_preds_list, average="weighted", zero_division=0)
    f1 = f1_score(all_labels_list, all_preds_list, average="weighted", zero_division=0)


    # Ergebnis-Logging
    trial_id = trial.number
    os.makedirs("output/trials", exist_ok=True)
    summary = {
        "trial_number": trial_id,
        "value": avg_val_loss,
        "accuracy": val_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "params": trial.params,
    }
    
    with open(f"output/trials/trial_{trial_id}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    
    return avg_val_loss  # or: return 1.0 - val_acc

if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        pruner=MedianPruner(n_warmup_steps=5, n_startup_trials=3)
    )
    study.optimize(objective, n_trials=50)

    best_trial = study.best_trial
    best_trial_number = best_trial.number
    best_params = best_trial.params

    os.makedirs("output", exist_ok=True)
    with open("output/best_trial_number.txt", "w") as f:
        f.write(str(best_trial_number))

    print("Beste Parameter:")
    print(best_params)
    print("Beste Validierungs-LOSS:", study.best_value)

    df = study.trials_dataframe()
    df.to_csv("optuna_results.csv", index=False)

    # Daten neu laden
    _, _, test_loader, input_length, _ = load_and_preprocess_data(
        "data/extracted_DI_polygons/all_polygons_time_series_wide_interpolated_6months.csv"
    )

    # Modell rekonstruieren
    model = TemporalCNN(
        input_channels=1,
        num_classes=3,
        num_filters=best_params["num_filters"],
        dropout=best_params["dropout"]
    )

    # Pfad zum besten Modell (früher gespeichert in train_model mit EarlyStopping)
    best_model_path = f"output/trial_{best_trial_number}/best_model_trial_{best_trial_number}.pth"
    model.load_state_dict(torch.load(best_model_path, map_location="cpu"))

    # Testen
    evaluate_model(model, test_loader, trial_number=best_trial_number, output_dir="output")