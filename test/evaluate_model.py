import torch
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

def evaluate_model(model, test_loader, save_path="output/confusion_matrix_test.png"):
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds, all_labels = [], []
    all_ids = []
    misclassified_records = []

    with torch.no_grad():
        for X_batch, y_batch, idxs in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            test_total += y_batch.size(0)
            test_correct += (predicted == y_batch).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_ids.extend(idxs.cpu().numpy())

            # Sammle Infos über Fehlklassifikationen
            for idx, pred, true in zip(idxs, predicted, y_batch):
                if pred != true:
                    misclassified_records.append({
                        'point_id': idx.item(),
                        'true_class': true.item(),
                        'predicted_class': pred.item()
                    })

    test_accuracy = 100 * test_correct / test_total
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Speichern der vollständigen Testergebnisse
    os.makedirs("output", exist_ok=True)
    results_df = pd.DataFrame({
        'point_id': all_ids,
        'true_class': all_labels,
        'predicted_class': all_preds
    })
    results_df.to_csv("output/test_predictions.csv", index=False)
    print("Gesamte Test-Predictions gespeichert in output/test_predictions.csv")

    # Speichern der Fehlklassifizierungen
    misclassified_df = pd.DataFrame(misclassified_records)
    misclassified_df.to_csv("output/misclassified_points.csv", index=False)
    print("Fehlklassifizierte Beispiele gespeichert in output/misclassified_points.csv")

    # Klassifikationsbericht
    print(classification_report(all_labels, all_preds))

    # Konfusionsmatrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ausgeräumte\nWaldflächen (1)', 'Stehendes\nTotholz (2)', 'Gesunder\nWald (3)'],
                yticklabels=['Ausgeräumte\nWaldflächen (1)', 'Stehendes\nTotholz (2)', 'Gesunder\nWald (3)'],
                cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)

