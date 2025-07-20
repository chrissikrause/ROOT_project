import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import re


def load_and_preprocess_data(path, batch_size=32):
    df = pd.read_csv(path)
    df.replace(-2147483648.0, np.nan, inplace=True)

    time_cols = [col for col in df.columns if col.startswith('di_t')]
    df = df[df['class'].isin([1, 2, 3])]
    df[time_cols] = df[time_cols].interpolate(axis=1).ffill(axis=1).bfill(axis=1)


    # Anzahl Zeilen mit mindestens einem NaN in den Zeitspalten
    num_rows_with_nan = df[time_cols].isnull().any(axis=1).sum()
    print(f"Anzahl Zeilen mit mindestens einem NaN: {num_rows_with_nan}")

    # Optional: Anzahl Zeilen ohne NaN in den Zeitspalten
    num_rows_without_nan = (~df[time_cols].isnull().any(axis=1)).sum()
    print(f"Anzahl Zeilen ohne NaN: {num_rows_without_nan}")
  
    df = df.dropna(subset=time_cols)
    print(f"Anzahl Zeilen nach Entfernen der NaN-Zeilen: {len(df)}")


    X = df[time_cols].values
    y = df['class'].values - 1
    
    # Keep pixel_id 
    point_ids = df['pixel_id_num'].values

    # Convert string pixel_ids to integer-encoded labels
    point_ids_encoded, uniques = pd.factorize(point_ids)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print("X stats: min =", np.min(X), "max =", np.max(X))

    # Split with indices to track point_ids
    idx = np.arange(len(X))
    idx_trainval, idx_test = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
    idx_train, idx_val = train_test_split(idx_trainval, test_size=0.25, stratify=y[idx_trainval], random_state=42)

    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]

    def print_label_distribution(name, y):
        unique, counts = np.unique(y, return_counts=True)
        print(f"{name} Label Distribution:")
        for u, c in zip(unique, counts):
            print(f"  Klasse {u}: {c}")
        print()
        
    '''
    print_label_distribution("Train", y_train)
    print_label_distribution("Val", y_val)
    print_label_distribution("Test", y_test)
    '''


    point_ids_train = point_ids[idx_train]
    point_ids_val = point_ids[idx_val]
    point_ids_test = point_ids[idx_test]

    print("X_test.shape:", X_test.shape)
    print("y_test.shape:", y_test.shape)
    print("point_ids_test.shape:", point_ids_test.shape)



    def to_tensor(x): return torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    def to_label(y): return torch.tensor(y, dtype=torch.long)
    def to_id(x): return torch.tensor(np.array(x), dtype=torch.long)

    print(X_train.shape)
    print("to_tensor(X_train) shape:", to_tensor(X_train).shape)

    train_dataset = TensorDataset(to_tensor(X_train), to_label(y_train), to_id(point_ids_train))
    val_dataset   = TensorDataset(to_tensor(X_val), to_label(y_val), to_id(point_ids_val))
    test_dataset  = TensorDataset(to_tensor(X_test), to_label(y_test), to_id(point_ids_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size)

    # Berechne class_counts aus Trainingslabels
    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = np.zeros(len(unique))
    class_counts[unique] = counts 
    
    # Berechne class_weights
    total = class_counts.sum()
    num_classes = len(class_counts)
    class_weights = [total / (num_classes * c) for c in class_counts]
    
    # Mach daraus einen Tensor
    weights = torch.tensor(class_weights, dtype=torch.float32)


    return train_loader, val_loader, test_loader, X_train.shape[1], weights
