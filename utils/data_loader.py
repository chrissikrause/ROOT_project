import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_and_preprocess_data(path, batch_size=32):
    df = pd.read_csv(path)
    df.replace(-2147483648.0, np.nan, inplace=True)
    df.interpolate(method='linear', axis=1, inplace=True)
    df.bfill(axis=1, inplace=True)
    df.ffill(axis=1, inplace=True)

    time_cols = [col for col in df.columns if col.startswith('di_t')]
    df = df[df['class'].isin([1, 2, 3])]
    df[time_cols] = df[time_cols].interpolate(axis=1).ffill(axis=1).bfill(axis=1)

    X = df[time_cols].values
    y = df['class'].values - 1

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42)

    def to_tensor(x): return torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    def to_label(y): return torch.tensor(y, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(to_tensor(X_train), to_label(y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(to_tensor(X_val), to_label(y_val)), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(to_tensor(X_test), to_label(y_test)), batch_size=batch_size)

    return train_loader, val_loader, test_loader, X_train.shape[1]
