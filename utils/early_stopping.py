import torch
import numpy as np

class EarlyStopping:
    def __init__(self, 
                 patience=5, 
                 verbose=True, 
                 delta=0.01, 
                 path='best_model.pth', 
                 monitor='val_acc',  # 'val_loss' oder 'val_acc'
                 trace_func=print):
        
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.monitor = monitor
        self.mode = 'max' if monitor == 'val_acc' else 'min' # for val_loss
        self.best_val = -np.inf if self.mode == 'max' else np.inf 

    def __call__(self, current_val, model):
        score = current_val

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(current_val, model)
        elif (self.mode == 'min' and score > self.best_score - self.delta) or \
             (self.mode == 'max' and score < self.best_score + self.delta):
            self.counter += 1
            self.trace_func(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(current_val, model)
            self.counter = 0

    def save_checkpoint(self, current_val, model):
        if self.verbose:
            if self.mode == 'max':
                self.trace_func(f"{self.monitor} improved ({self.best_val:.4f} → {current_val:.4f}). Saving model.")
            else:
                self.trace_func(f"{self.monitor} decreased ({self.best_val:.4f} → {current_val:.4f}). Saving model.")
        torch.save(model.state_dict(), self.path)
        self.best_val = current_val
