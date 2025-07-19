import torch
import numpy as np
class EarlyStopping:
    def __init__(self, path, patience, verbose=True, delta=0.001,
                 monitor='val_loss', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta 
        self.path = path
        self.trace_func = trace_func
        self.monitor = monitor

        if monitor == 'val_loss':
            self.mode = 'min'
            self.best_score = np.inf
        elif monitor == 'val_acc':
            self.mode = 'max'
            self.best_score = -np.inf
        else:
            raise ValueError("monitor must be 'val_loss' or 'val_acc'")
        print(self.mode)

    def __call__(self, current_val, model):
        score = current_val

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(current_val, model)
            print(f"Initial best_score set to {self.best_score:.6f}")
        else:
            print(f"Current score: {score:.6f}, Best score: {self.best_score:.6f}, Delta: {self.delta}")
            if self.mode == 'min':
                improved = score < self.best_score - self.delta
            else:  # mode == 'max'
                improved = score > self.best_score + self.delta


            if improved:
                self.best_score = score
                self.save_checkpoint(current_val, model)
                self.counter = 0
            else:
                self.counter += 1
                self.trace_func(f"EarlyStopping counter: {self.counter} / {self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, current_val, model):
        if self.verbose:
            self.trace_func(f"{self.monitor} improved. Saving model to {self.path}")
        torch.save(model.state_dict(), self.path)
