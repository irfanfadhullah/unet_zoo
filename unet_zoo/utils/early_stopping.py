import torch
import torch.nn as nn
import copy

class EarlyStopping:
    """Early stopping to stop training when dice score doesn't improve for a given patience."""
    
    def __init__(self, patience=20, min_delta=0.001, restore_best_weights=True, verbose=True, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.mode = mode.lower()
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.stopped_epoch = 0
        self.early_stop = False  

        if self.mode not in ['min', 'max']:
            raise ValueError(f"Mode must be 'min' or 'max', got '{mode}'")
        
    def __call__(self, val_score, model, epoch):
        current_score = val_score
        
        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(model)
            if self.verbose:
                print(f'EarlyStopping: Initial best score set to {self.best_score:.6f}')
        elif self._is_improvement(current_score):
            improvement = self._calculate_improvement(current_score)
            if self.verbose:
                print(f'EarlyStopping: New best score {current_score:.6f} (improvement: {improvement:+.6f})')
            self.best_score = current_score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience} (current: {current_score:.6f}, best: {self.best_score:.6f})')
            
            if self.counter >= self.patience:
                self.stopped_epoch = epoch
                self.early_stop = True 
                if self.restore_best_weights and self.best_weights is not None:
                    self._restore_best_weights(model)
                    if self.verbose:
                        print(f'Restored best weights from epoch with score: {self.best_score:.6f}')
                return True
            
        return False
    
    def _is_improvement(self, current_score):
        """Check if current score is an improvement based on mode."""
        if self.mode == 'max':
            return current_score > self.best_score + self.min_delta
        else: 
            return current_score < self.best_score - self.min_delta
    
    def _calculate_improvement(self, current_score):
        """Calculate improvement amount based on mode."""
        if self.mode == 'max':
            return current_score - self.best_score
        else: 
            return self.best_score - current_score
    
    def save_checkpoint(self, model):
        """Save model weights when score improves."""
        if self.restore_best_weights:
            try:
                if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                    self.best_weights = copy.deepcopy(model.module.state_dict())
                else:
                    self.best_weights = copy.deepcopy(model.state_dict())
            except Exception as e:
                print(f"Warning: Failed to save checkpoint for early stopping: {e}")
                self.best_weights = None
    
    def _restore_best_weights(self, model):
        """Restore the best weights to the model."""
        try:
            if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                model.module.load_state_dict(self.best_weights)
            else:
                model.load_state_dict(self.best_weights)
        except Exception as e:
            print(f"Warning: Failed to restore best weights: {e}")
    
    def get_best_score(self):
        """Return the current best score."""
        return self.best_score
    
    def reset(self):
        """Reset the early stopping state."""
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.stopped_epoch = 0
        self.early_stop = False