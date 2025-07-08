import torch.optim as optim
import math

class DiceScheduler:
    """Learning Rate Scheduler that monitors dice score instead of loss."""
    
    def __init__(self, optimizer: optim.Optimizer, patience=8, factor=0.5, min_lr=1e-7, min_delta=0.001, verbose=True, mode='max'):
        self.optimizer = optimizer
        self.patience = int(patience)
        self.factor = float(factor)
        self.min_lr = float(min_lr) 
        self.min_delta = float(min_delta) 
        self.verbose = verbose
        self.mode = mode.lower()
        self.best_score = None
        self.counter = 0
        self.num_bad_epochs = 0
        self.last_lr_reduction = 0
        
        if self.mode not in ['min', 'max']:
            raise ValueError(f"Mode must be 'min' or 'max', got '{mode}'")
        
    def step(self, val_score, epoch=None):
        current_score = val_score
        
        if self.best_score is None:
            self.best_score = current_score
            if self.verbose:
                print(f'DiceScheduler: Initial best score set to {self.best_score:.6f}')
        elif not self._is_improvement(current_score):
            self.counter += 1
            self.num_bad_epochs += 1
            
            if self.verbose and self.counter % 2 == 0: 
                print(f'DiceScheduler: No improvement for {self.counter} epochs (current: {current_score:.6f}, best: {self.best_score:.6f})')
            
            if self.counter >= self.patience:
                old_lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
                self._reduce_lr()
                new_lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
                
                if old_lrs != new_lrs:
                    self.last_lr_reduction = epoch if epoch is not None else self.last_lr_reduction + 1
                    if self.verbose:
                        print(f'DiceScheduler: Learning rate reduced at epoch {epoch}')
                
                self.counter = 0  
        else:
            improvement = self._calculate_improvement(current_score)
            if self.verbose and improvement > self.min_delta:
                print(f'DiceScheduler: New best score {current_score:.6f} (improvement: {improvement:+.6f})')
            self.best_score = current_score
            self.counter = 0
            self.num_bad_epochs = 0
    
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
    
    def _reduce_lr(self):
        """Reduce learning rate by factor."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lr)
            
            if new_lr < old_lr: 
                param_group['lr'] = new_lr
                if self.verbose:
                    print(f'Reducing learning rate of group {i} from {old_lr:.6f} to {new_lr:.6f}')
            elif self.verbose and old_lr <= self.min_lr:
                print(f'Learning rate {old_lr:.6f} already at minimum ({self.min_lr:.6f})')


    def get_last_lr(self):
        """Get current learning rates."""
        return [float(param_group['lr']) for param_group in self.optimizer.param_groups]
    
    def state_dict(self):
        """Return scheduler state."""
        return {
            'best_score': self.best_score,
            'counter': self.counter,
            'num_bad_epochs': self.num_bad_epochs,
            'last_lr_reduction': self.last_lr_reduction,
            'mode': self.mode
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state."""
        self.best_score = state_dict.get('best_score')
        self.counter = state_dict.get('counter', 0)
        self.num_bad_epochs = state_dict.get('num_bad_epochs', 0)
        self.last_lr_reduction = state_dict.get('last_lr_reduction', 0)
