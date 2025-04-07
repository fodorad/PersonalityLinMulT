import torch


class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, base_lr):
        """Warmup scheduler to increase learning rate linearly during warmup.
        
        Args:
            optimizer: The optimizer being used.
            warmup_steps: Number of steps to warm up the learning rate.
            base_lr: Base learning rate to reach after warmup.
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.current_step = 0

    def step(self):
        if self.current_step < self.warmup_steps:
            lr = (self.current_step / float(self.warmup_steps)) * self.base_lr
        else:
            lr = self.base_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_step += 1

    def state_dict(self):
        return {'current_step': self.current_step}

    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']

    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class CosineAnnealingWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, max_lr, total_steps):
        """Scheduler that combines warmup with cosine annealing.
        
        Args:
            optimizer: The optimizer being used.
            warmup_steps: Number of steps to warm up the learning rate.
            max_lr: Maximum learning rate (after warmup).
            total_steps: Total number of training steps.
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.current_step = 0
        self.cosine_annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps - warmup_steps)
    
    def step(self, val_loss=None):
        if self.current_step < self.warmup_steps:
            lr = (self.current_step / float(self.warmup_steps)) * self.max_lr
        else:
            self.cosine_annealing_scheduler.step(val_loss)
            lr = self.optimizer.param_groups[0]['lr']

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_step += 1

    def state_dict(self):
        return {
            'current_step': self.current_step,
            'cosine_scheduler_state': self.cosine_annealing_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']
        self.cosine_annealing_scheduler.load_state_dict(state_dict['cosine_scheduler_state'])

    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]