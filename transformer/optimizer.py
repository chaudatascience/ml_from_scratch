import torch
from torch import nn


class CustomAdam:
    """
        Follow section 5.3 Optimizer (page 7)
    """
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.step_num = 0
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def get_new_lr(self):
        return self.d_model ** (-0.5) * min(self.step_num**(-0.5), self.step_num * self.warmup_steps**(-1.5))

    def step(self):
        self.update_lr()
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def update_lr(self):
        self.step_num += 1

        for n, p in self.optimizer.param_groups:
            p["lr"] = self.get_new_lr()



