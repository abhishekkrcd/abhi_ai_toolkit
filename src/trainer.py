# src/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

class Trainer:
    """Manages the training and evaluation cycle for the AI model."""
    def __init__(self, model: nn.Module, loss_fn: nn.Module, optimizer: optim.Optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train_one_epoch(self, X_train: torch.Tensor, y_train: torch.Tensor) -> torch.Tensor:
        """Performs one full epoch of training."""
        # Crucial step: Set the model to training mode
        self.model.train()

        # 1. Zero the gradients (MUST do this at the start!)
        self.optimizer.zero_grad()

        # 2. Forward pass: Get predictions
        predictions = self.model(X_train)

        # 3. Calculate Loss: Measure the difference between predictions and truth
        loss = self.loss_fn(predictions, y_train)

        # 4. Backward pass: Calculate loss gradient w.r.t model parameters
        loss.backward()

        # 5. Optimization step: Update model weights/biases
        self.optimizer.step()

        # Return the loss to signal successful training
        return loss