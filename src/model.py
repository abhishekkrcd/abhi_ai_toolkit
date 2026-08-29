# src/model.py 
import torch
import torch.nn as nn
import torch.optim as optim

class SnoopyAIModel(nn.Module): # <--- Renamed here
    """A minimal, fully connected neural network for classification/regression, named Snoopy."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        # Layer 1: Input to hidden
        self.fc1 = nn.Linear(input_dim, 64)
        # Layer 2: Hidden to output
        self.fc2 = nn.Linear(64, output_dim)
        # Simple activation function like ReLU
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass logic for the model."""
        # The internal logic remains the same
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x