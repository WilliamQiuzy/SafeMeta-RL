"""Validation network approximating the short-term feasibility function B(τ)."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class ValidationNetwork(nn.Module):
    """Binary classifier over concatenated n-step trajectories."""

    def __init__(self, trajectory_dim: int, hidden_sizes: Sequence[int], activation: nn.Module = nn.Tanh()):
        super().__init__()
        layers = []
        input_dim = trajectory_dim
        for size in hidden_sizes:
            layers.extend([nn.Linear(input_dim, size), activation])
            input_dim = size
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, rolled_states: torch.Tensor) -> torch.Tensor:
        """Return signed feasibility value; <=0 means safe."""
        return self.net(rolled_states)
