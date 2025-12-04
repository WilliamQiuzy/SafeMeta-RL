"""State-value function estimator."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """MLP regressor used for both reward value and auxiliary critics."""

    def __init__(self, obs_dim: int, hidden_sizes: Sequence[int], activation: nn.Module = nn.Tanh()):
        super().__init__()
        layers = []
        input_dim = obs_dim
        for size in hidden_sizes:
            layers.extend([nn.Linear(input_dim, size), activation])
            input_dim = size
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
