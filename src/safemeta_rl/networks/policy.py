"""Policy network producing Gaussian distributions over continuous actions."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.distributions as D


class GaussianPolicy(nn.Module):
    """Two-layer MLP with state-dependent mean and learned log-std."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Sequence[int], activation: nn.Module = nn.Tanh()):
        super().__init__()
        layers = []
        input_dim = obs_dim
        for size in hidden_sizes:
            layers.extend([nn.Linear(input_dim, size), activation])
            input_dim = size
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(input_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> D.Independent:
        x = self.backbone(obs)
        mean = self.mean_head(x)
        std = torch.exp(self.log_std).clamp(min=1e-5)
        dist = D.Normal(mean, std)
        return D.Independent(dist, 1)
