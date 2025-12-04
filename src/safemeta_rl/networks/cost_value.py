"""Cost-value network mirroring reward value architecture."""
from __future__ import annotations

from typing import Sequence

import torch.nn as nn

from .value import ValueNetwork


def build_cost_value_network(obs_dim: int, hidden_sizes: Sequence[int], activation: nn.Module = nn.Tanh()) -> ValueNetwork:
    return ValueNetwork(obs_dim=obs_dim, hidden_sizes=hidden_sizes, activation=activation)
