"""Exact-penalty-inspired PPO variant with fixed long-term penalty."""
from __future__ import annotations

from typing import Dict, Any, Tuple

import torch
import torch.nn as nn

from .base import BasePPOAgent


class EPOAgent(BasePPOAgent):
    """Implements a fixed penalty term (not full projected EPO)."""

    def __init__(self, policy_net: nn.Module, value_net: nn.Module, cost_value_net: nn.Module, config: Dict[str, Any], *, penalty: float):
        constraints = config.get("constraints", {})
        super().__init__(
            policy_net,
            value_net,
            cost_value_net,
            config,
            lambda_long_init=penalty,
            cost_limit=constraints.get("cost_limit", 1.0),
            update_long=False,
        )
        self.fixed_penalty = torch.tensor(penalty, dtype=torch.float32, device=self.device)

    def penalty_term(self, adv: torch.Tensor, cost_adv: torch.Tensor, B: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
        return adv - self.fixed_penalty * cost_adv
