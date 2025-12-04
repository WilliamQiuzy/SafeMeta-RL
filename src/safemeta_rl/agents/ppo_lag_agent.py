"""PPO-Lagrangian style agent with adaptive long-term multiplier."""
from __future__ import annotations

from typing import Dict, Any, Tuple

import torch
import torch.nn as nn

from .base import BasePPOAgent


class PPOLagAgent(BasePPOAgent):
    """Adds long-term cost penalty with lambda updated on average cost."""

    def __init__(self, policy_net: nn.Module, value_net: nn.Module, cost_value_net: nn.Module, config: Dict[str, Any]):
        constraints = config.get("constraints", {})
        super().__init__(
            policy_net,
            value_net,
            cost_value_net,
            config,
            lambda_long_init=constraints.get("lambda_long_init", 0.0),
            cost_limit=constraints.get("cost_limit", 1.0),
            update_long=True,
        )

    def penalty_term(self, adv: torch.Tensor, cost_adv: torch.Tensor, B: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
        return adv - self.lambda_long * cost_adv

    def update_long_from_gap(self, cost_gap: float) -> None:
        if cost_gap > 0.0 and self.update_long:
            self.lambda_long += self.lambda_long_lr * cost_gap
            self.lambda_long = torch.clamp(self.lambda_long, min=0.0)
