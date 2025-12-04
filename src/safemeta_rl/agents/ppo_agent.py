"""Vanilla PPO agent (no safety penalties)."""
from __future__ import annotations

from typing import Dict, Any, Tuple

import torch
import torch.nn as nn

from .base import BasePPOAgent


class PPOAgent(BasePPOAgent):
    """Standard PPO; penalties are no-ops."""

    def __init__(self, policy_net: nn.Module, value_net: nn.Module, cost_value_net: nn.Module, config: Dict[str, Any]):
        super().__init__(
            policy_net,
            value_net,
            cost_value_net,
            config,
            lambda_long_init=0.0,
            cost_limit=config.get("constraints", {}).get("cost_limit", 1.0),
            update_long=False,
        )

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return super().act(obs)

    def penalty_term(self, adv: torch.Tensor, cost_adv: torch.Tensor, B: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
        return adv
