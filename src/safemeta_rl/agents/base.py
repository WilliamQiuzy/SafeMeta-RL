"""Common PPO-style agent utilities shared by baselines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn


@dataclass
class AgentOutput:
    action: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    cost_value: torch.Tensor


class BasePPOAgent:
    """Minimal PPO agent with optional long-term cost multiplier."""

    def __init__(
        self,
        policy_net: nn.Module,
        value_net: nn.Module,
        cost_value_net: nn.Module,
        config: Dict[str, Any],
        *,
        lambda_long_init: float = 0.0,
        cost_limit: float = 1.0,
        update_long: bool = False,
    ):
        self.policy_net = policy_net
        self.value_net = value_net
        self.cost_value_net = cost_value_net
        self.device = next(policy_net.parameters()).device

        self.lambda_long = torch.tensor(lambda_long_init, dtype=torch.float32, device=self.device)
        self.lambda_long_lr = config.get("constraints", {}).get("lambda_long_lr", 0.0)
        self.cost_limit = cost_limit
        self.update_long = update_long

    def act(self, obs: torch.Tensor) -> AgentOutput:
        dist = self.policy_net(obs)
        action = dist.sample()
        log_prob_raw = dist.log_prob(action)
        # D.Independent already sums over action dimension, so log_prob_raw is [batch_size]
        # We just need to add the keepdim dimension, NOT sum again
        if log_prob_raw.dim() == 1:
            log_prob = log_prob_raw.unsqueeze(-1)
        else:
            log_prob = log_prob_raw.sum(dim=-1, keepdim=True)
        value = self.value_net(obs)
        cost_value = self.cost_value_net(obs)
        return AgentOutput(action=action, log_prob=log_prob, value=value, cost_value=cost_value)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.policy_net(obs)
        log_prob_raw = dist.log_prob(actions)
        entropy_raw = dist.entropy()
        # D.Independent already sums over action dimension, so log_prob_raw is [batch_size]
        # We just need to add the keepdim dimension, NOT sum again
        if log_prob_raw.dim() == 1:
            log_prob = log_prob_raw.unsqueeze(-1)
            entropy = entropy_raw.unsqueeze(-1)
        else:
            # Fallback for non-Independent distributions
            log_prob = log_prob_raw.sum(dim=-1, keepdim=True)
            entropy = entropy_raw.sum(dim=-1, keepdim=True)
        values = self.value_net(obs)
        cost_values = self.cost_value_net(obs)
        return log_prob, entropy, values, cost_values

    @staticmethod
    def compute_advantages(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, last_value: torch.Tensor, gamma: float, gae_lambda: float):
        rewards_flat = rewards.view(-1)
        values_flat = values.view(-1)
        dones_flat = dones.view(-1)
        last_value_flat = last_value.view(1)

        T = rewards_flat.size(0)
        advantages = torch.zeros_like(rewards_flat)
        returns = torch.zeros_like(rewards_flat)
        values_ext = torch.cat([values_flat, last_value_flat], dim=0)
        gae = torch.zeros(1, device=rewards.device, dtype=rewards.dtype)
        for step in reversed(range(T)):
            delta = rewards_flat[step] + gamma * values_ext[step + 1] * (1.0 - dones_flat[step]) - values_ext[step]
            gae = delta + gamma * gae_lambda * (1.0 - dones_flat[step]) * gae
            advantages[step] = gae
            returns[step] = gae + values_ext[step]
        return advantages.unsqueeze(-1), returns.unsqueeze(-1)

    def penalty_term(self, adv: torch.Tensor, cost_adv: torch.Tensor, B: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
        """Override to implement algorithm-specific penalty shaping."""
        return adv

    def update_lagrange_multipliers(self, avg_cost: float, short_term_violation: float = 0.0) -> None:
        if self.update_long and avg_cost > self.cost_limit:
            self.lambda_long += self.lambda_long_lr * (avg_cost - self.cost_limit)
            self.lambda_long = torch.clamp(self.lambda_long, min=0.0)

    def update_long_from_gap(self, cost_gap: float) -> None:
        """Optional per-batch λ update; default no-op."""
        return
