"""On-policy rollout storage used for PPO-style updates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class Transition:
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    cost: torch.Tensor
    done: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    cost_value: torch.Tensor


class RolloutBuffer:
    """Simple list-backed storage for PPO rollouts."""

    def __init__(self):
        self._storage: List[Transition] = []

    def add(self, transition: Transition) -> None:
        self._storage.append(transition)

    def clear(self) -> None:
        self._storage.clear()

    def __len__(self) -> int:
        return len(self._storage)

    def stacked(self) -> Transition:
        """Stack transitions along first dimension for vectorized computations."""
        obs = torch.stack([t.obs for t in self._storage])
        action = torch.stack([t.action for t in self._storage])
        reward = torch.stack([t.reward for t in self._storage])
        cost = torch.stack([t.cost for t in self._storage])
        done = torch.stack([t.done for t in self._storage])
        log_prob = torch.stack([t.log_prob for t in self._storage])
        value = torch.stack([t.value for t in self._storage])
        cost_value = torch.stack([t.cost_value for t in self._storage])
        return Transition(obs=obs, action=action, reward=reward, cost=cost, done=done, log_prob=log_prob, value=value, cost_value=cost_value)
