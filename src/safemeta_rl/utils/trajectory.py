"""Utilities for constructing n-step trajectories for the validation network."""
from __future__ import annotations

from typing import List, Tuple

import torch


def sliding_trajectories(states: List[torch.Tensor], feasible_flags: List[float], horizon: int, limit: int) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Build flattened n-step trajectories and feasibility labels.

    Args:
        states: list of tensors shaped (obs_dim,)
        feasible_flags: list with same length as `states`, where 1 denotes feasible
        horizon: number of consecutive states per trajectory
        limit: maximum number of starting indices (usually rollout length)

    Returns:
        trajectories: Tensor [N, obs_dim * horizon]
        labels: Tensor [N] with 1 for feasible, 0 otherwise
        indices: list of starting indices that correspond to rollout steps
    """
    if len(states) < horizon:
        raise ValueError("Insufficient states to build trajectories")

    obs_dim = states[0].numel()
    max_start = min(limit, len(states) - horizon + 1)
    traj_tensors = []
    labels = []
    indices: List[int] = []
    for start in range(max_start):
        end = start + horizon
        if end > len(states):
            break
        window = torch.stack(states[start:end], dim=0)
        traj_tensors.append(window.reshape(obs_dim * horizon))
        window_flags = feasible_flags[start:end]
        label = 1.0 if all(flag >= 0.5 for flag in window_flags) else 0.0
        labels.append(label)
        indices.append(start)

    if not traj_tensors:
        raise ValueError("No valid trajectories constructed")

    trajectories = torch.stack(traj_tensors, dim=0)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    return trajectories, labels_tensor, indices
