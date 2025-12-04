"""PPO-based agent augmented with long/short-term safety constraints."""
from __future__ import annotations

from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn

from .base import AgentOutput, BasePPOAgent


class LSTCAgent(BasePPOAgent):
    """Connects policy/value networks and handles constraint multipliers."""

    def __init__(
        self,
        policy_net: nn.Module,
        value_net: nn.Module,
        cost_value_net: nn.Module,
        validation_net: Optional[nn.Module],
        config: Dict[str, Any],
        *,
        use_validation: bool = True,
        update_long: bool = True,
        update_short: bool = True,
        fixed_lambda_long: Optional[float] = None,
    ):
        constraints = config["constraints"]
        super().__init__(
            policy_net,
            value_net,
            cost_value_net,
            config,
            lambda_long_init=constraints["lambda_long_init"] if fixed_lambda_long is None else fixed_lambda_long,
            cost_limit=constraints["cost_limit"],
            update_long=update_long,
        )
        self.validation_net = validation_net
        self.use_validation = use_validation and validation_net is not None
        self.update_long = update_long
        self.update_short = update_short and self.use_validation

        self.lambda_short = torch.tensor(config["constraints"]["lambda_short_init"], dtype=torch.float32, device=self.device)
        self.lambda_short_lr = config["constraints"]["lambda_short_lr"]
        self.fixed_lambda_long = fixed_lambda_long

    def act(self, obs: torch.Tensor) -> AgentOutput:
        """Sample an action and gather baseline/value predictions."""
        return super().act(obs)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return log probabilities, entropy, and value estimates for PPO updates."""
        return super().evaluate_actions(obs, actions)

    def validation_scores(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Evaluate the validation network on a batch of trajectories."""
        if not self.use_validation or self.validation_net is None:
            raise RuntimeError("Validation network is disabled for this agent.")
        return self.validation_net(trajectories).squeeze(-1)

    @staticmethod
    def validation_loss(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Combine the feasible/infeasible hinge objectives from Eq. 17-19.

        labels == 1 => feasible trajectory (target <= 0)
        labels == 0 => infeasible trajectory (target > 0)
        """
        loss = torch.zeros(1, device=predictions.device)
        feasible_mask = labels == 1
        infeasible_mask = labels == 0
        if feasible_mask.any():
            loss = loss + torch.relu(predictions[feasible_mask]).mean()
        if infeasible_mask.any():
            loss = loss + torch.relu(-predictions[infeasible_mask]).mean()
        return loss

    @staticmethod
    def short_term_violation(predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """Estimate violation magnitude for updating λ_s."""
        feasible_mask = labels == 1
        if not feasible_mask.any():
            return 0.0
        violation = torch.relu(predictions[feasible_mask]).mean()
        return float(violation.item())

    def update_lagrange_multipliers(self, avg_cost: float, short_term_violation: float) -> None:
        """Gradient-ascent style updates following Alg. 1 in the paper."""
        super().update_lagrange_multipliers(avg_cost, short_term_violation)

        if self.update_short and short_term_violation > 0.0:
            self.lambda_short += self.lambda_short_lr * short_term_violation
            self.lambda_short = torch.clamp(self.lambda_short, min=0.0)

    def penalty_term(self, adv: torch.Tensor, cost_adv: torch.Tensor, B: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
        short_constraint = self.lambda_short * ratio * B
        return adv - self.lambda_long * cost_adv - short_constraint
