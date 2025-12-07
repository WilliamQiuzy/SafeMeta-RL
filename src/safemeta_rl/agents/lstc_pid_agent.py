"""LSTC agent with PID-style Lagrange multiplier updates (Stooke et al., 2020)."""
from __future__ import annotations

from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn

from .base import AgentOutput, BasePPOAgent


class PIDController:
    """PID controller for Lagrange multiplier updates.

    Instead of pure integral control (traditional Lagrange update),
    this uses PID control for faster convergence and less oscillation.

    Reference: Stooke et al., "Responsive Safety in Reinforcement Learning
    by PID Lagrangian Methods", ICML 2020.

    Key improvements over naive implementation:
    - Anti-windup: integral is clamped based on lambda_max to prevent windup
    - Derivative on error change (not raw value) for stability
    - Sliding window for derivative smoothing to reduce noise
    - Conservative default gains (PI first, D optional)
    """

    def __init__(
        self,
        kp: float = 0.05,
        ki: float = 0.01,
        kd: float = 0.0,  # Start with PI only, add D carefully
        setpoint: float = 0.0,
        lambda_max: float = 2.0,  # Conservative upper bound
        smoothing: float = 0.5,  # Lower = faster response (0.5-0.7 recommended)
        derivative_window: int = 5,  # Sliding window for derivative
    ):
        """
        Args:
            kp: Proportional gain - immediate response to current error
            ki: Integral gain - accumulates historical error
            kd: Derivative gain - responds to error trend (keep small or 0)
            setpoint: Target value (e.g., cost limit)
            lambda_max: Maximum allowed lambda value (scale with cost_limit)
            smoothing: Exponential smoothing factor (0.5-0.7 for faster response)
            derivative_window: Window size for derivative smoothing
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.lambda_max = lambda_max
        self.smoothing = smoothing
        self.derivative_window = derivative_window

        # PID state
        self.integral = 0.0
        self.prev_error = 0.0

        # Sliding window for derivative (reduces noise)
        self.error_history: list = []

        # Smoothed values for stability
        self.smoothed_error = 0.0

    def update(self, current_value: float) -> float:
        """Compute new lambda value using PID control.

        Args:
            current_value: Current measured value (e.g., episode cost or violation)

        Returns:
            New lambda value
        """
        # Error: how much we exceed the setpoint
        error = current_value - self.setpoint

        # Smooth the error for P term
        self.smoothed_error = self.smoothing * self.smoothed_error + (1 - self.smoothing) * error

        # Derivative: use sliding window to reduce noise
        # Only penalize when error is increasing (Stooke's insight)
        self.error_history.append(error)
        if len(self.error_history) > self.derivative_window:
            self.error_history.pop(0)

        if len(self.error_history) >= 2:
            # Average derivative over window
            derivatives = [
                max(self.error_history[i] - self.error_history[i - 1], 0.0)
                for i in range(1, len(self.error_history))
            ]
            derivative = sum(derivatives) / len(derivatives)
        else:
            derivative = 0.0

        # Integral with anti-windup and decay
        # When error < 0 (cost below limit), decay integral to allow lambda recovery
        if error < 0:
            # Decay integral when cost is below limit (allows lambda to decrease)
            decay_factor = 0.95  # Slow decay
            self.integral = self.integral * decay_factor + error
        else:
            self.integral = self.integral + error

        # Clamp integral with anti-windup
        if self.ki > 0:
            integral_max = self.lambda_max / self.ki
            self.integral = max(min(self.integral, integral_max), 0.0)  # Keep [0, integral_max]
        else:
            self.integral = max(self.integral, 0.0)

        # PID output
        p_term = self.kp * self.smoothed_error
        i_term = self.ki * self.integral
        d_term = self.kd * derivative

        lambda_new = max(p_term + i_term + d_term, 0.0)
        lambda_new = min(lambda_new, self.lambda_max)

        # Update state for next iteration
        self.prev_error = error

        return lambda_new

    def reset(self):
        """Reset PID state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.error_history = []
        self.smoothed_error = 0.0

    def get_state(self) -> dict:
        """Return internal state for debugging."""
        return {
            "integral": self.integral,
            "smoothed_error": self.smoothed_error,
            "error_history_len": len(self.error_history),
        }


class LSTCPIDAgent(BasePPOAgent):
    """LSTC Agent with PID-style Lagrange multiplier updates.

    Key insight from Stooke et al.:
    - Traditional Lagrange update is pure integral control (slow, oscillates)
    - PID adds proportional (immediate response) and derivative (trend prediction)
    - Results in faster constraint satisfaction with less overshoot
    """

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
    ):
        constraints = config["constraints"]
        super().__init__(
            policy_net,
            value_net,
            cost_value_net,
            config,
            lambda_long_init=constraints["lambda_long_init"],
            cost_limit=constraints["cost_limit"],
            update_long=update_long,
        )
        self.validation_net = validation_net
        self.use_validation = use_validation and validation_net is not None
        self.update_long = update_long
        self.update_short = update_short and self.use_validation

        # PID parameters from config
        # Default gains are conservative: PI first, D optional
        pid_config = config.get("pid", {})
        cost_limit = constraints["cost_limit"]

        # Long-term cost constraint PID controller
        # Scale lambda_max with cost_limit for reasonable bounds
        self.pid_long = PIDController(
            kp=pid_config.get("kp_long", 0.05),  # Conservative P
            ki=pid_config.get("ki_long", 0.01),  # Small I
            kd=pid_config.get("kd_long", 0.0),   # Start without D
            setpoint=cost_limit,
            lambda_max=pid_config.get("lambda_long_max", 2.0 * cost_limit),  # Scale with cost
            smoothing=pid_config.get("smoothing", 0.5),  # Faster response
            derivative_window=pid_config.get("derivative_window", 5),
        )

        # Short-term violation constraint PID controller
        # Target is zero violation, so gains can be similar to long-term
        self.pid_short = PIDController(
            kp=pid_config.get("kp_short", 0.1),  # Slightly higher for faster response
            ki=pid_config.get("ki_short", 0.02),
            kd=pid_config.get("kd_short", 0.0),  # Start without D
            setpoint=0.0,  # Target: zero violation
            lambda_max=pid_config.get("lambda_short_max", 2.0),
            smoothing=pid_config.get("smoothing", 0.5),
            derivative_window=pid_config.get("derivative_window", 5),
        )

        # Initialize lambda values
        self.lambda_long = torch.tensor(constraints["lambda_long_init"], dtype=torch.float32, device=self.device)
        self.lambda_short = torch.tensor(constraints["lambda_short_init"], dtype=torch.float32, device=self.device)

    def act(self, obs: torch.Tensor) -> AgentOutput:
        return super().act(obs)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return super().evaluate_actions(obs, actions)

    def validation_scores(self, trajectories: torch.Tensor) -> torch.Tensor:
        if not self.use_validation or self.validation_net is None:
            raise RuntimeError("Validation network is disabled for this agent.")
        return self.validation_net(trajectories).squeeze(-1)

    @staticmethod
    def validation_loss(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Hinge loss for validation network (Eq. 17-19)."""
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
        """Estimate violation magnitude for updating lambda_s."""
        feasible_mask = labels == 1
        if not feasible_mask.any():
            return 0.0
        violation = torch.relu(predictions[feasible_mask]).mean()
        return float(violation.item())

    def update_lagrange_multipliers(self, avg_cost: float, short_term_violation: float) -> None:
        """PID-style updates for both lambda_l and lambda_s.

        Unlike traditional gradient-ascent:
        - Faster response to constraint violations (P term)
        - Accumulated error correction (I term)
        - Anticipates and dampens overshoot (D term)
        """
        # Update long-term lambda using PID
        if self.update_long:
            new_lambda_long = self.pid_long.update(avg_cost)
            self.lambda_long = torch.tensor(new_lambda_long, dtype=torch.float32, device=self.device)

        # Update short-term lambda using PID
        if self.update_short:
            new_lambda_short = self.pid_short.update(short_term_violation)
            self.lambda_short = torch.tensor(new_lambda_short, dtype=torch.float32, device=self.device)

    def penalty_term(self, adv: torch.Tensor, cost_adv: torch.Tensor, B: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
        """Compute safe advantage: A - lambda_l * A_c - lambda_s * r(theta) * B(tau_n)"""
        short_constraint = self.lambda_short * ratio * B
        return adv - self.lambda_long * cost_adv - short_constraint

    def get_pid_stats(self) -> Dict[str, float]:
        """Return PID internal states for logging/debugging."""
        return {
            "pid_long_integral": self.pid_long.integral,
            "pid_long_smoothed_error": self.pid_long.smoothed_error,
            "pid_short_integral": self.pid_short.integral,
            "pid_short_smoothed_error": self.pid_short.smoothed_error,
        }
