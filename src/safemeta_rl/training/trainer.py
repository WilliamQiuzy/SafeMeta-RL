"""High-level training loop for the LSTC agent."""
from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from safemeta_rl.agents.base import AgentOutput
from safemeta_rl.agents.lstc_agent import LSTCAgent
from safemeta_rl.agents.lstc_pid_agent import LSTCPIDAgent
from safemeta_rl.agents.ppo_agent import PPOAgent
from safemeta_rl.agents.ppo_lag_agent import PPOLagAgent
from safemeta_rl.agents.epo_agent import EPOAgent
from safemeta_rl.envs.metadrive_wrapper import LSTCMetaDriveEnv
from safemeta_rl.networks.policy import GaussianPolicy
from safemeta_rl.networks.value import ValueNetwork
from safemeta_rl.networks.cost_value import build_cost_value_network
from safemeta_rl.networks.validation import ValidationNetwork
from safemeta_rl.utils.logger import CSVLogger
from safemeta_rl.utils.replay_buffer import RolloutBuffer, Transition
from safemeta_rl.utils.trajectory import sliding_trajectories


class Trainer:
    """Coordinates environment interaction, optimization, and evaluation."""

    def __init__(self, config: Dict[str, Any], output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = CSVLogger(log_dir=self.output_dir)
        tensorboard_enabled = config.get("logging", {}).get("tensorboard", True)
        self.writer: Optional[SummaryWriter] = SummaryWriter(log_dir=self.output_dir / "tb") if tensorboard_enabled else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        env_config = config.get("environment", {})
        self.env_config = env_config  # Save for restoring after evaluation
        self.env = LSTCMetaDriveEnv(env_config)
        self.eval_env_config = self._build_eval_env_config(env_config, config.get("evaluation_environment"))

        self.algorithm = config.get("algorithm", "lstc").lower()
        self.algorithm_params = config.get("algorithm_params", {})
        self.use_short_constraint = self.algorithm in ("lstc", "lstc_pid")
        self.use_long_constraint = self.algorithm in ("lstc", "lstc_pid", "ppo_lag", "epo")
        sample_obs = self.env.reset()
        self._pending_initial_obs = sample_obs
        self._pending_initial_feasible = self.env.current_feasible()
        obs_dim = sample_obs.shape[0]
        action_dim = env_config.get("action_dim", 2)
        hidden = [64, 64]
        self.policy = GaussianPolicy(obs_dim, action_dim, hidden).to(self.device)
        self.value_net = ValueNetwork(obs_dim, hidden).to(self.device)
        self.cost_value_net = build_cost_value_network(obs_dim, hidden).to(self.device)
        self.validation_net = None
        if self.use_short_constraint:
            traj_dim = obs_dim * config["constraints"]["short_term_horizon"]
            self.validation_net = ValidationNetwork(traj_dim, hidden).to(self.device)

        if self.algorithm == "ppo":
            self.agent = PPOAgent(self.policy, self.value_net, self.cost_value_net, config)
        elif self.algorithm == "ppo_lag":
            self.agent = PPOLagAgent(self.policy, self.value_net, self.cost_value_net, config)
        elif self.algorithm == "epo":
            penalty = self.algorithm_params.get("epo_penalty", config["constraints"]["lambda_long_init"])
            self.agent = EPOAgent(self.policy, self.value_net, self.cost_value_net, config, penalty=penalty)
        elif self.algorithm == "lstc_pid":
            # LSTC with PID-style Lagrange multiplier updates (Stooke et al., 2020)
            self.agent = LSTCPIDAgent(
                self.policy,
                self.value_net,
                self.cost_value_net,
                self.validation_net,
                config,
                use_validation=self.use_short_constraint,
                update_long=self.use_long_constraint,
                update_short=self.use_short_constraint,
            )
        else:
            self.agent = LSTCAgent(
                self.policy,
                self.value_net,
                self.cost_value_net,
                self.validation_net,
                config,
                use_validation=self.use_short_constraint,
                update_long=self.use_long_constraint,
                update_short=self.use_short_constraint,
                fixed_lambda_long=None,
            )

        optim_cfg = config.get("optimizer", {})
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=optim_cfg.get("policy_lr", 3e-4))
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=optim_cfg.get("value_lr", 3e-4))
        self.cost_value_optimizer = optim.Adam(self.cost_value_net.parameters(), lr=optim_cfg.get("cost_value_lr", 3e-4))
        self.validation_optimizer = (
            optim.Adam(self.validation_net.parameters(), lr=optim_cfg.get("validation_lr", 3e-4))
            if self.validation_net is not None
            else None
        )

        self.rollout_buffer = RolloutBuffer()
        self.total_env_steps = 0
        self.checkpoint_dir = self.output_dir / config.get("checkpoint_dir", "checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_eval_metric = float("-inf")
        self.best_checkpoint_path: Optional[Path] = None
        self.eval_env: Optional[LSTCMetaDriveEnv] = None
        self._resume_checkpoint_step: Optional[int] = None  # Track resume point to protect old checkpoints

    def train(self) -> None:
        training_cfg = self.config.get("training", {})
        logging_cfg = self.config.get("logging", {})
        total_steps = int(training_cfg.get("total_env_steps", 1_000_000))
        rollout_steps = int(training_cfg.get("rollout_steps", 2048))
        eval_interval = int(logging_cfg.get("eval_interval_steps", 200_000))
        checkpoint_interval = int(logging_cfg.get("checkpoint_interval_steps", 500_000))
        eval_episodes = logging_cfg.get("eval_episodes", 20)

        if self.total_env_steps >= total_steps:
            print("Target environment steps already reached; skipping training.")
            return

        if self._pending_initial_obs is not None:
            obs = self._pending_initial_obs
            current_feasible = self._pending_initial_feasible
            self._pending_initial_obs = None
            self._pending_initial_feasible = None
        else:
            obs = self.env.reset()
            current_feasible = self.env.current_feasible()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        next_eval = ((self.total_env_steps // eval_interval) + 1) * eval_interval if eval_interval > 0 else total_steps
        next_checkpoint = ((self.total_env_steps // checkpoint_interval) + 1) * checkpoint_interval if checkpoint_interval > 0 else total_steps
        episode_reward = 0.0
        episode_cost = 0.0
        episode_length = 0
        successes = 0
        completed_rewards: List[float] = []
        completed_costs: List[float] = []
        completed_lengths: List[int] = []

        try:
            while self.total_env_steps < total_steps:
                self.rollout_buffer.clear()
                states_window: List[torch.Tensor] = []
                state_flags: List[float] = []
                state_costs: List[float] = []
                reward_component_sums: Dict[str, float] = {}

                step_iterator = tqdm(range(rollout_steps), desc=f"Rollout @ {self.total_env_steps:,} steps")
                for _ in step_iterator:
                    states_window.append(obs_tensor.detach().cpu())
                    state_flags.append(1.0 if current_feasible else 0.0)
                    with torch.no_grad():
                        agent_output = self.agent.act(obs_tensor.unsqueeze(0))

                    action = agent_output.action.squeeze(0)
                    env_action = action.detach().cpu().numpy()
                    step_result = self.env.step(env_action)
                    reward = float(step_result.reward)
                    cost = float(step_result.cost)
                    done = bool(step_result.terminated or step_result.truncated)
                    next_obs_tensor = torch.tensor(step_result.obs, dtype=torch.float32, device=self.device)
                    state_costs.append(cost)

                    transition = Transition(
                        obs=obs_tensor.detach(),
                        action=action.detach(),
                        reward=torch.tensor(reward, dtype=torch.float32, device=self.device),
                        cost=torch.tensor(cost, dtype=torch.float32, device=self.device),
                        done=torch.tensor(float(done), dtype=torch.float32, device=self.device),
                        log_prob=agent_output.log_prob.squeeze(0).detach(),
                        value=agent_output.value.squeeze(0).detach(),
                        cost_value=agent_output.cost_value.squeeze(0).detach(),
                    )
                    self.rollout_buffer.add(transition)

                    for name, value in (step_result.reward_components or {}).items():
                        reward_component_sums[name] = reward_component_sums.get(name, 0.0) + float(value)

                    episode_reward += reward
                    episode_cost += cost
                    episode_length += 1
                    self.total_env_steps += 1
                    step_iterator.set_postfix(
                        reward=f"{episode_reward:.2f}",
                        cost=f"{episode_cost:.2f}",
                        steps=self.total_env_steps,
                    )
                    if step_result.info.get("arrive_dest"):
                        successes += 1

                    if done:
                        print(
                            f"[Episode done] steps={self.total_env_steps} episode_reward={episode_reward:.2f} "
                            f"episode_cost={episode_cost:.2f} length={episode_length}"
                        )
                        completed_rewards.append(episode_reward)
                        completed_costs.append(episode_cost)
                        completed_lengths.append(episode_length)
                        episode_reward = 0.0
                        episode_cost = 0.0
                        episode_length = 0
                        obs = self.env.reset()
                        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                        current_feasible = self.env.current_feasible()
                    else:
                        obs_tensor = next_obs_tensor
                        current_feasible = step_result.feasible

                    if self.total_env_steps >= total_steps:
                        break

                states_window.append(obs_tensor.detach().cpu())
                state_flags.append(1.0 if current_feasible else 0.0)
                state_costs.append(0.0)

                with torch.no_grad():
                    last_value = self.value_net(obs_tensor.unsqueeze(0)).detach()
                    last_cost_value = self.cost_value_net(obs_tensor.unsqueeze(0)).detach()

                train_stats = self._optimize_model(last_value, last_cost_value, states_window, state_flags, state_costs)

                avg_episode_reward = sum(completed_rewards) / len(completed_rewards) if completed_rewards else 0.0
                avg_episode_cost = sum(completed_costs) / len(completed_costs) if completed_costs else 0.0
                avg_episode_length = sum(completed_lengths) / len(completed_lengths) if completed_lengths else 0
                success_rate = successes / max(1, len(completed_rewards)) if completed_rewards else 0.0

                rollout_data = self.rollout_buffer.stacked()
                rollout_mean_reward = float(rollout_data.reward.mean().item())
                rollout_mean_cost = float(rollout_data.cost.mean().item())
                rollout_total_reward = float(rollout_data.reward.sum().item())
                rollout_total_cost = float(rollout_data.cost.sum().item())

                log_payload = {
                    "env_steps": self.total_env_steps,
                    "algorithm": self.algorithm,
                    "avg_episode_reward": avg_episode_reward,
                    "avg_episode_cost": avg_episode_cost,
                    "avg_episode_length": avg_episode_length,
                    "success_rate": success_rate,
                    "rollout_mean_reward": rollout_mean_reward,
                    "rollout_mean_cost": rollout_mean_cost,
                    "rollout_total_reward": rollout_total_reward,
                    "rollout_total_cost": rollout_total_cost,
                    "policy_loss": train_stats["policy_loss"],
                    "value_loss": train_stats["value_loss"],
                    "cost_value_loss": train_stats["cost_value_loss"],
                    "validation_loss": train_stats.get("validation_loss", 0.0),
                    "lambda_long": float(self.agent.lambda_long.item()) if hasattr(self.agent, "lambda_long") else 0.0,
                    "lambda_short": float(getattr(self.agent, "lambda_short", torch.tensor(0.0)).item()),
                    "num_episodes_completed": len(completed_rewards),
                    "current_episode_length": episode_length,
                }
                # Log PID internal states if using LSTC-PID
                if hasattr(self.agent, "get_pid_stats"):
                    pid_stats = self.agent.get_pid_stats()
                    log_payload.update(pid_stats)
                for name, total_value in reward_component_sums.items():
                    log_payload[f"avg_{name}"] = total_value / max(1, len(states_window))
                self.logger.log(log_payload)
                self._log_tensorboard_metrics(prefix="train", metrics=log_payload, step=self.total_env_steps)

                completed_rewards.clear()
                completed_costs.clear()
                completed_lengths.clear()
                successes = 0

                if self.total_env_steps >= next_eval:
                    eval_metrics = self.evaluate(episodes=eval_episodes)
                    print(f"[Eval @ {self.total_env_steps} steps] {eval_metrics}")
                    self._log_tensorboard_metrics(prefix="eval", metrics=eval_metrics, step=self.total_env_steps)
                    success_rate_eval = eval_metrics.get("success_rate", 0.0)
                    if success_rate_eval > self.best_eval_metric:
                        self.best_eval_metric = success_rate_eval
                        self.best_checkpoint_path = self.save_checkpoint(self.total_env_steps, is_best=True)
                    next_eval += eval_interval
                    # Evaluation reuses the training env (MetaDrive singleton). Reset training episode state.
                    obs = self.env.reset()
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                    current_feasible = self.env.current_feasible()
                    episode_reward = 0.0
                    episode_cost = 0.0
                    episode_length = 0

                if self.total_env_steps >= next_checkpoint:
                    self.save_checkpoint(self.total_env_steps)
                    next_checkpoint += checkpoint_interval
        finally:
            self.save_checkpoint(self.total_env_steps)
        if self.writer:
            self.writer.flush()

    def _optimize_model(self, last_value: torch.Tensor, last_cost_value: torch.Tensor, states: List[torch.Tensor], state_flags: List[float], costs: List[float]) -> Dict[str, float]:
        buffer_data = self.rollout_buffer.stacked()
        rewards = buffer_data.reward.view(-1, 1)
        costs_tensor = buffer_data.cost.view(-1, 1)
        dones = buffer_data.done.view(-1, 1)
        values = buffer_data.value.view(-1, 1)
        cost_values = buffer_data.cost_value.view(-1, 1)
        log_probs = buffer_data.log_prob.view(-1, 1)
        actions = buffer_data.action
        obs = buffer_data.obs

        gamma = self.config["training"].get("gamma", 0.99)
        gae_lambda = self.config["training"].get("gae_lambda", 0.95)
        advantages, returns = self.agent.compute_advantages(rewards, values, dones, last_value, gamma, gae_lambda)
        cost_advantages, cost_returns = self.agent.compute_advantages(costs_tensor, cost_values, dones, last_cost_value, gamma, gae_lambda)

        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std(unbiased=False) + 1e-8)

        validation_loss = 0.0
        short_term_violation = 0.0
        B_values = torch.zeros(advantages.size(0), device=self.device)
        if self.use_short_constraint and self.validation_net is not None:
            validation_batch = self._prepare_validation_targets(states, state_flags, advantages.size(0))
            if validation_batch is not None:
                trajectories, labels, indices = validation_batch
                preds = self.agent.validation_scores(trajectories)
                validation_loss_tensor = self.agent.validation_loss(preds, labels)
                short_term_violation = self.agent.short_term_violation(preds.detach(), labels.detach())
                B_values[indices] = preds.detach()
                if self.validation_optimizer is not None:
                    self.validation_optimizer.zero_grad()
                    validation_loss_tensor.backward()
                    self.validation_optimizer.step()
                validation_loss = float(validation_loss_tensor.detach().item())

        train_stats = self._ppo_update(
            obs=obs,
            actions=actions,
            old_log_probs=log_probs,
            advantages=advantages,
            returns=returns,
            cost_returns=cost_returns,
            cost_advantages=cost_advantages,
            B_values=B_values.unsqueeze(-1),
        )

        avg_cost = float(costs_tensor.mean().item())
        self.agent.update_lagrange_multipliers(avg_cost, short_term_violation)
        train_stats["validation_loss"] = validation_loss
        return train_stats

    def _ppo_update(self, **tensors: torch.Tensor) -> Dict[str, float]:
        obs = tensors["obs"]
        actions = tensors["actions"]
        old_log_probs = tensors["old_log_probs"]
        advantages = tensors["advantages"]
        returns = tensors["returns"]
        cost_returns = tensors["cost_returns"]
        cost_advantages = tensors["cost_advantages"]
        B_values = tensors["B_values"]

        training_cfg = self.config["training"]
        clip_ratio = training_cfg.get("clip_ratio", 0.2)
        value_coef = training_cfg.get("value_coef", 0.5)
        entropy_coef = training_cfg.get("entropy_coef", 0.0)
        max_grad_norm = training_cfg.get("max_grad_norm", 0.5)
        epochs = training_cfg.get("epochs_per_update", 10)
        minibatch_size = training_cfg.get("minibatch_size", 256)

        num_steps = obs.size(0)
        policy_loss_acc = 0.0
        value_loss_acc = 0.0
        cost_value_loss_acc = 0.0
        updates = 0

        for _ in range(epochs):
            permutation = torch.randperm(num_steps, device=self.device)
            for start in range(0, num_steps, minibatch_size):
                end = start + minibatch_size
                batch_idx = permutation[start:end]
                obs_b = obs[batch_idx]
                actions_b = actions[batch_idx]
                old_log_probs_b = old_log_probs[batch_idx]
                adv_b = advantages[batch_idx]
                returns_b = returns[batch_idx]
                cost_returns_b = cost_returns[batch_idx]
                cost_adv_b = cost_advantages[batch_idx]
                B_b = B_values[batch_idx]

                log_prob, entropy, values, cost_values = self.agent.evaluate_actions(obs_b, actions_b)
                ratio = torch.exp(log_prob - old_log_probs_b)
                penalty_term = self.agent.penalty_term(adv_b, cost_adv_b, B_b, ratio)
                surrogate1 = ratio * penalty_term
                surrogate2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * penalty_term
                policy_loss = -torch.min(surrogate1, surrogate2).mean() - entropy_coef * entropy.mean()

                value_loss = 0.5 * (returns_b - values).pow(2).mean()
                cost_value_loss = 0.5 * (cost_returns_b - cost_values).pow(2).mean()

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
                self.policy_optimizer.step()

                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_grad_norm)
                self.value_optimizer.step()

                self.cost_value_optimizer.zero_grad()
                cost_value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cost_value_net.parameters(), max_grad_norm)
                self.cost_value_optimizer.step()

                # Optional per-batch λ update for PPO-Lag style
                if hasattr(self.agent, "update_long_from_gap"):
                    cost_gap = float(cost_returns_b.mean().item() - training_cfg.get("cost_limit", self.config["constraints"]["cost_limit"]))
                    self.agent.update_long_from_gap(cost_gap)

                policy_loss_acc += float(policy_loss.detach().item())
                value_loss_acc += float(value_loss.detach().item())
                cost_value_loss_acc += float(cost_value_loss.detach().item())
                updates += 1

        return {
            "policy_loss": policy_loss_acc / max(1, updates),
            "value_loss": value_loss_acc / max(1, updates),
            "cost_value_loss": cost_value_loss_acc / max(1, updates),
        }

    def _prepare_validation_targets(self, states: List[torch.Tensor], state_flags: List[float], rollout_length: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, List[int]]]:
        horizon = self.config["constraints"].get("short_term_horizon", 5)
        if len(states) < horizon:
            return None
        try:
            trajectories, labels, indices = sliding_trajectories(states, state_flags, horizon, rollout_length)
        except ValueError:
            return None
        trajectories = trajectories.to(self.device)
        labels = labels.to(self.device)
        return trajectories, labels, indices

    def save_checkpoint(self, env_steps: int, is_best: bool = False) -> Path:
        checkpoint = {
            "policy": self.policy.state_dict(),
            "value_net": self.value_net.state_dict(),
            "cost_value_net": self.cost_value_net.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "cost_value_optimizer": self.cost_value_optimizer.state_dict(),
            "lambda_long": getattr(self.agent, "lambda_long", None),
            "lambda_short": getattr(self.agent, "lambda_short", None),
            "env_steps": env_steps,
        }
        if self.validation_net is not None:
            checkpoint["validation_net"] = self.validation_net.state_dict()
        if self.validation_optimizer is not None:
            checkpoint["validation_optimizer"] = self.validation_optimizer.state_dict()

        path = self.checkpoint_dir / f"checkpoint_{env_steps}.pt"
        try:
            torch.save(checkpoint, path)
            if not path.exists() or path.stat().st_size < 1024:
                raise IOError(f"Checkpoint file seems invalid: {path}")
            print(f"Checkpoint saved: {path}")
        except Exception as exc:
            backup = self.checkpoint_dir / f"checkpoint_{env_steps}_backup.pt"
            torch.save(checkpoint, backup)
            print(f"Error saving checkpoint ({exc}); backup written to {backup}")
            path = backup

        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            shutil.copy(path, best_path)
            path = best_path

        self._cleanup_old_checkpoints()
        return path

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.value_net.load_state_dict(checkpoint["value_net"])
        self.cost_value_net.load_state_dict(checkpoint["cost_value_net"])
        if self.validation_net is not None and "validation_net" in checkpoint:
            self.validation_net.load_state_dict(checkpoint["validation_net"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
        self.cost_value_optimizer.load_state_dict(checkpoint["cost_value_optimizer"])
        if self.validation_optimizer is not None and "validation_optimizer" in checkpoint:
            self.validation_optimizer.load_state_dict(checkpoint["validation_optimizer"])
        if "lambda_long" in checkpoint:
            loaded_long = checkpoint["lambda_long"]
            if loaded_long is not None:
                self.agent.lambda_long = loaded_long.to(self.device)
        if "lambda_short" in checkpoint:
            loaded_short = checkpoint["lambda_short"]
            if loaded_short is not None and hasattr(self.agent, "lambda_short"):
                self.agent.lambda_short = loaded_short.to(self.device)
        self.total_env_steps = checkpoint.get("env_steps", 0)
        # Track resume point to protect checkpoints at or before this step
        self._resume_checkpoint_step = self.total_env_steps
        # Enable resume mode for logger to append to existing CSV
        self.logger.set_resume(True)
        print(f"Resumed from checkpoint: env_steps={self.total_env_steps}")

    def _cleanup_old_checkpoints(self, keep_last: int = -1) -> None:
        """Clean up old checkpoints, keeping the most recent ones.

        Args:
            keep_last: Number of checkpoints to keep. Set to -1 to keep all (default).
        """
        # By default, keep all checkpoints (no cleanup)
        if keep_last < 0:
            return

        # If cleanup is enabled, proceed with deletion logic
        ckpts = sorted(
            [p for p in self.checkpoint_dir.glob("checkpoint_*.pt") if p.is_file()],
            key=lambda p: p.stat().st_mtime,
        )
        best_path = self.checkpoint_dir / "best.pt"

        # Extract step number from checkpoint filename
        def get_step(path: Path) -> int:
            try:
                # checkpoint_1000000.pt -> 1000000
                return int(path.stem.split("_")[1])
            except (IndexError, ValueError):
                return 0

        while len(ckpts) > keep_last:
            cand = ckpts.pop(0)
            # Don't delete best.pt
            if best_path.exists() and cand.samefile(best_path):
                continue
            # Don't delete checkpoints at or before resume point (protect history)
            if self._resume_checkpoint_step is not None:
                cand_step = get_step(cand)
                if cand_step <= self._resume_checkpoint_step:
                    continue
            cand.unlink(missing_ok=True)

    def evaluate(self, checkpoint_path: Optional[Path] = None, episodes: int = 20) -> Dict[str, float]:
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        # MetaDrive is a singleton - we must close training env, create eval env, then restore training env
        # Close training environment
        try:
            self.env.close()
        except Exception:
            pass
        LSTCMetaDriveEnv._cleanup_engine()

        # Create evaluation environment with different seed range (unseen maps)
        eval_env = LSTCMetaDriveEnv(self.eval_env_config)

        rewards = []
        costs = []
        successes = 0

        for _ in range(episodes):
            # No seed needed - eval env is configured with start_seed=20, num_scenarios=20
            # MetaDrive will randomly select from seeds 20-39
            obs = eval_env.reset()
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            done = False
            episode_reward = 0.0
            episode_cost = 0.0

            while not done:
                with torch.no_grad():
                    action = self.agent.act(obs_tensor.unsqueeze(0)).action.squeeze(0).detach().cpu().numpy()
                step_result = eval_env.step(action)
                episode_reward += float(step_result.reward)
                episode_cost += float(step_result.cost)
                done = bool(step_result.terminated or step_result.truncated)
                obs_tensor = torch.tensor(step_result.obs, dtype=torch.float32, device=self.device)
                if step_result.info.get("arrive_dest"):
                    successes += 1

            rewards.append(episode_reward)
            costs.append(episode_cost)

        # Close evaluation environment
        try:
            eval_env.close()
        except Exception:
            pass
        LSTCMetaDriveEnv._cleanup_engine()

        # Restore training environment
        self.env = LSTCMetaDriveEnv(self.env_config)

        metrics = {
            "success_rate": successes / max(1, episodes),
            "episode_reward": sum(rewards) / max(1, len(rewards)),
            "episode_cost": sum(costs) / max(1, len(costs)),
        }
        return metrics

    def _build_eval_env_config(self, train_env_config: Dict[str, Any], eval_override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a disjoint evaluation env config (separate map seed range)."""
        base = copy.deepcopy(train_env_config)
        if eval_override:
            # evaluation_environment can override any environment key
            base.update(eval_override)
            override_mcfg = eval_override.get("metadrive_config")
            if override_mcfg:
                merged = copy.deepcopy(base.get("metadrive_config", {}))
                merged.update(override_mcfg)
                base["metadrive_config"] = merged

        mcfg = copy.deepcopy(base.get("metadrive_config", {}))
        train_mcfg = train_env_config.get("metadrive_config", {})
        train_start = train_mcfg.get("start_seed", 0)
        train_num = train_mcfg.get("num_scenarios", 20)
        mcfg.setdefault("start_seed", train_start + train_num)
        mcfg.setdefault("num_scenarios", train_num)
        base["metadrive_config"] = mcfg
        return base

    def smoke_test(self, episodes: int = 1, max_steps: int = 200) -> Dict[str, float]:
        """Run a tiny rollout to verify reward/cost conventions."""
        env = self.env
        stats = {"episodes": episodes, "steps": 0, "feasible_rate": 0.0, "avg_cost": 0.0, "avg_reward": 0.0}
        feasible_counts = 0
        total_states = 0
        total_cost = 0.0
        total_reward = 0.0

        for _ in range(episodes):
            obs = env.reset()
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            feasible = env.current_feasible()
            done = False
            steps = 0
            while not done and steps < max_steps:
                feasible_counts += 1.0 if feasible else 0.0
                total_states += 1
                with torch.no_grad():
                    action = self.agent.act(obs_tensor.unsqueeze(0)).action.squeeze(0).detach().cpu().numpy()
                result = env.step(action)
                total_cost += result.cost
                total_reward += result.reward
                obs_tensor = torch.tensor(result.obs, dtype=torch.float32, device=self.device)
                feasible = result.feasible
                done = bool(result.terminated or result.truncated)
                steps += 1
                stats["steps"] += 1

        stats["feasible_rate"] = feasible_counts / max(1, total_states)
        stats["avg_cost"] = total_cost / max(1, stats["steps"])
        stats["avg_reward"] = total_reward / max(1, stats["steps"])
        return stats

    def _log_tensorboard_metrics(self, prefix: str, metrics: Dict[str, Any], step: int) -> None:
        if not self.writer:
            return
        for key, value in metrics.items():
            if isinstance(value, (float, int)):
                self.writer.add_scalar(f"{prefix}/{key}", value, step)
