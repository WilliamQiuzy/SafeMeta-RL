"""Save PPO trajectories (positions only) for later plotting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import yaml

from safemeta_rl.agents import PPOAgent
from safemeta_rl.envs.metadrive_wrapper import LSTCMetaDriveEnv
from safemeta_rl.networks.policy import GaussianPolicy
from safemeta_rl.networks.value import ValueNetwork
from safemeta_rl.networks.cost_value import build_cost_value_network


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save PPO trajectories (positions) for offline plotting.")
    parser.add_argument("--config", type=Path, default=Path("configs/ppo.yaml"), help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to PPO checkpoint.")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes to record.")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/trajectories/ppo_positions.json"), help="Output path.")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_agent(config: Dict[str, Any], checkpoint: Path, device: torch.device) -> PPOAgent:
    obs_dim = config["environment"].get("observation_dim", 49)
    action_dim = config["environment"].get("action_dim", 2)
    hidden = [64, 64]
    policy = GaussianPolicy(obs_dim, action_dim, hidden).to(device)
    value_net = ValueNetwork(obs_dim, hidden).to(device)
    cost_value_net = build_cost_value_network(obs_dim, hidden).to(device)
    agent = PPOAgent(policy, value_net, cost_value_net, config)

    state = torch.load(checkpoint, map_location=device)
    if "policy" in state:
        policy.load_state_dict(state["policy"])
    if "value_net" in state:
        value_net.load_state_dict(state["value_net"])
    if "cost_value_net" in state:
        cost_value_net.load_state_dict(state["cost_value_net"])
    return agent


def collect_positions(env: LSTCMetaDriveEnv, agent: PPOAgent, episodes: int, max_steps: int, device: torch.device) -> List[List[List[float]]]:
    all_trajs: List[List[List[float]]] = []
    for _ in range(episodes):
        obs = env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        done = False
        steps = 0
        positions: List[List[float]] = []
        vehicle = env._current_vehicle()
        if vehicle is not None:
            pos = env._extract_position(vehicle)
            if pos is not None and len(pos) >= 2:
                positions.append([float(pos[0]), float(pos[1])])
        while not done and steps < max_steps:
            with torch.no_grad():
                action = agent.act(obs_t.unsqueeze(0)).action.squeeze(0).cpu().numpy()
            result = env.step(action)
            vehicle = env._current_vehicle()
            if vehicle is not None:
                pos = env._extract_position(vehicle)
                if pos is not None and len(pos) >= 2:
                    positions.append([float(pos[0]), float(pos[1])])
            obs_t = torch.tensor(result.obs, dtype=torch.float32, device=device)
            done = bool(result.terminated or result.truncated)
            steps += 1
        all_trajs.append(positions)
    return all_trajs


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Force headless rollout
    env_cfg = dict(cfg["environment"])
    md_cfg = dict(env_cfg.get("metadrive_config", {}))
    md_cfg["use_render"] = False
    md_cfg["offscreen_render"] = True
    md_cfg["headless_machine_render"] = True
    env_cfg["metadrive_config"] = md_cfg

    agent = build_agent(cfg, args.checkpoint, device)
    env = LSTCMetaDriveEnv(env_cfg)
    trajs = collect_positions(env, agent, episodes=args.episodes, max_steps=args.max_steps, device=device)
    env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump({"trajectories": trajs}, f)
    print(f"Saved trajectories to {args.output}")


if __name__ == "__main__":
    main()
