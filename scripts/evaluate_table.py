"""Evaluate all algorithms on 3 maps for the paper table."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List
import os

os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import torch
import yaml

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from safemeta_rl.agents import PPOAgent, PPOLagAgent, LSTCAgent
from safemeta_rl.agents.lstc_pid_agent import LSTCPIDAgent
from safemeta_rl.envs.metadrive_wrapper import LSTCMetaDriveEnv
from safemeta_rl.networks.policy import GaussianPolicy
from safemeta_rl.networks.value import ValueNetwork
from safemeta_rl.networks.cost_value import build_cost_value_network
from safemeta_rl.networks.validation import ValidationNetwork


# Algorithm configurations
ALGORITHMS = {
    "PPO": {
        "config": "configs/ppo.yaml",
        "checkpoint": "runs/ppo/checkpoints/checkpoint_10000000.pt",
        "agent_class": PPOAgent,
        "needs_validation": False,
    },
    "PPO-Lag": {
        "config": "configs/ppo_lag.yaml",
        "checkpoint": "runs/ppo_lag/checkpoints/checkpoint_10000000.pt",
        "agent_class": PPOLagAgent,
        "needs_validation": False,
    },
    "LSTC": {
        "config": "configs/default.yaml",
        "checkpoint": "runs/lstc/checkpoints/checkpoint_10000000.pt",
        "agent_class": LSTCAgent,
        "needs_validation": True,
    },
    "LSTC-PID": {
        "config": "configs/lstc_pid.yaml",
        "checkpoint": "runs/lstc_pid/checkpoints/checkpoint_10000000.pt",
        "agent_class": LSTCPIDAgent,
        "needs_validation": True,
    },
}

# Map configurations matching the table
MAPS = {
    "InRamp": "rS",      # Ramp merge
    "Slet": "SCS",       # Sharp left turn (curve)
    "Rodab": "SOS",      # Roundabout
}


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_agent(algo_name: str, algo_cfg: Dict[str, Any], device: torch.device):
    """Build and load an agent from checkpoint."""
    config = load_config(Path(algo_cfg["config"]))
    checkpoint_path = Path(algo_cfg["checkpoint"])

    if not checkpoint_path.exists():
        print(f"Warning: Checkpoint not found: {checkpoint_path}")
        return None

    obs_dim = config["environment"].get("observation_dim", 49)
    action_dim = config["environment"].get("action_dim", 2)
    hidden = [64, 64]

    policy = GaussianPolicy(obs_dim, action_dim, hidden).to(device)
    value_net = ValueNetwork(obs_dim, hidden).to(device)
    cost_value_net = build_cost_value_network(obs_dim, hidden).to(device)

    validation_net = None
    if algo_cfg["needs_validation"]:
        short_term_horizon = config.get("constraints", {}).get("short_term_horizon", 5)
        trajectory_dim = obs_dim * short_term_horizon
        validation_net = ValidationNetwork(trajectory_dim, hidden).to(device)

    agent_class = algo_cfg["agent_class"]
    if agent_class == PPOAgent:
        agent = PPOAgent(policy, value_net, cost_value_net, config)
    elif agent_class == PPOLagAgent:
        agent = PPOLagAgent(policy, value_net, cost_value_net, config)
    elif agent_class == LSTCAgent:
        agent = LSTCAgent(policy, value_net, cost_value_net, validation_net, config)
    elif agent_class == LSTCPIDAgent:
        agent = LSTCPIDAgent(policy, value_net, cost_value_net, validation_net, config)
    else:
        raise ValueError(f"Unknown agent class: {agent_class}")

    state = torch.load(checkpoint_path, map_location=device)
    if "policy" in state:
        policy.load_state_dict(state["policy"])
    if "value_net" in state:
        value_net.load_state_dict(state["value_net"])
    if "cost_value_net" in state:
        cost_value_net.load_state_dict(state["cost_value_net"])
    if validation_net is not None and "validation_net" in state:
        validation_net.load_state_dict(state["validation_net"])

    return agent


def create_env(map_string: str) -> LSTCMetaDriveEnv:
    """Create environment with specified map."""
    env_cfg = {
        "observation_dim": 49,
        "action_dim": 2,
        "max_episode_steps": 1000,
        "metadrive_config": {
            "map": map_string,
            "traffic_density": 0.0,
            "accident_prob": 0.0,
            "use_render": False,
            "offscreen_render": False,
            "headless_machine_render": True,
            "environment_num": 1,
            "start_seed": 0,
        }
    }
    return LSTCMetaDriveEnv(env_cfg)


def evaluate_agent(env: LSTCMetaDriveEnv, agent, episodes: int, device: torch.device) -> Dict[str, float]:
    """Evaluate agent and return success rate and episode cost."""
    successes = []
    costs = []

    for ep in range(episodes):
        obs = env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        done = False
        episode_cost = 0.0
        final_info = {}
        step_count = 0

        while not done:
            with torch.no_grad():
                action = agent.act(obs_t.unsqueeze(0)).action.squeeze(0).cpu().numpy()

            result = env.step(action)
            obs_t = torch.tensor(result.obs, dtype=torch.float32, device=device)
            episode_cost += result.cost
            final_info = result.info  # Keep last info
            done = bool(result.terminated or result.truncated)
            step_count += 1

        # Check success (reached destination)
        success = final_info.get("arrive_dest", False)

        # Debug print for first few episodes
        if ep < 3:
            print(f"    Ep {ep}: steps={step_count}, cost={episode_cost:.2f}, "
                  f"arrive={success}, crash={final_info.get('crash', False)}, "
                  f"out_of_road={final_info.get('out_of_road', False)}")

        successes.append(1.0 if success else 0.0)
        costs.append(episode_cost)

    return {
        "success_rate": np.mean(successes),
        "episode_cost": np.mean(costs),
        "std_success": np.std(successes),
        "std_cost": np.std(costs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per evaluation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load all agents
    print("Loading agents...")
    agents = {}
    for algo_name, algo_cfg in ALGORITHMS.items():
        agents[algo_name] = build_agent(algo_name, algo_cfg, device)
        if agents[algo_name]:
            print(f"  Loaded {algo_name}")

    # Results storage
    results = {algo: {map_name: {} for map_name in MAPS} for algo in ALGORITHMS}

    # Evaluate each algorithm on each map
    for map_name, map_string in MAPS.items():
        print(f"\n=== Evaluating on {map_name} ({map_string}) ===")
        env = create_env(map_string)

        for algo_name in ALGORITHMS:
            agent = agents[algo_name]
            if agent is None:
                print(f"  {algo_name}: SKIPPED (no checkpoint)")
                results[algo_name][map_name] = {"success_rate": 0.0, "episode_cost": 0.0}
                continue

            print(f"  Evaluating {algo_name}...", end=" ", flush=True)
            metrics = evaluate_agent(env, agent, args.episodes, device)
            results[algo_name][map_name] = metrics
            print(f"Success: {metrics['success_rate']:.2f}, Cost: {metrics['episode_cost']:.2f}")

        env.close()

    # Print LaTeX table
    print("\n" + "="*60)
    print("LaTeX Table Results:")
    print("="*60)

    map_names = list(MAPS.keys())

    print("\\begin{tabular}{lcccccc}")
    print("    \\toprule")
    print("    \\multirow{2}{*}{Method} &")
    print("    \\multicolumn{3}{c}{Success rate $\\uparrow$} &")
    print("    \\multicolumn{3}{c}{Episode cost $\\downarrow$} \\\\")
    print("    \\cmidrule(lr){2-4}\\cmidrule(lr){5-7}")
    print(f"    & {' & '.join(map_names)} & {' & '.join(map_names)} \\\\")
    print("    \\midrule")

    for algo_name in ALGORITHMS:
        success_vals = [f"{results[algo_name][m]['success_rate']:.2f}" for m in map_names]
        cost_vals = [f"{results[algo_name][m]['episode_cost']:.2f}" for m in map_names]

        display_name = algo_name
        if algo_name == "LSTC-PID":
            display_name = "LSTC-PID (ours)"

        print(f"    {display_name} & {' & '.join(success_vals)} & {' & '.join(cost_vals)} \\\\")

    print("    \\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()
