#!/usr/bin/env python3
"""Plot combined reward curves for PPO, PPO-Lag, and LSTC with shaded std regions."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def smooth_with_std(data, window=100):
    """Apply moving average smoothing and compute rolling std."""
    series = pd.Series(data)
    mean = series.rolling(window=window, min_periods=1).mean().values
    std = series.rolling(window=window, min_periods=1).std().values
    std = np.nan_to_num(std, nan=0.0)
    return mean, std


def main():
    # Load data
    base_dir = Path(__file__).parent.parent

    ppo_df = pd.read_csv(base_dir / "runs/ppo/metrics.csv")
    ppo_lag_df = pd.read_csv(base_dir / "runs/ppo_lag/metrics.csv")
    lstc_df = pd.read_csv(base_dir / "runs/lstc/metrics.csv")

    # Create figure like paper's Fig.5(a) - Step-episode reward curves
    fig, ax = plt.subplots(figsize=(8, 6))

    window = 100

    # Colors matching the reference image style
    colors = {
        'ppo': '#e74c3c',      # red
        'ppo_lag': '#3498db',  # blue
        'lstc': '#27ae60',     # green
    }

    # PPO - red
    x = ppo_df['env_steps'] / 1e6
    mean, std = smooth_with_std(ppo_df['avg_episode_reward'], window)
    ax.fill_between(x, mean - std, mean + std, color=colors['ppo'], alpha=0.2)
    ax.plot(x, mean, color=colors['ppo'], linewidth=2, label='PPO')

    # PPO-Lag - blue
    x = ppo_lag_df['env_steps'] / 1e6
    mean, std = smooth_with_std(ppo_lag_df['avg_episode_reward'], window)
    ax.fill_between(x, mean - std, mean + std, color=colors['ppo_lag'], alpha=0.2)
    ax.plot(x, mean, color=colors['ppo_lag'], linewidth=2, label='PPO-Lag')

    # LSTC - green
    x = lstc_df['env_steps'] / 1e6
    mean, std = smooth_with_std(lstc_df['avg_episode_reward'], window)
    ax.fill_between(x, mean - std, mean + std, color=colors['lstc'], alpha=0.2)
    ax.plot(x, mean, color=colors['lstc'], linewidth=2, label='LSTC')

    # Styling
    ax.set_xlabel('Environment Steps (×10⁶)', fontsize=12)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim(0, 10)
    ax.set_ylim(-50, 400)

    # Clean style like reference
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save figure
    output_dir = base_dir / "artifacts/figures/comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "reward_curves_combined.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")

    plt.close()


if __name__ == "__main__":
    main()
