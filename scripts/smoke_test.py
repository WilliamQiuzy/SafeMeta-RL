"""Quick smoke test to verify rollout signals from MetaDrive."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short rollout to verify safety signals.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"),
                        help="Path to YAML config file.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of test episodes.")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum steps per episode.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    from safemeta_rl.training.trainer import Trainer

    trainer = Trainer(config=config, output_dir=Path(config.get("log_dir", "runs")))
    stats = trainer.smoke_test(episodes=args.episodes, max_steps=args.max_steps)
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
