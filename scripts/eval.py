"""Evaluate a trained SafeMeta-RL (LSTC) checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LSTC agent")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"),
                        help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to trained policy checkpoint")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of evaluation episodes")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    from safemeta_rl.training.trainer import Trainer

    trainer = Trainer(config=config, output_dir=Path("runs"))
    metrics = trainer.evaluate(checkpoint_path=args.checkpoint, episodes=args.episodes)

    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
