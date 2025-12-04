"""Entry point for training the SafeMeta-RL (LSTC) agent."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTC agent in MetaDrive")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"),
                        help="Path to YAML config file")
    parser.add_argument("--output", type=Path, default=Path("runs"),
                        help="Directory for logs and checkpoints")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Optional checkpoint to resume training from")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # Deferred imports to keep CLI fast when only inspecting configs
    from safemeta_rl.training.trainer import Trainer

    trainer = Trainer(config=config, output_dir=args.output)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
