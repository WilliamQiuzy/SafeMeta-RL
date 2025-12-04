# SafeMeta-RL

Dual Constarints for safe autonomous driving. We train PPO variants with dual Lagrangian constraints to balance reward maximization and safety. The environment is MetaDrive with 20 randomly generated maps for training and 20 unseen maps for evaluation.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train LSTC
PYTHONPATH=src python scripts/train.py --config configs/default.yaml --output runs/lstc

# Train baselines
PYTHONPATH=src python scripts/train.py --config configs/ppo.yaml --output runs/ppo
PYTHONPATH=src python scripts/train.py --config configs/ppo_lag.yaml --output runs/ppo_lag
```

## Project Structure

```
src/safemeta_rl/
├── agents/      # LSTC, PPO, PPO-Lag implementations
├── envs/        # MetaDrive wrapper with reward/cost functions
├── networks/    # Policy, Value, Cost-Value, Validation networks
└── training/    # Trainer loop
```
