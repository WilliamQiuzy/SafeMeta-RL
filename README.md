# SafeMeta-RL: Dual-Constraint Safe RL for Autonomous Driving

This repository contains the code for **dual-constraint safe reinforcement learning** on the MetaDrive *European Town* benchmark.
We study how increasingly safety-aware variants of PPO behave in dense urban traffic:

- **PPO** – vanilla reward-only PPO  
- **PPO-Lag** – PPO with long-term Lagrangian safety constraints  
- **LSTC** – Long- and Short-Term Constraints RL with trajectory-level correction  
- **LSTC-PID** – our PID-enhanced variant of LSTC

All methods are trained and evaluated on 20 randomly generated town maps for training and 20 held-out maps for evaluation.

---

## Installation

Tested with Python 3.10+ on macOS (Apple Silicon) and Linux.

```bash
git clone https://github.com/WilliamQiuzy/SafeMeta-RL.git
cd SafeMeta-RL

# Create / activate your virtualenv first, then:
pip install -r requirements.txt
```

MetaDrive will be installed as part of the requirements; see the MetaDrive documentation if you need GPU rendering or other advanced features.

---

## Training

All training scripts assume `PYTHONPATH=src` so that the `safemeta_rl` package is visible.

```bash
# PPO baseline
PYTHONPATH=src python scripts/train.py \
  --config configs/ppo.yaml \
  --output runs/ppo

# PPO-Lag baseline
PYTHONPATH=src python scripts/train.py \
  --config configs/ppo_lag.yaml \
  --output runs/ppo_lag

# LSTC (long/short-term constraints)
PYTHONPATH=src python scripts/train.py \
  --config configs/default.yaml \
  --output runs/lstc

# LSTC-PID (ours)
PYTHONPATH=src python scripts/train.py \
  --config configs/lstc_pid.yaml \
  --output runs/lstc_pid
```

Training uses `10^7` environment steps by default and may take several hours on a single machine.
You can reduce `total_env_steps` in the config files for quick sanity checks.

---

## Evaluation and Figures

### Evaluate and export metrics

```bash
# Evaluate all trained checkpoints and export metrics.csv into runs/*/
PYTHONPATH=src python scripts/eval.py
```

The repository ignores `runs/*/metrics.csv` via `.gitignore` so you can freely regenerate them locally.

---

## Project Structure

```text
src/safemeta_rl/
├── agents/        # PPO, PPO-Lag, LSTC, LSTC-PID agents
├── envs/          # MetaDrive wrapper, reward and safety cost functions
├── networks/      # Policy, value, cost-value, validation networks
└── training/      # Trainer, rollout collection, logging

configs/           # YAML configs for all algorithms
scripts/           # Training, evaluation, plotting, smoke tests
runs/              # Output directory for experiments (ignored in part by git)
artifacts/         # Saved figures and example trajectories
```

---

## Reference

If you find this code useful, please consider citing our work:

```bibtex
@misc{qiu2024safemeta,
  author  = {Ziyue Qiu},
  title   = {Dual Constraints Safe Reinforcement Learning for Autonomous Driving},
  year    = {2024},
  note    = {Technical report},
  howpublished = {\url{https://github.com/WilliamQiuzy/SafeMeta-RL}}
}
```

For questions or feedback, feel free to open an issue or contact: `ziyueqiu@mit.edu`. ***!
