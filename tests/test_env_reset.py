"""Basic smoke tests for MetaDrive wrapper reset/step/close in headless mode."""
import pytest

from safemeta_rl.envs.metadrive_wrapper import LSTCMetaDriveEnv
import yaml
from pathlib import Path


def _headless_env_config():
    cfg = yaml.safe_load(Path("configs/ppo.yaml").read_text())
    env_cfg = dict(cfg["environment"])
    md_cfg = dict(env_cfg.get("metadrive_config", {}))
    md_cfg["use_render"] = False
    md_cfg["offscreen_render"] = True
    md_cfg["headless_machine_render"] = True
    env_cfg["metadrive_config"] = md_cfg
    return env_cfg


@pytest.mark.filterwarnings("ignore:.*np.bool.*")
def test_reset_step_close_twice():
    env_cfg = _headless_env_config()
    env = LSTCMetaDriveEnv(env_cfg)
    obs1 = env.reset()
    assert len(obs1) == env_cfg.get("observation_dim", 49)
    result = env.step(env.action_space.sample())
    assert result.obs is not None
    env.close()

    # Rebuild and reset again to ensure singleton is handled
    env = LSTCMetaDriveEnv(env_cfg)
    obs2 = env.reset()
    assert len(obs2) == env_cfg.get("observation_dim", 49)
    env.close()
