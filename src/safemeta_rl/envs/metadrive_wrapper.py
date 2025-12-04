"""Environment utilities for MetaDrive integration."""
from __future__ import annotations

import configparser
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import numpy as np

# Force Panda3D headless mode to avoid window creation issues in headless/macOS runs
# These MUST be set before ANY panda3d or metadrive imports
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ["PANDA_PRC_DIR"] = "/tmp"  # Avoid loading any user prc files

try:  # pragma: no cover - Panda3D config
    from panda3d.core import loadPrcFileData  # type: ignore

    # Use 'none' window type - this completely disables window creation
    loadPrcFileData("", "window-type none")
    loadPrcFileData("", "audio-library-name null")
    # Try headlessgl first (better for rendering if available), fall back to tinydisplay
    loadPrcFileData("", "load-display p3headlessgl")
    loadPrcFileData("", "aux-display p3tinydisplay")
    loadPrcFileData("", "notify-level-display error")
    loadPrcFileData("", "notify-level-pnmimage error")
except Exception:
    pass

# Compatibility for older MetaDrive using deprecated numpy aliases
# These were removed in numpy 2.0, we add them back for compatibility
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    try:
        _ = np.float  # type: ignore
    except AttributeError:
        np.float = float  # type: ignore
    try:
        _ = np.int  # type: ignore
    except AttributeError:
        np.int = int  # type: ignore
    try:
        _ = np.uint  # type: ignore
    except AttributeError:
        np.uint = np.uint64  # type: ignore
    try:
        _ = np.bool  # type: ignore
    except AttributeError:
        np.bool = np.bool_  # type: ignore

from .safety_labeler import SafetyLabeler

# Populated lazily to allow config bootstrap and support multiple MetaDrive versions
MetaDriveEnv = None  # type: ignore


@dataclass
class StepResult:
    """Container returned by `LSTCMetaDriveEnv.step`."""

    obs: Any
    reward: float
    cost: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]
    feasible: bool
    reward_components: Dict[str, float]


class LSTCMetaDriveEnv:
    """Thin wrapper that exposes reward/cost signals expected by the agent."""

    @staticmethod
    def _cleanup_engine() -> None:
        """Best-effort teardown of Panda3D/MetaDrive singletons."""
        import gc
        # First close MetaDrive engine
        try:
            from metadrive.engine.engine_utils import close_engine, get_engine  # type: ignore
            engine = get_engine()
            if engine is not None:
                try:
                    engine.close()
                except Exception:
                    pass
            close_engine()
        except Exception:
            pass
        # Then destroy ShowBase instances
        try:
            from direct.showbase.ShowBase import ShowBase  # type: ignore
            if getattr(ShowBase, "instances", None):
                for inst in list(ShowBase.instances):
                    try:
                        inst.destroy()
                    except Exception:
                        pass
                try:
                    ShowBase.instances.clear()
                except Exception:
                    pass
        except Exception:
            pass
        # Also try to clear the global base reference
        try:
            import builtins
            if hasattr(builtins, "base"):
                try:
                    builtins.base.destroy()
                except Exception:
                    pass
                try:
                    delattr(builtins, "base")
                except Exception:
                    pass
        except Exception:
            pass
        # Clear cached ShowBase singleton holders if present
        try:
            import direct.showbase.ShowBaseGlobal as ShowBaseGlobal  # type: ignore

            if hasattr(ShowBaseGlobal, "base"):
                try:
                    ShowBaseGlobal.base.destroy()
                except Exception:
                    pass
                ShowBaseGlobal.base = None
        except Exception:
            pass
        # Force garbage collection
        gc.collect()

    def __init__(self, env_config: Dict[str, Any]):
        # Ensure any stale MetaDrive engine is closed before creating a new one (singleton engine)
        self._cleanup_engine()
        bootstrap_metadrive_config(env_config.get("metadrive_credentials", {}))

        global MetaDriveEnv  # pylint: disable=global-statement
        if MetaDriveEnv is None:
            try:
                from metadrive.envs.metadrive_env import MetaDriveEnv as _MetaDriveEnv  # type: ignore
            except ImportError:
                try:
                    from metadrive import MetaDriveEnv as _MetaDriveEnv  # type: ignore
                except ImportError as exc:  # pragma: no cover
                    raise ImportError(
                        "MetaDrive is not installed or incompatible. Please install metadrive>=1.4."
                    ) from exc
            MetaDriveEnv = _MetaDriveEnv

        metadrive_cfg = dict(env_config.get("metadrive_config", {}))

        # Remove termination flags unsupported in 0.2.x
        for key in ["crash_vehicle_done", "crash_object_done", "crash_human_done", "out_of_road_done", "truncate_as_terminate"]:
            metadrive_cfg.pop(key, None)

        base_cfg = MetaDriveEnv.default_config()
        cfg_keys = set(base_cfg.keys()) if hasattr(base_cfg, "keys") else set(base_cfg)
        # Keep only keys the installed MetaDrive understands
        if "environment_num" not in cfg_keys and "num_scenarios" in metadrive_cfg:
            # 0.2.x uses num_scenarios; leave as-is
            metadrive_cfg.pop("environment_num", None)
        if "environment_num" in cfg_keys and "num_scenarios" in metadrive_cfg and "environment_num" not in metadrive_cfg:
            metadrive_cfg["environment_num"] = metadrive_cfg.pop("num_scenarios")
        for key in ["offscreen_render", "headless_machine_render"]:
            if key not in cfg_keys:
                metadrive_cfg.pop(key, None)
        # Force headless/offscreen to avoid window creation issues
        if "use_render" in cfg_keys:
            base_cfg["use_render"] = False
        if "image_observation" in cfg_keys:
            base_cfg["image_observation"] = False
        if "offscreen_render" in cfg_keys:
            base_cfg["offscreen_render"] = False  # No rendering at all
        if "headless_machine_render" in cfg_keys:
            base_cfg["headless_machine_render"] = True
        if "_render_mode" in cfg_keys:
            base_cfg["_render_mode"] = "none"
        if "render_pipeline" in cfg_keys:
            base_cfg["render_pipeline"] = False
        if "force_destroy" in cfg_keys:
            base_cfg["force_destroy"] = True
        if "vehicle_config" not in metadrive_cfg:
            metadrive_cfg["vehicle_config"] = {"lidar": {"num_lasers": 30, "distance": 50.0}}

        # Respect episode horizon from config so rollouts terminate as intended
        horizon = env_config.get("max_episode_steps")
        if horizon is not None:
            metadrive_cfg["horizon"] = horizon
            # truncate_as_terminate not present in 0.2.x; horizon alone bounds episode
        base_cfg.update(metadrive_cfg)
        self._base_cfg = base_cfg
        self._env = MetaDriveEnv(self._base_cfg)
        self._safety = SafetyLabeler(env_config.get("safety_thresholds"))
        self._last_feasible = True
        self._last_info: Dict[str, Any] = {}
        self._cost_keys = ["cost", "crash_vehicle", "crash_object", "crash_human", "out_of_road"]
        self._obs_dim = env_config.get("observation_dim", 49)
        self._lidar_dim = int(base_cfg.get("vehicle_config", {}).get("lidar", {}).get("num_lasers", 30))
        self._state_dim = max(0, self._obs_dim - self._lidar_dim)
        self._lane_width = float(base_cfg.get("map_config", {}).get("LANE_WIDTH", 3.5))
        self._max_speed = env_config.get("max_speed", 22.22)  # 80 km/h
        self._last_position = None
        self._last_heading = None  # Track heading for longitudinal distance calculation
        self._last_reward_components: Dict[str, float] = {}

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self, seed: Optional[int] = None, **kwargs) -> Any:
        """Reset environment, handling MetaDrive 0.4.x tuple returns.

        Args:
            seed: Optional seed to select a specific scenario. MetaDrive uses this
                  to pick from num_scenarios maps starting at start_seed.
                  If None, MetaDrive randomly selects from available scenarios.
        """
        # Remove unsupported kwargs that may cause issues
        kwargs.pop("options", None)

        # Build reset kwargs, including seed if provided for scenario selection
        reset_kwargs = dict(kwargs)
        if seed is not None:
            reset_kwargs["seed"] = seed

        # Try simple reset first (most common case)
        try:
            result = self._env.reset(**reset_kwargs)
            if isinstance(result, tuple):
                obs = result[0]
            else:
                obs = result
        except TypeError:
            # Some MetaDrive versions don't support seed kwarg, try without
            try:
                result = self._env.reset()
                if isinstance(result, tuple):
                    obs = result[0]
                else:
                    obs = result
            except Exception as e:
                raise e
        except Exception as e:
            # If reset fails, the environment may be in a bad state
            # We need to recreate it entirely
            import time
            last_error = e
            obs = None

            for attempt in range(3):
                # Clean up everything
                self._cleanup_engine()
                try:
                    self._env.close()
                except Exception:
                    pass
                self._cleanup_engine()

                # Wait a bit for resources to be released
                time.sleep(0.5)

                # Try to recreate environment
                try:
                    self._env = MetaDriveEnv(self._base_cfg)
                    result = self._env.reset()
                    if isinstance(result, tuple):
                        obs = result[0]
                    else:
                        obs = result
                    break  # Success
                except Exception as e2:
                    last_error = e2
                    obs = None

            if obs is None:
                raise RuntimeError(
                    f"Failed to reset MetaDrive environment after multiple attempts. "
                    f"Last error: {last_error}"
                )
        vehicle = self._current_vehicle()
        position = self._extract_position(vehicle)
        heading = self._extract_heading(vehicle)
        self._last_position = position if position is not None else None
        self._last_heading = heading if heading is not None else None
        normalized_obs = self._build_observation(obs)
        self._last_info = {}
        self._last_feasible = self._safety.classify(normalized_obs, info=self._last_info)
        return normalized_obs

    def step(self, action) -> StepResult:
        step_result = self._env.step(action)
        obs, reward, terminated, truncated, info = self._unpack_step_result(step_result)
        obs_vec = self._build_observation(obs)
        # IMPORTANT: Augment info BEFORE computing reward so heading_error etc. are available
        self._augment_info_with_state(info, obs_vec)
        reward = self._compute_reward(obs_vec, info)
        cost = self._compute_cost(info)
        feasible = self._safety.classify(obs_vec, info=info)
        info["feasible_state"] = feasible
        self._last_feasible = feasible
        self._last_info = info
        if terminated or truncated:
            vehicle = self._current_vehicle()
            position = self._extract_position(vehicle)
            self._last_position = position if position is not None else None

        return StepResult(
            obs=obs_vec,
            reward=reward,
            cost=float(cost),
            terminated=terminated,
            truncated=truncated,
            info=info,
            feasible=feasible,
            reward_components=self._last_reward_components,
        )

    def close(self) -> None:
        self._env.close()

    def current_feasible(self) -> bool:
        return self._last_feasible

    def _unpack_step_result(self, result: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Support both 4-tuple (obs, reward, done, info) and 5-tuple gymnasium API."""
        if isinstance(result, (list, tuple)):
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
            elif len(result) == 4:
                obs, reward, done, info = result
                terminated, truncated = bool(done), False
            else:  # pragma: no cover - unexpected API
                raise ValueError(f"Unexpected step() return structure: {result}")
        else:  # pragma: no cover
            raise ValueError(f"Unexpected step() return type: {type(result)}")
        return obs, float(reward), bool(terminated), bool(truncated), dict(info)

    def _compute_cost(self, info: Dict[str, Any]) -> float:
        """Compute cost based on safety violations.

        According to the paper, cost is +1 when the vehicle crashes or departs from lane.
        """
        # Direct cost from MetaDrive cost_function if present
        if "cost" in info:
            try:
                return float(info["cost"])
            except (TypeError, ValueError):
                pass

        violations = any(info.get(key) for key in self._cost_keys)

        # MetaDrive 0.2.x returns TerminationState enums inside info; check their values as a fallback
        try:
            from metadrive.constants import TerminationState  # type: ignore

            violation_states = [
                TerminationState.CRASH,
                TerminationState.CRASH_VEHICLE,
                TerminationState.CRASH_OBJECT,
                TerminationState.CRASH_HUMAN,
                TerminationState.CRASH_BUILDING,
                TerminationState.OUT_OF_ROAD,
            ]
            violations = violations or any(info.get(state, False) for state in violation_states)
        except Exception:
            pass

        return 1.0 if violations else 0.0

    @staticmethod
    def _flatten_observation(obs: Any) -> np.ndarray:
        """Legacy helper kept for compatibility."""
        if isinstance(obs, dict):
            values = []
            for key in sorted(obs.keys()):
                val = obs[key]
                if isinstance(val, (dict, list, tuple)):
                    values.append(LSTCMetaDriveEnv._flatten_observation(val))
                elif isinstance(val, str):
                    continue
                else:
                    values.append(np.asarray(val, dtype=np.float32).ravel())
            if values:
                return np.concatenate(values)
            return np.array([], dtype=np.float32)
        if isinstance(obs, (list, tuple)):
            return np.concatenate([LSTCMetaDriveEnv._flatten_observation(x) for x in obs])
        if isinstance(obs, str):
            return np.array([], dtype=np.float32)
        return np.asarray(obs, dtype=np.float32).ravel()

    def _build_observation(self, raw_obs: Any) -> np.ndarray:
        if isinstance(raw_obs, np.ndarray):
            obs_vec = raw_obs.astype(np.float32).flatten()
        elif isinstance(raw_obs, (list, tuple)):
            obs_vec = self._flatten_any(raw_obs)
        else:
            lidar = self._extract_lidar(raw_obs)
            state = self._extract_state(raw_obs)
            checkpoints = self._extract_checkpoints(raw_obs)
            obs_vec = np.concatenate([lidar, state, checkpoints], axis=0)

        # Reorder to [lidar, state...] because MetaDrive 0.2.x concatenates state before lidar.
        if isinstance(raw_obs, np.ndarray) and obs_vec.shape[0] >= self._lidar_dim and self._state_dim > 0:
            lidar_part = obs_vec[-self._lidar_dim :]
            state_part = obs_vec[: obs_vec.shape[0] - self._lidar_dim]
            obs_vec = np.concatenate([lidar_part, state_part], axis=0)

        if obs_vec.shape[0] < self._obs_dim:
            padded = np.zeros(self._obs_dim, dtype=np.float32)
            padded[: obs_vec.shape[0]] = obs_vec
            return padded
        return obs_vec[: self._obs_dim]

    def _extract_lidar(self, raw_obs: Any) -> np.ndarray:
        values: Optional[np.ndarray] = None
        lidar = None
        if isinstance(raw_obs, dict):
            lidar = raw_obs.get("lidar")
        if isinstance(lidar, dict):
            if "distance" in lidar:
                values = np.asarray(lidar["distance"], dtype=np.float32)
            elif "point_cloud" in lidar:
                pc = np.asarray(lidar["point_cloud"], dtype=np.float32)
                if pc.ndim >= 2:
                    values = np.linalg.norm(pc[..., :2], axis=-1)
        elif lidar is not None:
            values = np.asarray(lidar, dtype=np.float32)
        if values is None and isinstance(raw_obs, dict) and "observation" in raw_obs:
            values = np.asarray(raw_obs["observation"][:30], dtype=np.float32)
        if values is None:
            values = np.zeros(30, dtype=np.float32)
        lidar_dim = 30
        if values.shape[0] >= lidar_dim:
            return values[:lidar_dim]
        padded = np.zeros(lidar_dim, dtype=np.float32)
        padded[: values.shape[0]] = values
        return padded

    def _extract_state(self, raw_obs: Any) -> np.ndarray:
        state_source = None
        if isinstance(raw_obs, dict):
            state_source = raw_obs.get("state") or raw_obs.get("vehicle_state") or raw_obs.get("vector")
        if state_source is None:
            return np.zeros(9, dtype=np.float32)
        values = np.asarray(state_source, dtype=np.float32).flatten()
        if values.shape[0] >= 9:
            return values[:9]
        padded = np.zeros(9, dtype=np.float32)
        padded[: values.shape[0]] = values
        return padded

    def _extract_checkpoints(self, raw_obs: Any) -> np.ndarray:
        nav = None
        if isinstance(raw_obs, dict):
            nav = raw_obs.get("navigation") or raw_obs.get("route") or raw_obs.get("navi_distance")
        if isinstance(nav, dict):
            for key in ("distance", "route_dists", "navi_distance"):
                if key in nav:
                    nav = nav[key]
                    break
        if nav is None:
            return np.zeros(10, dtype=np.float32)
        values = np.asarray(nav, dtype=np.float32).flatten()
        if values.shape[0] >= 10:
            return values[:10]
        padded = np.zeros(10, dtype=np.float32)
        padded[: values.shape[0]] = values
        return padded

    def _extract_position(self, vehicle) -> Optional[np.ndarray]:
        if vehicle is None:
            return None
        try:
            return np.array(vehicle.position, dtype=np.float32)
        except Exception:
            return None

    def _extract_heading(self, vehicle) -> Optional[float]:
        """Extract vehicle heading angle in radians."""
        if vehicle is None:
            return None
        try:
            # MetaDrive provides heading_theta in radians
            return float(vehicle.heading_theta)
        except Exception:
            pass
        try:
            # Fallback: some versions use heading property
            return float(vehicle.heading)
        except Exception:
            return None

    def _current_vehicle(self):
        return getattr(self._env, "agent", None) or getattr(self._env, "vehicle", None)

    @staticmethod
    def _flatten_any(value: Any) -> np.ndarray:
        parts: List[np.ndarray] = []

        def _collect(v: Any) -> None:
            if isinstance(v, dict):
                for key in sorted(v.keys()):
                    _collect(v[key])
            elif isinstance(v, (list, tuple)):
                for item in v:
                    _collect(item)
            elif isinstance(v, str) or v is None:
                return
            else:
                arr = np.asarray(v, dtype=np.float32)
                if arr.ndim == 0:
                    arr = arr.reshape(1)
                parts.append(arr.ravel())

        _collect(value)
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts, axis=0)

    def _compute_reward(self, obs_vec: np.ndarray, info: Dict[str, Any]) -> float:
        c1, c2, c3, c4 = 1.0, 0.1, -0.4, -0.4

        # R_dis: Longitudinal distance (Paper Eq.20)
        # "the longitudinal distance that the vehicle moves between two consecutive time steps"
        # We compute the projection of displacement onto the heading direction
        vehicle = self._current_vehicle()
        current_pos = self._extract_position(vehicle)
        current_heading = self._extract_heading(vehicle)

        R_dis = 0.0
        if current_pos is not None and self._last_position is not None and self._last_heading is not None:
            # Displacement vector
            delta_pos = current_pos - self._last_position
            # Heading direction vector (using last heading for consistency)
            heading_vec = np.array([np.cos(self._last_heading), np.sin(self._last_heading)], dtype=np.float32)
            # Longitudinal distance = projection onto heading direction
            R_dis = float(np.dot(delta_pos[:2], heading_vec))

        # Update position and heading for next step
        if current_pos is not None:
            self._last_position = current_pos
        if current_heading is not None:
            self._last_heading = current_heading

        # Fallback to MetaDrive's driving_reward if our calculation gives 0
        if R_dis == 0.0:
            R_dis = self._info_value(info, ["driving_reward", "distance_reward", "progress_reward"])
        if R_dis == 0.0:
            R_dis = float(info.get("delta_distance", info.get("step_distance", 0.0)))

        speed_reward = self._info_value(info, ["speed_reward", "reward_speed"])
        if speed_reward == 0.0:
            speed = self._info_value(info, ["speed", "velocity"])
            speed_reward = speed / self._max_speed if self._max_speed > 0 else 0.0
        R_speed = speed_reward

        # R_yaw: heading deviation between two consecutive time steps (yaw rate)
        # Paper Eq.20: This should be the yaw rate, not absolute heading error
        # yaw_rate is already available in info after _augment_info_with_state
        yaw_rate = self._info_value(info, ["yaw_rate", "yaw_penalty", "yaw_reward"])
        if yaw_rate == 0.0:
            yaw_rate = float(info.get("yaw_rate", 0.0))
        # yaw_rate is normalized [0, 1], convert to magnitude for penalty
        # Take absolute value since both left and right yaw changes should be penalized
        R_yaw = abs(yaw_rate)

        # R_steer: steering angle magnitude (should be penalized for large values)
        steering_component = self._info_value(info, ["steering_penalty", "steering_reward", "steering_cost", "steering"])
        if steering_component == 0.0:
            steering_component = float(info.get("steering", 0.0))
        # steering is normalized [-1, 1], take absolute value for penalty
        R_steer = abs(steering_component)

        R_term = 0.0
        if info.get("arrive_dest"):
            R_term += 10.0
        if info.get("out_of_road"):
            R_term -= 5.0
        try:
            from metadrive.constants import TerminationState  # type: ignore

            if info.get(TerminationState.SUCCESS):
                R_term += 10.0
            if info.get(TerminationState.OUT_OF_ROAD):
                R_term -= 5.0
        except Exception:
            pass

        components = {
            "R_dis": R_dis,
            "R_speed": R_speed,
            "R_yaw": R_yaw,
            "R_steer": R_steer,
            "R_term": R_term,
        }
        self._last_reward_components = components
        return c1 * R_dis + c2 * R_speed + c3 * R_yaw + c4 * R_steer + R_term

    @staticmethod
    def _info_value(info: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
        for key in keys:
            if key in info and info[key] is not None:
                value = info[key]
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return default

    def _augment_info_with_state(self, info: Dict[str, Any], obs_vec: np.ndarray) -> None:
        """
        Fill commonly used fields (speed, heading_error, lane_offset, yaw_rate) from the observation
        so downstream code does not depend on MetaDrive-specific info keys.
        """
        if obs_vec.shape[0] < self._lidar_dim + 4:
            return
        state_part = obs_vec[self._lidar_dim :]
        try:
            heading_error = float(state_part[2])
            speed_norm = float(state_part[3])
            steering_norm = float(state_part[4])
            yaw_rate_norm = float(state_part[7])
            lane_offset_norm = float(state_part[8]) if len(state_part) > 8 else 0.5
        except (IndexError, ValueError):
            return

        # Approximate conversions back to physical units
        max_speed_kmh = self._max_speed * 3.6
        speed_kmh = max(0.0, (speed_norm * (max_speed_kmh + 1.0)) - 1.0)
        lane_offset_m = (lane_offset_norm * 2.0 - 1.0) * 0.5 * self._lane_width

        info.setdefault("heading_error", heading_error)
        info.setdefault("heading_diff", heading_error)
        info.setdefault("speed", speed_kmh / 3.6)
        info.setdefault("velocity", speed_kmh / 3.6)
        info.setdefault("steering", steering_norm)
        info.setdefault("yaw_rate", yaw_rate_norm)
        info.setdefault("lane_offset", lane_offset_m)


def bootstrap_metadrive_config(credentials_cfg: Dict[str, Any]) -> None:
    """Create ~/.metadrive/config to avoid interactive prompts."""
    username = (
        credentials_cfg.get("github_username")
        if credentials_cfg
        else None
    ) or os.environ.get("METADRIVE_GITHUB_USERNAME")
    if not username:
        return

    home = Path.home()
    default_location = home / ".metadrive"
    config_path = default_location / "config"
    if config_path.exists():
        return

    default_location.mkdir(parents=True, exist_ok=True)
    config = configparser.ConfigParser()
    config["GITHUB"] = {"USERNAME": username}
    config["PROXIES"] = {"http": "", "https": ""}
    config["DRIVERS"] = {"auto_upgrade": "False"}
    config["SELENIUM"] = {"headless": "False"}
    config["DRIVER_BACKENDS"] = {"CHROME": "/usr/bin/chromedriver"}

    with config_path.open("w", encoding="utf-8") as handle:
        config.write(handle)
