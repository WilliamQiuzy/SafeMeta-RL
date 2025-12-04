"""Safety heuristics used to classify feasible and infeasible states."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np


@dataclass
class SafetyThresholds:
    speed_limit: float = 22.22  # 80 km/h
    heading_limit_deg: float = 45.0
    lane_offset_limit: float = 1.5
    min_ttc: float = 1.5
    min_lidar_distance: float = 1.0
    lidar_num_bins: int = 30


class SafetyLabeler:
    """Implements Def. 1 rules (speed, TTC, heading deviation, etc.)."""

    def __init__(self, thresholds: Optional[Dict[str, Any]] = None):
        cfg = thresholds or {}
        self.thresholds = SafetyThresholds(
            speed_limit=cfg.get("speed_limit", SafetyThresholds.speed_limit),
            heading_limit_deg=cfg.get("heading_limit_deg", SafetyThresholds.heading_limit_deg),
            lane_offset_limit=cfg.get("lane_offset_limit", SafetyThresholds.lane_offset_limit),
            min_ttc=cfg.get("min_ttc", SafetyThresholds.min_ttc),
            min_lidar_distance=cfg.get("min_lidar_distance", SafetyThresholds.min_lidar_distance),
            lidar_num_bins=cfg.get("lidar_num_bins", SafetyThresholds.lidar_num_bins),
        )

    def classify(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> bool:
        """Return True if the state is considered feasible."""
        info = info or {}
        lidar_bins = obs[: self.thresholds.lidar_num_bins]
        min_distance = float(np.min(lidar_bins))
        lidar_safe = min_distance >= self.thresholds.min_lidar_distance

        speed = abs(float(info.get("speed", info.get("vehicle_speed", info.get("velocity", 0.0)))))
        speed_safe = speed <= self.thresholds.speed_limit

        heading_error = abs(float(info.get("heading_error", info.get("heading_diff", 0.0))))
        heading_safe = heading_error <= np.deg2rad(self.thresholds.heading_limit_deg)

        lane_offset = abs(float(info.get("lane_offset", info.get("lateral_distance", 0.0))))
        lane_safe = lane_offset <= self.thresholds.lane_offset_limit

        # TTC derived from lidar + ego speed
        ttc = self._compute_ttc(min_distance, speed)
        ttc_safe = ttc >= self.thresholds.min_ttc

        return lidar_safe and speed_safe and heading_safe and lane_safe and ttc_safe

    def _compute_ttc(self, distance: float, speed: float) -> float:
        eps = 1e-3
        if speed < eps:
            return float("inf")
        return distance / max(speed, eps)
