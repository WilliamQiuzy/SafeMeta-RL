"""Minimal trajectory plotting for PPO on a single ramp scene with MetaDrive top-down rendering."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from safemeta_rl.agents import PPOAgent
from safemeta_rl.envs.metadrive_wrapper import LSTCMetaDriveEnv
from safemeta_rl.networks.policy import GaussianPolicy
from safemeta_rl.networks.value import ValueNetwork
from safemeta_rl.networks.cost_value import build_cost_value_network


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PPO trajectories on a ramp scene with MetaDrive top-down rendering.")
    parser.add_argument("--config", type=Path, default=Path("configs/ppo.yaml"), help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to PPO checkpoint.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to sample.")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum steps per episode.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/figures/ppo_ramp.png"), help="Output PNG path.")
    parser.add_argument("--use-official-render", action="store_true", default=True,
                        help="Use MetaDrive official top-down rendering for background.")
    parser.add_argument("--no-official-render", dest="use_official_render", action="store_false",
                        help="Use custom matplotlib rendering for background.")
    parser.add_argument("--random-policy", action="store_true", default=False,
                        help="Use random policy instead of trained agent for trajectory collection.")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_agent(config: Dict[str, Any], checkpoint: Path, device: torch.device):
    obs_dim = config["environment"].get("observation_dim", 49)
    action_dim = config["environment"].get("action_dim", 2)
    hidden = [64, 64]
    policy = GaussianPolicy(obs_dim, action_dim, hidden).to(device)
    value_net = ValueNetwork(obs_dim, hidden).to(device)
    cost_value_net = build_cost_value_network(obs_dim, hidden).to(device)
    agent = PPOAgent(policy, value_net, cost_value_net, config)

    state = torch.load(checkpoint, map_location=device)
    if "policy" in state:
        policy.load_state_dict(state["policy"])
    if "value_net" in state:
        value_net.load_state_dict(state["value_net"])
    if "cost_value_net" in state:
        cost_value_net.load_state_dict(state["cost_value_net"])
    return agent


def collect_multiple_trajectories(
    env: LSTCMetaDriveEnv,
    agent,
    episodes: int,
    max_steps: int,
    device: torch.device,
    use_random_policy: bool = False,
    lane_y_offset: float = 0.0,
) -> List[np.ndarray]:
    """Collect trajectories from multiple episodes.

    Args:
        use_random_policy: If True, use random actions with forward bias for more
                          natural-looking trajectories on unfamiliar maps.
        lane_y_offset: Y offset to apply to vehicle spawn position (for multi-lane collection).

    Returns:
        trajs: List of trajectory arrays, each is (N x 2)
    """
    trajs: List[np.ndarray] = []

    for ep in range(episodes):
        obs = env.reset()

        # Apply lane offset by moving the vehicle
        if lane_y_offset != 0.0:
            vehicle = env._current_vehicle()
            if vehicle is not None:
                current_pos = vehicle.position
                # MetaDrive's set_position takes (x, y) in world coords
                new_pos = (float(current_pos[0]), float(current_pos[1]) + lane_y_offset)
                vehicle.set_position(new_pos)
                # Note: observation may be slightly stale, but that's OK for first step

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        done = False
        steps = 0
        positions: List[np.ndarray] = []

        vehicle = env._current_vehicle()
        start_pos = None
        if vehicle is not None:
            pos = env._extract_position(vehicle)
            if pos is not None and len(pos) >= 2:
                positions.append(pos[:2].copy())
                start_pos = pos[:2].copy()

        # Keep running until crash or out of road
        # Use a high step limit as safety (env should terminate on crash/out-of-road)
        max_safety_steps = 5000
        while not done and steps < max_safety_steps:
            if use_random_policy:
                # Random policy with forward bias and lane-following tendency
                # This produces more natural-looking trajectories that follow the road
                throttle = np.random.uniform(0.4, 0.9)  # Strong forward bias
                # Smaller steering range with slight rightward bias (to merge onto main road)
                steering = np.random.uniform(-0.15, 0.25)
                action = np.array([steering, throttle])
            else:
                with torch.no_grad():
                    action = agent.act(obs_t.unsqueeze(0)).action.squeeze(0).cpu().numpy()

            result = env.step(action)
            vehicle = env._current_vehicle()
            if vehicle is not None:
                pos = env._extract_position(vehicle)
                if pos is not None and len(pos) >= 2:
                    positions.append(pos[:2].copy())
            obs_t = torch.tensor(result.obs, dtype=torch.float32, device=device)
            done = bool(result.terminated or result.truncated)
            steps += 1

        if positions:
            trajs.append(np.stack(positions, axis=0))
            print(f"Episode {ep + 1}/{episodes}: {len(positions)} steps, "
                  f"start=({positions[0][0]:.1f}, {positions[0][1]:.1f}), "
                  f"end=({positions[-1][0]:.1f}, {positions[-1][1]:.1f})")

    return trajs


def _draw_dashed_line(draw, points, color, width=1, dash_len=6, gap_len=6):
    """Draw a dashed line through a list of points."""
    if len(points) < 2:
        return

    # Calculate total path length and create segments
    import math
    total_len = 0
    segments = []
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        seg_len = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if seg_len > 0:
            segments.append((x1, y1, x2, y2, seg_len))
            total_len += seg_len

    # Draw dashes along the path
    current_pos = 0
    drawing = True
    seg_idx = 0
    seg_pos = 0

    while seg_idx < len(segments):
        x1, y1, x2, y2, seg_len = segments[seg_idx]

        # Calculate direction
        dx = (x2 - x1) / seg_len
        dy = (y2 - y1) / seg_len

        # Process this segment
        while seg_pos < seg_len:
            # Determine how much to draw/skip
            if drawing:
                draw_end = min(seg_pos + dash_len, seg_len)
                # Draw from seg_pos to draw_end
                sx = x1 + dx * seg_pos
                sy = y1 + dy * seg_pos
                ex = x1 + dx * draw_end
                ey = y1 + dy * draw_end
                draw.line([(int(sx), int(sy)), (int(ex), int(ey))], fill=color, width=width)
                seg_pos = draw_end
                if draw_end - seg_pos + dash_len <= dash_len:
                    drawing = False
                    seg_pos += gap_len - (seg_len - draw_end) if draw_end == seg_len else 0
            else:
                seg_pos += gap_len
                drawing = True

            if seg_pos >= seg_len:
                break

        seg_pos -= seg_len
        seg_idx += 1


def render_map_with_all_trajectories(
    env: LSTCMetaDriveEnv,
    trajs: List[np.ndarray],
    output: Path,
) -> None:
    """Render the map using MetaDrive's official top-down renderer and overlay trajectories.

    Uses the official pygame-based top-down renderer for accurate road visualization.
    """
    from PIL import Image, ImageDraw
    import pygame

    if not trajs:
        print("No trajectories to render")
        return

    # Calculate bounds across ALL trajectories
    all_points = np.vstack(trajs)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()

    print(f"\n=== Rendering {len(trajs)} trajectories ===")
    print(f"Trajectory bounds: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]")

    # The renderer centers on the current vehicle position.
    # We need to reset the env to get the vehicle back to the start position
    # so the render is centered on the trajectory start.
    env.reset()
    vehicle = env._current_vehicle()
    if vehicle is not None:
        vehicle_pos = env._extract_position(vehicle)
        vx, vy = float(vehicle_pos[0]), float(vehicle_pos[1])
    else:
        # Use first trajectory start as center
        vx, vy = float(trajs[0][0, 0]), float(trajs[0][0, 1])
    print(f"Vehicle/render center: ({vx:.1f}, {vy:.1f})")

    # Render using official top-down mode
    # The renderer centers on the vehicle position
    surface = env._env.render(mode='topdown')
    screen_width, screen_height = surface.get_size()
    print(f"Official render size: {screen_width}x{screen_height}")

    # Convert pygame surface to numpy array
    arr = pygame.surfarray.array3d(surface)
    # Pygame uses (width, height, channels), transpose to (height, width, channels)
    arr = arr.transpose(1, 0, 2)

    # Convert to PIL Image for drawing trajectories
    pil_image = Image.fromarray(arr)
    draw = ImageDraw.Draw(pil_image)

    # Find the vehicle's pixel location to calibrate the world-to-pixel mapping
    # The vehicle is rendered as a teal rectangle
    teal_pixels = []
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            r, g, b = arr[y, x]
            if g > 100 and b > 100 and r < 100 and abs(int(g) - int(b)) < 50:
                teal_pixels.append((x, y))

    if teal_pixels:
        xs = [p[0] for p in teal_pixels]
        ys = [p[1] for p in teal_pixels]
        vehicle_px = (min(xs) + max(xs)) / 2
        vehicle_py = (min(ys) + max(ys)) / 2
        print(f"Vehicle pixel position: ({vehicle_px:.0f}, {vehicle_py:.0f})")
    else:
        # Fallback: assume vehicle at screen center
        vehicle_px, vehicle_py = screen_width / 2, screen_height / 2
        print(f"Vehicle pixel position (fallback): ({vehicle_px:.0f}, {vehicle_py:.0f})")

    # Calculate scaling from known vehicle position
    # Vehicle is at world (vx, vy) and pixel (vehicle_px, vehicle_py)
    # MetaDrive's default top-down scaling is ~5 pixels per meter
    # But we need to match the actual rendered image scaling
    # From observation: road appears to be ~60 pixels wide for 3 lanes (~11m)
    # So actual scaling is approximately 60/11 ≈ 5.5 pixels/meter
    scaling = 5.0  # Match MetaDrive's default top-down scaling

    def world_to_pixel(wx: float, wy: float) -> tuple:
        # Use vehicle position as reference point
        # World coords: vehicle at (vx, vy), target at (wx, wy)
        # Pixel coords: vehicle at (vehicle_px, vehicle_py)
        # In MetaDrive top-down: +x is right, +y is DOWN (same as screen coords)
        px = int(vehicle_px + (wx - vx) * scaling)
        py = int(vehicle_py + (wy - vy) * scaling)  # Same direction as screen
        return (px, py)

    # Draw all trajectories with colored lines per lane
    # Use different colors for each lane for clarity
    lane_colors = [
        (255, 100, 100),  # Red for lane 0 (bottom)
        (100, 255, 100),  # Green for lane 1 (middle)
        (100, 100, 255),  # Blue for lane 2 (top)
    ]

    for i, traj in enumerate(trajs):
        pts = [world_to_pixel(p[0], p[1]) for p in traj]

        # Determine lane color based on starting Y position
        # Y coordinates: Lane 0 ~ 0, Lane 1 ~ 2.5, Lane 2 ~ 5.0
        start_y = traj[0, 1] if len(traj) > 0 else 0
        if start_y < 1.5:
            color = lane_colors[0]  # Lane 0 (y ~ 0) - RED
        elif start_y < 4.0:
            color = lane_colors[1]  # Lane 1 (y ~ 2.5) - GREEN
        else:
            color = lane_colors[2]  # Lane 2 (y ~ 5.0) - BLUE

        # Draw trajectory with thicker line
        if len(pts) > 1:
            draw.line(pts, fill=color, width=3)

        # Mark termination with black dot
        if len(pts) > 0:
            ex, ey = pts[-1]
            draw.ellipse([ex - 4, ey - 4, ex + 4, ey + 4], fill=(0, 0, 0))

    # Save
    output.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(str(output))
    print(f"Saved {len(trajs)} trajectories to {output}")


def _capture_topdown_frame(env: LSTCMetaDriveEnv) -> tuple[Optional[np.ndarray], Optional[tuple]]:
    """Capture a top-down rendered frame from MetaDrive's official renderer.

    The top-down renderer centers on the vehicle position. We calculate the world
    bounds based on the vehicle position, screen size, and scaling factor.

    Returns:
        frame: RGB image (H x W x 3) or None
        world_bounds: (x_min, x_max, y_min, y_max) or None
    """
    # Get vehicle position (this is the center of the rendered view)
    vehicle = env._current_vehicle()
    vehicle_pos = env._extract_position(vehicle)
    if vehicle_pos is None:
        vehicle_pos = np.array([0.0, 0.0])

    # Rendering parameters - MetaDrive's top-down renderer uses these
    # Higher scaling = more zoomed in (more pixels per world unit)
    screen_size = (1200, 1200)
    scaling = 50.0  # pixels per world unit - zoomed in to show trajectory detail

    # Calculate world bounds based on vehicle position and scaling
    # The view is centered on the vehicle
    half_width_world = screen_size[0] / (2 * scaling)
    half_height_world = screen_size[1] / (2 * scaling)

    cx, cy = float(vehicle_pos[0]), float(vehicle_pos[1])
    world_bounds = (
        cx - half_width_world,   # x_min
        cx + half_width_world,   # x_max
        cy - half_height_world,  # y_min
        cy + half_height_world,  # y_max
    )

    try:
        # Use MetaDrive's built-in top-down rendering (centered on vehicle)
        frame = env._env.render(
            mode="topdown",
            window=False,
            screen_size=screen_size,
            draw_target_vehicle_trajectory=False,
            scaling=scaling,
        )
        if frame is not None and isinstance(frame, np.ndarray):
            print(f"Captured top-down frame: {frame.shape}")
            print(f"Vehicle position: ({cx:.2f}, {cy:.2f})")
            print(f"World bounds: x=[{world_bounds[0]:.2f}, {world_bounds[1]:.2f}], y=[{world_bounds[2]:.2f}, {world_bounds[3]:.2f}]")
            return frame, world_bounds
    except Exception as e:
        print(f"Warning: Failed to capture top-down frame via render(): {e}")

    # Fallback: try alternative API for older MetaDrive versions
    try:
        if hasattr(env._env, "top_down_renderer"):
            renderer = env._env.top_down_renderer
            if renderer is not None and hasattr(renderer, "get_map_image"):
                frame = renderer.get_map_image()
                if frame is not None:
                    return frame, world_bounds
    except Exception as e:
        print(f"Warning: Fallback top-down capture also failed: {e}")

    return None, None


def extract_lane_shapes(env: LSTCMetaDriveEnv) -> List[Dict[str, Any]]:
    """Pull lane polygons and centerlines for simple top-down plotting."""
    lanes: List[Dict[str, Any]] = []
    try:
        road_network = env._env.current_map.road_network  # type: ignore[attr-defined]
        for outgoing in road_network.graph.values():
            if not isinstance(outgoing, dict):
                continue
            for lane_list in outgoing.values():
                if isinstance(lane_list, list):
                    iter_lanes = lane_list
                else:
                    iter_lanes = [lane_list]
                for lane in iter_lanes:
                    length = float(getattr(lane, "length", 0.0) or 0.0)
                    poly = np.asarray(getattr(lane, "polygon", []), dtype=np.float32)
                    if length <= 0 or poly.size == 0:
                        continue
                    samples = max(int(length // 2) + 2, 2)
                    centerline = np.asarray([lane.position(s, 0)[:2] for s in np.linspace(0, length, samples)], dtype=np.float32)
                    lanes.append(
                        {
                            "polygon": poly,
                            "center": centerline,
                            "line_types": list(getattr(lane, "line_types", [])),
                        }
                    )
    except Exception:
        pass
    return lanes


def _line_style(line_type: str) -> Dict[str, Any]:
    if "YELLOW" in line_type:
        color = "#f5b544"
    else:
        color = "#c8c8c8"
    if "BROKEN" in line_type:
        return {"color": color, "linewidth": 1.4, "dashes": (6, 8), "alpha": 0.9}
    return {"color": color, "linewidth": 1.8, "alpha": 0.9}


def plot_trajs(trajs: List[np.ndarray], lanes: List[Dict[str, Any]], output: Path) -> None:
    """Plot trajectories with custom matplotlib-drawn background (fallback)."""
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.set_title("PPO ramp trajectories")
    ax.set_axis_off()
    xs: List[float] = []
    ys: List[float] = []
    # Draw road lanes (polygons + markings)
    for lane in lanes:
        poly = lane.get("polygon")
        if poly is None or poly.size == 0:
            continue
        # road surface
        ax.fill(poly[:, 0], poly[:, 1], color="#f0f0f0", alpha=1.0, zorder=0)
        n = poly.shape[0] // 2
        lower = poly[:n]
        upper = poly[n:]
        line_types = lane.get("line_types", [])
        left_style = _line_style(line_types[0]) if len(line_types) > 0 else {"color": "#b3b3b3", "linewidth": 1.2}
        right_style = _line_style(line_types[1]) if len(line_types) > 1 else {"color": "#b3b3b3", "linewidth": 1.2}
        ax.plot(lower[:, 0], lower[:, 1], zorder=1, **left_style)
        ax.plot(upper[:, 0], upper[:, 1], zorder=1, **right_style)
        center = lane.get("center")
        if center is not None and center.size > 0:
            ax.plot(center[:, 0], center[:, 1], color="#d6d6d6", linewidth=1.0, alpha=0.65, zorder=1)
    for traj in trajs:
        if traj.size == 0:
            continue
        ax.plot(traj[:, 0], traj[:, 1], color="#c187d2", alpha=0.75, linewidth=1.3, zorder=2)
        # sprinkle dark markers along the path to mimic traffic footprints
        if traj.shape[0] > 0:
            indices = np.linspace(0, traj.shape[0] - 1, num=min(traj.shape[0], 80)).astype(int)
            ax.scatter(traj[indices, 0], traj[indices, 1], color="#1a1a1a", s=6, alpha=0.6, zorder=3)
        xs.extend(traj[:, 0].tolist())
        ys.extend(traj[:, 1].tolist())
    if xs and ys:
        pad_x = (max(xs) - min(xs)) * 0.05 + 1e-3
        pad_y = (max(ys) - min(ys)) * 0.05 + 1e-3
        ax.set_xlim([min(xs) - pad_x, max(xs) + pad_x])
        ax.set_ylim([min(ys) - pad_y, max(ys) + pad_y])
        ax.set_aspect("equal", adjustable="box")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"Saved PPO ramp trajectories to {output}")


def plot_trajs_with_official_background(
    trajs: List[np.ndarray],
    topdown_frame: np.ndarray,
    output: Path,
    world_bounds: Optional[tuple] = None,
) -> None:
    """Plot trajectories overlaid on MetaDrive's official top-down rendered background.

    Args:
        trajs: List of trajectory arrays (N x 2) in world coordinates
        topdown_frame: RGB image from MetaDrive's top-down renderer (H x W x 3)
        output: Output path for the figure
        world_bounds: (x_min, x_max, y_min, y_max) for coordinate mapping
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title("PPO ramp trajectories (MetaDrive official rendering)")
    ax.set_axis_off()

    if world_bounds is None:
        print("Warning: No world_bounds provided, cannot map trajectories correctly")
        ax.imshow(topdown_frame, origin="upper")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output), dpi=300)
        plt.close(fig)
        return

    x_min, x_max, y_min, y_max = world_bounds

    # Display the image with world coordinate extent
    # MetaDrive's top-down renderer has Y increasing downward in image space,
    # but world coordinates have Y increasing upward
    # Using origin="upper" and flipping y extent handles this
    ax.imshow(topdown_frame, origin="upper", extent=[x_min, x_max, y_min, y_max], zorder=0)

    # Print debug info
    print(f"Trajectory bounds:")
    for i, traj in enumerate(trajs):
        if traj.size > 0:
            print(f"  Traj {i}: x=[{traj[:, 0].min():.2f}, {traj[:, 0].max():.2f}], y=[{traj[:, 1].min():.2f}, {traj[:, 1].max():.2f}]")

    # Draw trajectories directly in world coordinates
    for traj in trajs:
        if traj.size == 0:
            continue
        # Plot in world coordinates - imshow extent handles the mapping
        ax.plot(traj[:, 0], traj[:, 1], color="#ff4081", alpha=0.9, linewidth=2.5, zorder=2)
        # Markers along the path
        if traj.shape[0] > 0:
            indices = np.linspace(0, traj.shape[0] - 1, num=min(traj.shape[0], 60)).astype(int)
            ax.scatter(
                traj[indices, 0], traj[indices, 1],
                color="#ffffff", s=20, alpha=0.95, zorder=3,
                edgecolors="#333333", linewidths=0.8
            )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PPO ramp trajectories with official background to {output}")


def _extract_lanes_from_env(env: LSTCMetaDriveEnv) -> List[Dict[str, Any]]:
    """Extract lane geometry from the environment's road network.

    Since MetaDrive lanes don't have polygon attribute directly, we construct
    polygons from the lane's position() method and width.
    """
    lanes: List[Dict[str, Any]] = []
    road_network = env._env.current_map.road_network  # type: ignore[attr-defined]
    for outgoing in road_network.graph.values():
        if not isinstance(outgoing, dict):
            continue
        for lane_list in outgoing.values():
            if isinstance(lane_list, list):
                iter_lanes = lane_list
            else:
                iter_lanes = [lane_list]
            for lane in iter_lanes:
                length = float(getattr(lane, "length", 0.0) or 0.0)
                if length <= 0:
                    continue

                # Get lane width
                width = float(getattr(lane, "width", getattr(lane, "DEFAULT_WIDTH", 3.5)))
                half_width = width / 2

                # Sample points along the lane
                samples = max(int(length // 2) + 2, 10)
                centerline = []
                left_edge = []
                right_edge = []

                for s in np.linspace(0, length, samples):
                    # Center position
                    center = lane.position(s, 0)[:2]
                    centerline.append(list(center))
                    # Left and right edges (lateral offset)
                    left = lane.position(s, half_width)[:2]
                    right = lane.position(s, -half_width)[:2]
                    left_edge.append(list(left))
                    right_edge.append(list(right))

                # Create polygon: left edge forward, then right edge backward
                polygon = left_edge + right_edge[::-1]

                lanes.append(
                    {
                        "polygon": polygon,
                        "center": centerline,
                        "line_types": list(getattr(lane, "line_types", [])),
                        "left_edge": left_edge,
                        "right_edge": right_edge,
                    }
                )
    return lanes


def save_frame_directly(frame: np.ndarray, output: Path) -> None:
    """Save the rendered frame directly as an image."""
    from PIL import Image
    output.parent.mkdir(parents=True, exist_ok=True)
    pil_image = Image.fromarray(frame)
    pil_image.save(str(output))
    print(f"Saved trajectory visualization to {output}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Force headless for rollout
    env_cfg = dict(cfg["environment"])
    md_cfg = dict(env_cfg.get("metadrive_config", {}))
    md_cfg["use_render"] = False
    md_cfg["offscreen_render"] = False  # Disable to avoid display issues
    md_cfg["headless_machine_render"] = True
    # Disable traffic and obstacles for cleaner visualization
    md_cfg["traffic_density"] = 0.0
    md_cfg["accident_prob"] = 0.0  # No obstacles
    # Use InRamp map: "r" = InRampOnStraight (vehicle spawns on ramp, merges onto straight)
    # Followed by Straight to give room to merge
    # Use InRamp map for ramp merge visualization
    md_cfg["map"] = "rS"  # InRamp first, then Straight
    # Remove num_scenarios and set environment_num (0.2.x naming)
    md_cfg.pop("num_scenarios", None)
    md_cfg["environment_num"] = 1  # Single fixed scenario
    md_cfg["start_seed"] = 0  # Fixed seed for reproducibility

    # Spawn on main road (not ramp) to test safety during merge
    # Main road has 3 lanes. Vehicle spawns at lane center.
    # Lane 0 (red) is correct - near yellow center line
    # Lane 1 and Lane 2 should be BELOW Lane 0 (toward the ramp)
    # Positive Y offset moves toward the ramp (downward in render)
    lane_width = 2.5  # Adjusted lane width for tighter spacing
    lane_offsets = [
        0.0,                 # Lane 0 - near yellow line (original spawn) - RED
        lane_width,          # Lane 1 - middle lane (below lane 0) - GREEN
        lane_width * 2,      # Lane 2 - outer lane (near ramp merge) - BLUE
    ]

    agent = build_agent(cfg, args.checkpoint, device)

    # Spawn at lane 0 initially
    md_cfg["target_vehicle_configs"] = {
        "default_agent": {
            "spawn_lane_index": (">", ">>", 0),
        }
    }
    env_cfg["metadrive_config"] = md_cfg

    env = LSTCMetaDriveEnv(env_cfg)
    env.reset()

    # Collect trajectories from each lane by offsetting Y position
    all_trajs: List[np.ndarray] = []
    episodes_per_lane = max(1, args.episodes // 3)  # Split episodes across 3 lanes

    for lane_idx, y_offset in enumerate(lane_offsets):
        print(f"\n=== Collecting from lane {lane_idx} (y_offset={y_offset:.1f}m) ===")

        # Collect trajectories for this lane
        lane_trajs = collect_multiple_trajectories(
            env,
            agent,
            episodes=episodes_per_lane,
            max_steps=args.max_steps,
            device=device,
            use_random_policy=args.random_policy,
            lane_y_offset=y_offset,
        )
        all_trajs.extend(lane_trajs)

    trajs = all_trajs

    if not trajs:
        print("No trajectories collected!")
        return

    print(f"\nCollected {len(trajs)} trajectories from all 3 lanes")

    # Save trajectory data to JSON for later analysis
    traj_data = {"trajectories": [traj.tolist() for traj in trajs]}
    traj_path = args.output.parent / "ppo_positions.json"
    with open(traj_path, "w") as f:
        json.dump(traj_data, f)
    print(f"Saved trajectory data to {traj_path}")

    # Use existing env for rendering (already spawned at lane 0)
    if args.use_official_render:
        # Step 2: Render map with all trajectories overlaid
        # Reset to get a fresh environment state for rendering
        env.reset()
        render_map_with_all_trajectories(env, trajs, args.output)
    else:
        # Fallback to custom matplotlib rendering
        lanes_raw = _extract_lanes_from_env(env)
        lanes = [
            {
                "polygon": np.asarray(item.get("polygon", []), dtype=np.float32),
                "center": np.asarray(item.get("center", []), dtype=np.float32),
                "line_types": item.get("line_types", []),
            }
            for item in lanes_raw
        ]
        # Cache lane data for reproducibility
        cache_path = Path("artifacts/maps/ramp_map.json")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump({"lanes": lanes_raw}, f)
        plot_trajs(trajs, lanes, args.output)

    env.close()


if __name__ == "__main__":
    main()
