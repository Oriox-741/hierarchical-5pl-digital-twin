#!/usr/bin/env python
"""
Read-only PPO sanity diagnostic for the joint 5PL Meta-Brain.

This script DOES NOT attach to, modify, pause, or inspect the live training process.
It only loads a checkpoint from disk and runs deterministic PPO inference on
synthetic stress-test observations.

Run example:
    python ppo_in_situ_sanity_check.py --checkpoint-path models/checkpoints/joint_torch/joint_torch_latest.pt
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import numpy as np
import torch

from src.act.action_projector import ActionProjector
from src.act.observation_builder import OBSERVATION_DIM
from src.think.joint_policies import JointPolicyBundle


ACTION_NAMES: Final[tuple[str, ...]] = (
    "reorder_fraction",
    "dispatch_intensity",
    "speed_multiplier",
    "safety_stock_multiplier",
    "capacity_buffer_fraction",
)

DEFAULT_CHECKPOINT_PATH: Final[Path] = Path("models/checkpoints/joint_torch/joint_torch_latest.pt")


# Observation index map from src/act/observation_builder.py
IDX = {
    "time": 0,
    "pending_orders": 1,
    "delivered_orders": 2,
    "late_orders": 3,
    "failed_orders": 4,
    "service_level": 5,
    "transport_cost": 6,
    "inventory_on_hand": 7,
    "inventory_reserved": 8,
    "inventory_in_transit": 9,
    "inventory_backlog": 10,
    "safety_stock": 11,
    "demand_mean": 12,
    "demand_var": 13,
    "lead_mean": 14,
    "lead_var": 15,
    "vehicle_util": 16,
    "active_vehicle_ratio": 17,
    "capacity_pressure": 18,
    "disruption_score": 19,
    "network_safety_potential": 20,
    "hub_count": 21,
    "customer_count": 22,
    "arc_count": 23,
    "db_asset_count": 24,
    "db_avg_speed": 25,
    "db_avg_battery": 26,
    "db_sensor_quality": 27,
    "db_safety_potential": 28,
    "db_disruption_score": 29,
    "db_congestion_score": 30,
    "bias": 31,
}


def base_observation() -> np.ndarray:
    obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
    obs[IDX["time"]] = 0.50
    obs[IDX["pending_orders"]] = 0.25
    obs[IDX["delivered_orders"]] = 0.70
    obs[IDX["late_orders"]] = 0.04
    obs[IDX["failed_orders"]] = 0.00
    obs[IDX["service_level"]] = 0.94
    obs[IDX["transport_cost"]] = 0.08
    obs[IDX["inventory_on_hand"]] = 0.55
    obs[IDX["inventory_reserved"]] = 0.10
    obs[IDX["inventory_in_transit"]] = 0.10
    obs[IDX["inventory_backlog"]] = 0.05
    obs[IDX["safety_stock"]] = 0.20
    obs[IDX["demand_mean"]] = 0.35
    obs[IDX["demand_var"]] = 0.18
    obs[IDX["lead_mean"]] = 0.12
    obs[IDX["lead_var"]] = 0.08
    obs[IDX["vehicle_util"]] = 0.45
    obs[IDX["active_vehicle_ratio"]] = 0.90
    obs[IDX["capacity_pressure"]] = 0.35
    obs[IDX["disruption_score"]] = 0.10
    obs[IDX["network_safety_potential"]] = 0.15
    obs[IDX["hub_count"]] = 0.05
    obs[IDX["customer_count"]] = 0.02
    obs[IDX["arc_count"]] = 0.01
    obs[IDX["db_asset_count"]] = 0.02
    obs[IDX["db_avg_speed"]] = 0.45
    obs[IDX["db_avg_battery"]] = 0.80
    obs[IDX["db_sensor_quality"]] = 0.95
    obs[IDX["db_safety_potential"]] = 0.15
    obs[IDX["db_disruption_score"]] = 0.10
    obs[IDX["db_congestion_score"]] = 0.20
    obs[IDX["bias"]] = 1.00
    return obs


def synthetic_observations() -> dict[str, np.ndarray]:
    calm = base_observation()
    calm[IDX["pending_orders"]] = 0.08
    calm[IDX["service_level"]] = 0.99
    calm[IDX["transport_cost"]] = 0.03
    calm[IDX["inventory_backlog"]] = 0.01
    calm[IDX["vehicle_util"]] = 0.25
    calm[IDX["capacity_pressure"]] = 0.15
    calm[IDX["disruption_score"]] = 0.02
    calm[IDX["network_safety_potential"]] = 0.05

    backlog_crisis = base_observation()
    backlog_crisis[IDX["pending_orders"]] = 0.92
    backlog_crisis[IDX["service_level"]] = 0.62
    backlog_crisis[IDX["inventory_on_hand"]] = 0.12
    backlog_crisis[IDX["inventory_backlog"]] = 0.72
    backlog_crisis[IDX["demand_mean"]] = 0.82
    backlog_crisis[IDX["demand_var"]] = 0.76
    backlog_crisis[IDX["vehicle_util"]] = 0.88
    backlog_crisis[IDX["capacity_pressure"]] = 0.82

    capacity_crisis = base_observation()
    capacity_crisis[IDX["pending_orders"]] = 0.18
    capacity_crisis[IDX["service_level"]] = 0.93
    capacity_crisis[IDX["inventory_backlog"]] = 0.08
    capacity_crisis[IDX["vehicle_util"]] = 0.96
    capacity_crisis[IDX["active_vehicle_ratio"]] = 0.55
    capacity_crisis[IDX["capacity_pressure"]] = 0.97
    capacity_crisis[IDX["db_congestion_score"]] = 0.80

    disruption = base_observation()
    disruption[IDX["pending_orders"]] = 0.55
    disruption[IDX["late_orders"]] = 0.35
    disruption[IDX["failed_orders"]] = 0.08
    disruption[IDX["service_level"]] = 0.48
    disruption[IDX["transport_cost"]] = 0.35
    disruption[IDX["disruption_score"]] = 0.95
    disruption[IDX["network_safety_potential"]] = 0.88
    disruption[IDX["db_disruption_score"]] = 0.92
    disruption[IDX["db_congestion_score"]] = 0.75

    # Replace this vector with one copied from a real trace if you want exact replay.
    recent_real = np.array(
        [
            0.9826, 0.3100, 0.6900, 0.0200, 0.0000, 0.9800, 0.0700, 0.4800,
            0.0900, 0.1400, 0.0600, 0.2300, 0.4200, 0.2200, 0.1300, 0.0900,
            0.6200, 0.9200, 0.4100, 0.1200, 0.1800, 0.0500, 0.0200, 0.0100,
            0.0200, 0.5000, 0.7800, 0.9600, 0.1600, 0.1100, 0.2400, 1.0000,
        ],
        dtype=np.float32,
    )

    return {
        "Calm": calm,
        "Backlog_Crisis": backlog_crisis,
        "Capacity_Crisis": capacity_crisis,
        "Disruption": disruption,
        "Recent_Real": recent_real,
    }


def load_policy(checkpoint_path: Path, device: torch.device) -> tuple[JointPolicyBundle, Mapping[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint).__name__}.")

    policy = JointPolicyBundle().to(device)
    policy.load_checkpoint_state(checkpoint)
    policy.eval()
    return policy, checkpoint


@torch.no_grad()
def infer_ppo_actions(
    policy: JointPolicyBundle,
    observations: Mapping[str, np.ndarray],
    device: torch.device,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = list(observations.keys())
    batch = np.stack([observations[name] for name in names]).astype(np.float32)
    tensor = torch.as_tensor(batch, dtype=torch.float32, device=device)

    output = policy.select_ppo_action(tensor, deterministic=True)
    raw_actions = output.action.detach().cpu().numpy().astype(np.float64)

    projector = ActionProjector()
    projected = []
    for raw in raw_actions:
        physical = projector.project(raw).as_array()
        projected.append(physical)

    projected_actions = np.asarray(projected, dtype=np.float64)
    return names, raw_actions, projected_actions


def print_table(names: list[str], raw_actions: np.ndarray, projected_actions: np.ndarray) -> None:
    print("\n# PPO In-Situ Sanity Check")
    print("\n## Raw Normalized PPO Actions [-1, 1]")
    header = "| State | " + " | ".join(ACTION_NAMES) + " |"
    sep = "|---|" + "|".join("---:" for _ in ACTION_NAMES) + "|"
    print(header)
    print(sep)
    for name, row in zip(names, raw_actions, strict=True):
        values = " | ".join(f"{value: .6f}" for value in row)
        print(f"| {name} | {values} |")

    print("\n## Projected Physical Actions")
    print(header)
    print(sep)
    for name, row in zip(names, projected_actions, strict=True):
        values = " | ".join(f"{value: .6f}" for value in row)
        print(f"| {name} | {values} |")

    raw_std = raw_actions.std(axis=0)
    projected_std = projected_actions.std(axis=0)

    print("\n## Per-Dimension Standard Deviation")
    print("| Dimension | Raw Std Dev | Projected Std Dev |")
    print("|---|---:|---:|")
    for action_name, raw_value, projected_value in zip(ACTION_NAMES, raw_std, projected_std, strict=True):
        print(f"| {action_name} | {raw_value:.8f} | {projected_value:.8f} |")

    print("\n## Diagnostic Interpretation")
    if float(np.max(projected_std)) < 1e-4:
        print("WARNING: projected actions are effectively constant across stress states.")
        print("This supports the AlwaysMax/AlwaysConstant collapse hypothesis.")
    elif float(np.max(raw_std)) < 1e-4:
        print("WARNING: raw policy outputs are effectively constant, but projection may still create small differences.")
        print("This suggests weak state dependence in the PPO actor.")
    else:
        print("OK: PPO outputs vary across synthetic stress states.")
        print("This argues against a pure AlwaysMax constant policy, though projection-rate analysis is still needed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only PPO checkpoint sanity diagnostic.")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to a joint/ppo .pt checkpoint produced by train_joint_torch.py.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Use CPU by default to avoid competing with the live trainer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint_path.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    policy, checkpoint = load_policy(checkpoint_path, device)
    observations = synthetic_observations()
    names, raw_actions, projected_actions = infer_ppo_actions(policy, observations, device)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Global step: {checkpoint.get('global_step', 'unknown')}")
    print(f"Artifact kind: {checkpoint.get('artifact_kind', 'unknown')}")
    print_table(names, raw_actions, projected_actions)


if __name__ == "__main__":
    main()