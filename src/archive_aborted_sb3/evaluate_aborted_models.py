"""Autopsy evaluation for the aborted staggered PPO+DQN joint-training checkpoints."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from stable_baselines3 import DQN, PPO

from src.act.action_projector import CONTINUOUS_ACTION_DIM
from src.act.discrete_action_mapper import DiscreteActionMapper
from src.act.env_5pl import EnvironmentConfig, FivePLDigitalTwinEnv


CONFIG_PATH = Path("configs/training_joint.json")
DEFAULT_EPISODES = 100
DEFAULT_SEED = 42
DEFAULT_STEP_LOG_PATH = Path("models/checkpoints/joint_from_scratch/aborted_autopsy_steps.jsonl")
PPO_LATEST = Path("models/checkpoints/joint_from_scratch/ppo_joint_latest.zip")
DQN_LATEST = Path("models/checkpoints/joint_from_scratch/dqn_joint_latest.zip")
PPO_PATTERN = "models/checkpoints/ppo/ppo_joint_from_scratch_*_steps.zip"
DQN_PATTERN = "models/checkpoints/dqn/dqn_joint_from_scratch_*_steps.zip"


@dataclass(slots=True)
class EpisodeSummary:
    episode: int
    seed: int
    steps: int
    total_reward: float
    global_reward: float
    ppo_local_reward: float
    dqn_local_reward: float
    service_level: float
    safety_potential: float
    transport_cost: float
    blocked_steps: int
    projected_steps: int
    terminated: bool
    truncated: bool


@dataclass(slots=True)
class AutopsyStats:
    episodes: list[EpisodeSummary] = field(default_factory=list)
    raw_discrete_actions: Counter[int] = field(default_factory=Counter)
    dispatch: Counter[str] = field(default_factory=Counter)
    route: Counter[str] = field(default_factory=Counter)
    mode: Counter[str] = field(default_factory=Counter)
    reorder: Counter[str] = field(default_factory=Counter)
    continuous_raw: list[np.ndarray] = field(default_factory=list)
    continuous_physical: list[dict[str, float]] = field(default_factory=list)
    blocked_reasons: Counter[str] = field(default_factory=Counter)
    projected_reasons: Counter[str] = field(default_factory=Counter)
    rolling_demand_orders: int = 0

    @property
    def total_steps(self) -> int:
        return sum(item.steps for item in self.episodes)


def main() -> None:
    args = parse_args()
    max_steps = int(args.max_steps) if args.max_steps is not None else _config_max_steps()
    ppo_path = cast(Path | None, args.ppo_path) or _resolve_checkpoint(PPO_LATEST, PPO_PATTERN, "PPO")
    dqn_path = cast(Path | None, args.dqn_path) or _resolve_checkpoint(DQN_LATEST, DQN_PATTERN, "DQN")

    ppo_model, dqn_model = load_models(ppo_path, dqn_path)
    stats = evaluate(
        ppo_model=ppo_model,
        dqn_model=dqn_model,
        ppo_path=ppo_path,
        dqn_path=dqn_path,
        episodes=int(args.episodes),
        seed=int(args.seed),
        max_steps=max_steps,
        step_log_path=None if bool(args.no_step_log) else cast(Path, args.step_log_path),
    )
    print_report(stats, ppo_path=ppo_path, dqn_path=dqn_path, max_steps=max_steps)


def load_models(ppo_path: Path, dqn_path: Path) -> tuple[PPO, DQN]:
    if not ppo_path.exists():
        raise FileNotFoundError(f"PPO checkpoint not found: {ppo_path}")
    if not dqn_path.exists():
        raise FileNotFoundError(f"DQN checkpoint not found: {dqn_path}")
    return PPO.load(ppo_path), DQN.load(dqn_path)


def evaluate(
    *,
    ppo_model: PPO,
    dqn_model: DQN,
    ppo_path: Path,
    dqn_path: Path,
    episodes: int,
    seed: int,
    max_steps: int,
    step_log_path: Path | None,
) -> AutopsyStats:
    stats = AutopsyStats()
    mapper = DiscreteActionMapper()
    log_handle = None
    if step_log_path is not None:
        step_log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = step_log_path.open("w", encoding="utf-8")

    try:
        for episode_index in range(episodes):
            episode_seed = seed + episode_index
            env = FivePLDigitalTwinEnv(
                EnvironmentConfig(action_mode="joint", random_seed=episode_seed, max_steps=max_steps)
            )
            try:
                observation, _ = env.reset(seed=episode_seed)
                terminated = False
                truncated = False
                episode_total = 0.0
                episode_global = 0.0
                episode_ppo = 0.0
                episode_dqn = 0.0
                blocked_steps = 0
                projected_steps = 0
                info: dict[str, Any] = {}

                while not terminated and not truncated:
                    obs = np.asarray(observation, dtype=np.float32)
                    ppo_action, _ = ppo_model.predict(obs, deterministic=True)
                    dqn_action, _ = dqn_model.predict(obs, deterministic=True)

                    ppo_vector = np.asarray(ppo_action, dtype=np.float32).reshape(-1)
                    if ppo_vector.shape[0] != CONTINUOUS_ACTION_DIM:
                        raise RuntimeError(f"PPO returned invalid action shape: {ppo_vector.shape}")
                    discrete_id = int(np.asarray(dqn_action).reshape(-1)[0])
                    decoded_discrete = mapper.map(discrete_id).as_dict()
                    projected_physical = env.action_projector.project(ppo_vector).as_dict()

                    observation, reward, terminated, truncated, info = env.step(
                        {"continuous": ppo_vector, "discrete": discrete_id}
                    )
                    components = _reward_components(info)
                    episode_total += float(reward)
                    episode_global += components.get("global", 0.0)
                    episode_ppo += components.get("ppo_local", 0.0)
                    episode_dqn += components.get("dqn_local", 0.0)

                    blocked = bool(info.get("blocked", False))
                    projected = bool(info.get("projected", False))
                    blocked_steps += int(blocked)
                    projected_steps += int(projected)
                    stats.raw_discrete_actions[discrete_id] += 1
                    stats.dispatch[decoded_discrete["dispatch"]] += 1
                    stats.route[decoded_discrete["route"]] += 1
                    stats.mode[decoded_discrete["mode"]] += 1
                    stats.reorder[decoded_discrete["reorder"]] += 1
                    stats.continuous_raw.append(ppo_vector.copy())
                    stats.continuous_physical.append(projected_physical)
                    stats.rolling_demand_orders += int(info.get("rolling_demand_orders", 0))
                    _count_reasons(info, stats.blocked_reasons, stats.projected_reasons)

                    if log_handle is not None:
                        _write_step_log(
                            log_handle,
                            episode=episode_index + 1,
                            seed=episode_seed,
                            step=int(info.get("step", env._step_count)),
                            reward=float(reward),
                            reward_components=components,
                            ppo_raw=ppo_vector,
                            ppo_physical=projected_physical,
                            dqn_action=discrete_id,
                            dqn_decoded=decoded_discrete,
                            info=info,
                            ppo_path=ppo_path,
                            dqn_path=dqn_path,
                        )

                snapshot = _snapshot(info, env.simulation.snapshot())
                summary = EpisodeSummary(
                    episode=episode_index + 1,
                    seed=episode_seed,
                    steps=env._step_count,
                    total_reward=episode_total,
                    global_reward=episode_global,
                    ppo_local_reward=episode_ppo,
                    dqn_local_reward=episode_dqn,
                    service_level=_float_snapshot(snapshot, "service_level"),
                    safety_potential=_float_snapshot(snapshot, "network_safety_potential"),
                    transport_cost=_float_snapshot(snapshot, "total_transport_cost"),
                    blocked_steps=blocked_steps,
                    projected_steps=projected_steps,
                    terminated=terminated,
                    truncated=truncated,
                )
                stats.episodes.append(summary)
                print(
                    f"episode={summary.episode:03d} seed={summary.seed} steps={summary.steps} "
                    f"reward={summary.total_reward:.6f} global={summary.global_reward:.6f} "
                    f"ppo={summary.ppo_local_reward:.6f} dqn={summary.dqn_local_reward:.6f} "
                    f"blocked={summary.blocked_steps} projected={summary.projected_steps} "
                    f"terminated={summary.terminated} truncated={summary.truncated}"
                )
            finally:
                env.close()
    finally:
        if log_handle is not None:
            log_handle.close()
    return stats


def print_report(stats: AutopsyStats, *, ppo_path: Path, dqn_path: Path, max_steps: int) -> None:
    episodes = stats.episodes
    total_steps = max(stats.total_steps, 1)
    print("\n# Aborted Joint-Training Autopsy Report")
    print()
    print(f"PPO checkpoint: `{ppo_path}`")
    print(f"DQN checkpoint: `{dqn_path}`")
    print(f"Episodes: {len(episodes)} | Max steps/episode: {max_steps} | Observed steps: {stats.total_steps}")
    print()
    print("| Metric | Mean | Std | Min | Max |")
    print("|---|---:|---:|---:|---:|")
    _metric_row("Total Reward / Episode", [item.total_reward for item in episodes])
    _metric_row("Global Reward / Episode", [item.global_reward for item in episodes])
    _metric_row("PPO Local Reward / Episode", [item.ppo_local_reward for item in episodes])
    _metric_row("DQN Local Reward / Episode", [item.dqn_local_reward for item in episodes])
    _metric_row("Service Level", [item.service_level for item in episodes])
    _metric_row("Safety Potential", [item.safety_potential for item in episodes])
    _metric_row("Transport Cost", [item.transport_cost for item in episodes])
    _metric_row("Episode Length", [float(item.steps) for item in episodes])

    print()
    print("| Operational Signal | Value |")
    print("|---|---:|")
    print(f"| Blocked Action Rate | {sum(item.blocked_steps for item in episodes) / total_steps:.6f} |")
    print(f"| Projected Action Rate | {sum(item.projected_steps for item in episodes) / total_steps:.6f} |")
    print(f"| Termination Rate | {sum(1 for item in episodes if item.terminated) / max(len(episodes), 1):.6f} |")
    print(f"| Truncation Rate | {sum(1 for item in episodes if item.truncated) / max(len(episodes), 1):.6f} |")
    print(f"| Rolling Demand Orders / Step | {stats.rolling_demand_orders / total_steps:.6f} |")

    print_distribution("Raw DQN Action", stats.raw_discrete_actions, limit=12)
    print_distribution("Dispatch", stats.dispatch)
    print_distribution("Route", stats.route)
    print_distribution("Mode", stats.mode)
    print_distribution("Reorder", stats.reorder)
    print_continuous_profile(stats)
    print_distribution("Blocked/Projection Reasons", stats.blocked_reasons + stats.projected_reasons, limit=12)
    print_behavioral_read(stats)


def print_continuous_profile(stats: AutopsyStats) -> None:
    labels = (
        "reorder_fraction_raw",
        "dispatch_intensity_raw",
        "speed_head_raw",
        "safety_stock_head_raw",
        "capacity_buffer_head_raw",
    )
    if not stats.continuous_raw:
        return
    matrix = np.vstack(stats.continuous_raw)
    print("\n## PPO Continuous Raw Action Profile")
    print("| Dimension | Mean | Std | Min | Max |")
    print("|---|---:|---:|---:|---:|")
    for index, label in enumerate(labels):
        values = matrix[:, index]
        print(f"| {label} | {np.mean(values):.6f} | {np.std(values):.6f} | {np.min(values):.6f} | {np.max(values):.6f} |")

    physical: defaultdict[str, list[float]] = defaultdict(list)
    for row in stats.continuous_physical:
        for key, value in row.items():
            physical[key].append(float(value))
    print("\n## PPO Projected Physical Action Profile")
    print("| Control | Mean | Std | Min | Max |")
    print("|---|---:|---:|---:|---:|")
    for key in sorted(physical):
        values = np.asarray(physical[key], dtype=np.float64)
        print(f"| {key} | {np.mean(values):.6f} | {np.std(values):.6f} | {np.min(values):.6f} | {np.max(values):.6f} |")


def print_behavioral_read(stats: AutopsyStats) -> None:
    total_steps = max(stats.total_steps, 1)
    dispatch_rate = stats.dispatch.get("dispatch", 0) / total_steps
    hold_rate = stats.dispatch.get("hold", 0) / total_steps
    emergency_rate = stats.reorder.get("emergency", 0) / total_steps
    secondary_rate = stats.mode.get("secondary_fleet", 0) / total_steps
    top_action, top_count = stats.raw_discrete_actions.most_common(1)[0] if stats.raw_discrete_actions else (-1, 0)
    raw_collapse_rate = top_count / total_steps
    physical = defaultdict(list)
    for row in stats.continuous_physical:
        for key, value in row.items():
            physical[key].append(float(value))
    mean_speed = _mean(physical.get("speed_multiplier", []))
    speed_std = _std(physical.get("speed_multiplier", []))

    print("\n## Engineering Behavioral Read")
    if raw_collapse_rate >= 0.80:
        print(f"- DQN shows strong action collapse: raw action `{top_action}` appears in {raw_collapse_rate:.1%} of steps.")
    else:
        print(f"- DQN uses a mixed tactical policy: top raw action `{top_action}` appears in {raw_collapse_rate:.1%} of steps.")

    if hold_rate > dispatch_rate:
        print(f"- Dispatch behavior is stall-biased: hold={hold_rate:.1%}, dispatch={dispatch_rate:.1%}.")
    else:
        print(f"- Dispatch behavior is movement-biased: dispatch={dispatch_rate:.1%}, hold={hold_rate:.1%}.")

    if emergency_rate >= 0.30:
        print(f"- Reorder behavior is emergency-heavy: emergency={emergency_rate:.1%}, suggesting tactical reward hacking or panic replenishment.")
    else:
        print(f"- Emergency reorder use is not dominant: emergency={emergency_rate:.1%}.")

    if secondary_rate >= 0.70:
        print(f"- Fleet sourcing is skewed toward secondary fleet: secondary={secondary_rate:.1%}.")
    else:
        print(f"- Fleet sourcing is not fully collapsed: secondary={secondary_rate:.1%}.")

    if mean_speed < 0.80:
        print(f"- PPO appears to slow the network materially: mean speed multiplier={mean_speed:.3f}, std={speed_std:.3f}.")
    elif mean_speed > 1.20:
        print(f"- PPO appears to run the network aggressively: mean speed multiplier={mean_speed:.3f}, std={speed_std:.3f}.")
    else:
        print(f"- PPO speed control stays near neutral: mean speed multiplier={mean_speed:.3f}, std={speed_std:.3f}.")


def print_distribution(title: str, counter: Counter[Any], *, limit: int | None = None) -> None:
    print(f"\n## {title} Distribution")
    print("| Value | Count | Share |")
    print("|---|---:|---:|")
    total = sum(counter.values())
    if total <= 0:
        print("| none | 0 | 0.000000 |")
        return
    rows = counter.most_common(limit)
    for value, count in rows:
        print(f"| {value} | {count} | {count / total:.6f} |")
    if limit is not None and len(counter) > limit:
        shown = sum(count for _, count in rows)
        print(f"| other | {total - shown} | {(total - shown) / total:.6f} |")


def _write_step_log(
    handle: Any,
    *,
    episode: int,
    seed: int,
    step: int,
    reward: float,
    reward_components: dict[str, float],
    ppo_raw: np.ndarray,
    ppo_physical: dict[str, float],
    dqn_action: int,
    dqn_decoded: dict[str, str],
    info: dict[str, Any],
    ppo_path: Path,
    dqn_path: Path,
) -> None:
    snapshot = _snapshot(info, {})
    payload = {
        "episode": episode,
        "seed": seed,
        "step": step,
        "reward": reward,
        "reward_components": reward_components,
        "ppo_raw": [float(value) for value in ppo_raw],
        "ppo_physical": ppo_physical,
        "dqn_action": dqn_action,
        "dqn_decoded": dqn_decoded,
        "blocked": bool(info.get("blocked", False)),
        "projected": bool(info.get("projected", False)),
        "projection_reasons": list(info.get("projection_reasons", ())),
        "rolling_demand_orders": int(info.get("rolling_demand_orders", 0)),
        "service_level": _float_snapshot(snapshot, "service_level"),
        "safety_potential": _float_snapshot(snapshot, "network_safety_potential"),
        "transport_cost": _float_snapshot(snapshot, "total_transport_cost"),
        "ppo_checkpoint": str(ppo_path),
        "dqn_checkpoint": str(dqn_path),
    }
    handle.write(json.dumps(payload, separators=(",", ":"), default=str) + "\n")


def _resolve_checkpoint(preferred: Path, pattern: str, label: str) -> Path:
    if preferred.exists():
        return preferred
    candidates = sorted(Path().glob(pattern), key=_checkpoint_sort_key)
    if not candidates:
        raise FileNotFoundError(f"No {label} checkpoint found. Tried {preferred} and {pattern}.")
    return candidates[-1]


def _checkpoint_sort_key(path: Path) -> tuple[int, float]:
    match = re.search(r"_(\d+)_steps\.zip$", path.name)
    step = int(match.group(1)) if match else -1
    return step, path.stat().st_mtime


def _config_max_steps() -> int:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return int(data["shared_global_parameters"]["max_steps"])


def _reward_components(info: Mapping[str, Any]) -> dict[str, float]:
    raw = info.get("reward_components", {})
    if not isinstance(raw, Mapping):
        return {}
    output: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            continue
        output[str(key)] = float(value)
    return output


def _snapshot(info: Mapping[str, Any], fallback: Mapping[str, Any]) -> dict[str, Any]:
    raw = info.get("snapshot", fallback)
    return dict(raw) if isinstance(raw, Mapping) else dict(fallback)


def _float_snapshot(snapshot: Mapping[str, Any], key: str) -> float:
    value = snapshot.get(key, 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return 0.0
    return float(value)


def _count_reasons(info: Mapping[str, Any], blocked: Counter[str], projected: Counter[str]) -> None:
    reasons = info.get("projection_reasons", ())
    if not isinstance(reasons, Iterable) or isinstance(reasons, (str, bytes)):
        return
    target = blocked if bool(info.get("blocked", False)) else projected
    for reason in reasons:
        target[str(reason)] += 1


def _metric_row(name: str, values: list[float]) -> None:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        print(f"| {name} | n/a | n/a | n/a | n/a |")
        return
    print(
        f"| {name} | {np.mean(array):.6f} | {np.std(array):.6f} | "
        f"{np.min(array):.6f} | {np.max(array):.6f} |"
    )


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.mean(array)) if array.size else 0.0


def _std(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.std(array)) if array.size else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autopsy evaluation for aborted staggered joint PPO+DQN models.")
    parser.add_argument("--ppo-path", type=Path)
    parser.add_argument("--dqn-path", type=Path)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--step-log-path", type=Path, default=DEFAULT_STEP_LOG_PATH)
    parser.add_argument("--no-step-log", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
