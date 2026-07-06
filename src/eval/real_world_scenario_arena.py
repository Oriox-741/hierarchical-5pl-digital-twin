"""Evaluation-only real-world scenario arena for trained joint policies."""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.act.env_5pl import EnvironmentConfig, FivePLDigitalTwinEnv
from src.act.observation_builder import OBSERVATION_DIM
from src.learn.train_joint_torch import MDP_CONTRACT_VERSION, build_environment_config
from src.think.joint_policies import FLAT_DQN_ARCHITECTURE, JointPolicyBundle
from src.eval.scenario_metrics import (
    EpisodeMetricAccumulator,
    EpisodeMetrics,
    GLOBAL_FATAL_HARD_BLOCKER_FIELDS,
    evaluate_global_hard_blockers,
    evaluate_thresholds,
    evaluate_warnings,
    summarize_episodes,
)
from src.eval.scenario_overrides import (
    OverrideCapability,
    apply_config_overrides,
    apply_environment_overrides,
    capability_matrix_as_dict,
)


EXPECTED_OBSERVATION_DIM = OBSERVATION_DIM
EXPECTED_MDP_CONTRACT = MDP_CONTRACT_VERSION


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    scenario_id: str
    name: str
    description: str
    seeds: tuple[int, ...]
    episode_count: int
    environment_overrides: dict[str, Any] = field(default_factory=dict)
    unsupported_environment_overrides: dict[str, Any] = field(default_factory=dict)
    expected_behavior: str = ""
    expected_override_effects: dict[str, str] = field(default_factory=dict)
    pass_fail_thresholds: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ScenarioConfig":
        required = ("scenario_id", "name", "description", "seeds", "episode_count")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"scenario config missing required keys: {', '.join(missing)}")
        seeds = tuple(int(seed) for seed in payload["seeds"])
        if not seeds:
            raise ValueError("scenario config seeds must not be empty.")
        episode_count = int(payload["episode_count"])
        if episode_count <= 0:
            raise ValueError("scenario config episode_count must be positive.")
        overrides = payload.get("environment_overrides", {})
        unsupported = payload.get("unsupported_environment_overrides", {})
        expected_effects = payload.get("expected_override_effects", {})
        thresholds = payload.get("pass_fail_thresholds", {})
        if not isinstance(overrides, dict):
            raise ValueError("environment_overrides must be an object.")
        if not isinstance(unsupported, dict):
            raise ValueError("unsupported_environment_overrides must be an object.")
        if not isinstance(thresholds, dict):
            raise ValueError("pass_fail_thresholds must be an object.")
        if not isinstance(expected_effects, dict):
            raise ValueError("expected_override_effects must be an object.")
        return cls(
            scenario_id=str(payload["scenario_id"]),
            name=str(payload["name"]),
            description=str(payload["description"]),
            seeds=seeds,
            episode_count=episode_count,
            environment_overrides=dict(overrides),
            unsupported_environment_overrides=dict(unsupported),
            expected_behavior=str(payload.get("expected_behavior", "")),
            expected_override_effects=dict(expected_effects),
            pass_fail_thresholds=dict(thresholds),
        )


@dataclass(frozen=True, slots=True)
class CheckpointPolicy:
    checkpoint_path: Path
    checkpoint: Mapping[str, Any]
    policies: JointPolicyBundle


def load_scenario_configs(scenario_dir: Path) -> list[ScenarioConfig]:
    if not scenario_dir.exists():
        raise FileNotFoundError(f"scenario directory not found: {scenario_dir}")
    scenarios: list[ScenarioConfig] = []
    for path in sorted(scenario_dir.glob("*.json")):
        scenarios.append(ScenarioConfig.from_mapping(json.loads(path.read_text(encoding="utf-8"))))
    if not scenarios:
        raise ValueError(f"no scenario JSON files found under {scenario_dir}")
    ids = [scenario.scenario_id for scenario in scenarios]
    duplicates = sorted({scenario_id for scenario_id in ids if ids.count(scenario_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate scenario_id values: {', '.join(duplicates)}")
    return scenarios


def load_policy_checkpoint(checkpoint_path: Path, *, device: torch.device) -> CheckpointPolicy:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping.")
    validate_checkpoint_metadata(checkpoint)

    policies = JointPolicyBundle(
        observation_dim=int(checkpoint.get("observation_dim", OBSERVATION_DIM)),
        continuous_action_dim=int(checkpoint.get("continuous_action_dim", 5)),
        discrete_action_count=int(checkpoint.get("discrete_action_count", 48)),
        dqn_architecture=str(checkpoint.get("dqn_architecture", FLAT_DQN_ARCHITECTURE)),
    ).to(device)
    policies.load_checkpoint_state(dict(checkpoint))
    policies.eval()
    return CheckpointPolicy(checkpoint_path=checkpoint_path, checkpoint=checkpoint, policies=policies)


def validate_checkpoint_metadata(checkpoint: Mapping[str, Any]) -> None:
    config = checkpoint.get("config")
    version = config.get("mdp_contract_version") if isinstance(config, Mapping) else None
    if version != EXPECTED_MDP_CONTRACT or version != MDP_CONTRACT_VERSION:
        raise ValueError(
            f"checkpoint MDP contract mismatch: expected {EXPECTED_MDP_CONTRACT!r}, got {version!r}"
        )
    observation_dim = checkpoint.get("observation_dim")
    if observation_dim is not None and int(observation_dim) != EXPECTED_OBSERVATION_DIM:
        raise ValueError(
            f"checkpoint observation_dim mismatch: expected {EXPECTED_OBSERVATION_DIM}, got {observation_dim}"
        )
    shared = config.get("shared_global_parameters", {}) if isinstance(config, Mapping) else {}
    configured_dim = shared.get("observation_dim") if isinstance(shared, Mapping) else None
    if configured_dim is not None and int(configured_dim) != EXPECTED_OBSERVATION_DIM:
        raise ValueError(
            f"config observation_dim mismatch: expected {EXPECTED_OBSERVATION_DIM}, got {configured_dim}"
        )


def build_scenario_environment_config(
    base_training_config: Mapping[str, Any],
    scenario: ScenarioConfig,
    *,
    seed: int,
    max_steps: int | None = None,
) -> tuple[EnvironmentConfig, list[str]]:
    shared = base_training_config.get("shared_global_parameters", {})
    default_steps = int(shared.get("max_steps", 288)) if isinstance(shared, Mapping) else 288
    env_config = build_environment_config(base_training_config, max_steps=max_steps or default_steps, seed=seed)
    matrix = apply_config_overrides(env_config, scenario.environment_overrides)
    unsupported = [
        item["key"]
        for item in capability_matrix_as_dict(matrix)
        if item["status"] in {"UNSUPPORTED", "UNKNOWN"}
    ]
    unsupported.extend(sorted(scenario.unsupported_environment_overrides))
    return env_config, unsupported


def build_scenario_environment(
    base_training_config: Mapping[str, Any],
    scenario: ScenarioConfig,
    *,
    seed: int,
    max_steps: int | None = None,
) -> tuple[FivePLDigitalTwinEnv, list[dict[str, str]]]:
    shared = base_training_config.get("shared_global_parameters", {})
    default_steps = int(shared.get("max_steps", 288)) if isinstance(shared, Mapping) else 288
    env_config = build_environment_config(base_training_config, max_steps=max_steps or default_steps, seed=seed)
    matrix = apply_config_overrides(env_config, scenario.environment_overrides)
    env = FivePLDigitalTwinEnv(config=env_config)
    matrix = apply_environment_overrides(env, scenario.environment_overrides, matrix)
    for key, explanation in scenario.unsupported_environment_overrides.items():
        matrix[key] = OverrideCapability(key, "UNSUPPORTED", str(explanation))
    return env, capability_matrix_as_dict(matrix)


def run_scenario_episode(
    *,
    scenario: ScenarioConfig,
    policy: CheckpointPolicy,
    seed: int,
    episode_index: int,
    deterministic: bool,
    max_steps: int | None = None,
) -> EpisodeMetrics:
    config = policy.checkpoint.get("config", {})
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint config is missing or invalid.")
    env, _capability_matrix = build_scenario_environment(config, scenario, seed=seed, max_steps=max_steps)
    try:
        observation, _info = env.reset(seed=seed)
        accumulator = EpisodeMetricAccumulator(
            scenario_id=scenario.scenario_id,
            seed=seed,
            episode_index=episode_index,
        )
        device = next(policy.policies.parameters()).device
        terminated = False
        truncated = False
        with torch.no_grad():
            while not (terminated or truncated):
                observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                if observation_tensor.shape != (1, EXPECTED_OBSERVATION_DIM):
                    raise ValueError(
                        f"expected observation shape (1, {EXPECTED_OBSERVATION_DIM}), got {tuple(observation_tensor.shape)}"
                    )
                ppo = policy.policies.select_ppo_action(observation_tensor, deterministic=deterministic)
                dqn = policy.policies.select_dqn_action(observation_tensor, epsilon=0.0)
                observation, reward, terminated, truncated, info = env.step(
                    {
                        "continuous": ppo.action.squeeze(0).detach().cpu().numpy().astype(np.float32),
                        "discrete": int(dqn.action.squeeze(0).detach().cpu().item()),
                    }
                )
                accumulator.observe(reward=float(reward), info=info)
        return accumulator.finish()
    finally:
        env.close()


def build_dry_run_episode(scenario: ScenarioConfig, *, seed: int, episode_index: int) -> EpisodeMetrics:
    return EpisodeMetrics(
        scenario_id=scenario.scenario_id,
        seed=seed,
        episode_index=episode_index,
        steps=1,
        total_reward=0.0,
        total_delivered=0,
        delivered_delta=None,
        service_level=None,
        dispatch_attempts=0,
        dispatch_successes=0,
        dispatch_success_per_attempt=None,
        dispatch_rate=0.0,
        hold_rate=1.0,
    )


def write_evaluation_outputs(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    scenarios: list[ScenarioConfig],
    episode_rows: list[EpisodeMetrics],
    override_capabilities: Mapping[str, list[dict[str, str]]],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_scenario: dict[str, list[EpisodeMetrics]] = {}
    for row in episode_rows:
        by_scenario.setdefault(row.scenario_id, []).append(row)

    scenario_summaries: list[dict[str, Any]] = []
    for scenario in scenarios:
        summary = summarize_episodes(scenario.scenario_id, by_scenario.get(scenario.scenario_id, []))
        summary["name"] = scenario.name
        capabilities = override_capabilities.get(scenario.scenario_id, [])
        summary["override_capabilities"] = capabilities
        summary["unsupported_overrides"] = [
            item["key"] for item in capabilities if item.get("status") == "UNSUPPORTED"
        ]
        summary["partial_overrides"] = [
            item["key"] for item in capabilities if item.get("status") == "PARTIAL"
        ]
        failures = evaluate_thresholds(summary, scenario.pass_fail_thresholds)
        hard_blocker_failures = evaluate_global_hard_blockers(summary)
        scenario_threshold_failures = [
            failure for failure in failures if failure not in hard_blocker_failures
        ]
        warnings = evaluate_warnings(summary, scenario.pass_fail_thresholds)
        summary["scenario_threshold_failures"] = scenario_threshold_failures
        summary["hard_blocker_failures"] = hard_blocker_failures
        summary["scenario_threshold_verdict"] = "PASS" if not scenario_threshold_failures else "FAIL"
        summary["hard_blocker_verdict"] = "PASS" if not hard_blocker_failures else "FAIL"
        summary["threshold_failures"] = failures
        summary["threshold_warnings"] = warnings
        summary["verdict"] = "PASS" if not failures else "FAIL"
        scenario_summaries.append(summary)

    episode_path = output_dir / "episode_metrics.jsonl"
    with episode_path.open("w", encoding="utf-8") as handle:
        for row in episode_rows:
            handle.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")

    summary_json = {
        "checkpoint": str(checkpoint_path),
        "contract": EXPECTED_MDP_CONTRACT,
        "observation_dim": EXPECTED_OBSERVATION_DIM,
        "scenarios": scenario_summaries,
    }
    (output_dir / "scenario_summary.json").write_text(
        json.dumps(summary_json, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary_columns = sorted({key for row in scenario_summaries for key in row})
    with (output_dir / "scenario_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns)
        writer.writeheader()
        for row in scenario_summaries:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})

    report_lines = [
        "# Real-World Scenario Evaluation Report",
        "",
        f"Checkpoint: `{checkpoint_path}`",
        f"Contract: `{EXPECTED_MDP_CONTRACT}`",
        f"Observation dim: `{EXPECTED_OBSERVATION_DIM}`",
        "",
        "delivered_delta is a backward-compatible alias for mean_step_delivered_delta; final_step_delivered_delta is reported separately.",
        "Global hard blockers override scenario threshold pass.",
        "",
        "| Scenario | Verdict | Scenario Threshold Verdict | Hard-Blocker Verdict | Threshold Failures | Hard-Blocker Failures | Threshold Warnings | Unsupported Overrides | Partial Overrides |",
        "|---|---:|---:|---:|---|---|---|---|---|",
    ]
    for row in scenario_summaries:
        report_lines.append(
            f"| {row['scenario_id']} | {row['verdict']} | "
            f"{row['scenario_threshold_verdict']} | "
            f"{row['hard_blocker_verdict']} | "
            f"{'; '.join(row['threshold_failures']) or 'none'} | "
            f"{'; '.join(row['hard_blocker_failures']) or 'none'} | "
            f"{'; '.join(row['threshold_warnings']) or 'none'} | "
            f"{', '.join(row['unsupported_overrides']) or 'none'} | "
            f"{', '.join(row['partial_overrides']) or 'none'} |"
        )
    report_lines.extend(
        [
            "",
            "## Hard-Blocker Telemetry",
            "",
            "| Scenario | "
            + " | ".join(key for key, _label in GLOBAL_FATAL_HARD_BLOCKER_FIELDS)
            + " |",
            "|---|" + "|".join("---:" for _key, _label in GLOBAL_FATAL_HARD_BLOCKER_FIELDS) + "|",
        ]
    )
    for row in scenario_summaries:
        report_lines.append(
            f"| {row['scenario_id']} | "
            + " | ".join(str(row.get(key, 0)) for key, _label in GLOBAL_FATAL_HARD_BLOCKER_FIELDS)
            + " |"
        )
    premium_guard_fields = (
        "premium_sla_fleet_credit_blocked_no_current_work",
        "premium_primary_adaptation_credit_blocked_no_current_work",
        "premium_fleet_credit_allowed",
    )
    report_lines.extend(
        [
            "",
            "## Premium Reward Guard Telemetry",
            "",
            "| Scenario | "
            + " | ".join(premium_guard_fields)
            + " |",
            "|---|" + "|".join("---:" for _field in premium_guard_fields) + "|",
        ]
    )
    for row in scenario_summaries:
        report_lines.append(
            f"| {row['scenario_id']} | "
            + " | ".join(str(row.get(field, 0)) for field in premium_guard_fields)
            + " |"
        )
    report_lines.extend(
        [
            "",
            "## Route/Premium/Mixed Diagnostic Telemetry",
            "",
            "mixed_stress remains a watch unless candidate-quality telemetry proves avoidable overconservatism.",
            "",
            "| Scenario | action31 attempts | action31 successes | action31 failed/no-op | action31 success ratio | Top failed/no-op actions | Successful dispatch routes | Failed dispatch routes | Hold routes | Premium hold under useful dispatch | Premium missed reorder opportunity | Shortest near-best not selected | High-resilience selected when shortest near-best |",
            "|---|---:|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in scenario_summaries:
        report_lines.append(
            f"| {row['scenario_id']} | "
            f"{row.get('action31_attempts', 0)} | "
            f"{row.get('action31_successes', 0)} | "
            f"{row.get('action31_failed_noop', 0)} | "
            f"{row.get('action31_dispatch_success_ratio')} | "
            f"{json.dumps(row.get('top_failed_noop_actions', []), sort_keys=True)} | "
            f"{json.dumps(row.get('scenario_route_distribution_successful_dispatch', {}), sort_keys=True)} | "
            f"{json.dumps(row.get('scenario_route_distribution_failed_dispatch', {}), sort_keys=True)} | "
            f"{json.dumps(row.get('scenario_route_distribution_hold', {}), sort_keys=True)} | "
            f"{row.get('premium_hold_under_useful_dispatch_opportunity', 0)} | "
            f"{row.get('premium_reorder_missed_opportunity_steps', 0)} | "
            f"{row.get('shortest_near_best_but_not_selected_count', 0)} | "
            f"{row.get('high_resilience_selected_when_shortest_near_best_count', 0)} |"
        )
    report_lines.extend(
        [
            "",
            "### Route Candidate Quality",
            "",
            "| Scenario | route_candidate_score_mean_by_route | selected_route_candidate_score_mean_by_route | selected_route_rank_counts | selected_route_best_count | selected_route_near_best_count | selected_route_margin_to_best_mean | shortest_near_best_but_not_selected_count | high_resilience_selected_when_shortest_near_best_count | high_resilience_selected_when_not_best_count | mixed_route_overconservative_candidate_count |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in scenario_summaries:
        report_lines.append(
            f"| {row['scenario_id']} | "
            f"{json.dumps(row.get('route_candidate_score_mean_by_route', {}), sort_keys=True)} | "
            f"{json.dumps(row.get('selected_route_candidate_score_mean_by_route', {}), sort_keys=True)} | "
            f"{json.dumps(row.get('selected_route_rank_counts', {}), sort_keys=True)} | "
            f"{row.get('selected_route_best_count', 0)} | "
            f"{row.get('selected_route_near_best_count', 0)} | "
            f"{row.get('selected_route_margin_to_best_mean')} | "
            f"{row.get('shortest_near_best_but_not_selected_count', 0)} | "
            f"{row.get('high_resilience_selected_when_shortest_near_best_count', 0)} | "
            f"{row.get('high_resilience_selected_when_not_best_count', 0)} | "
            f"{row.get('mixed_route_overconservative_candidate_count', 0)} |"
        )
    report_lines.extend(
        [
            "",
            "### Premium SLA Responsiveness",
            "",
            "| Scenario | premium_pressure_steps | premium_hold_rate | premium_dispatch_rate | premium_missed_useful_dispatch_rate | premium_no_vehicle_steps | premium_already_assigned_steps | premium_no_unassigned_steps | premium_reorder_none_steps | premium_reorder_conservative_steps | premium_reorder_aggressive_steps | premium_reorder_emergency_steps | premium_missed_useful_reorder_rate | premium_service_pressure_steps | premium_late_or_at_risk_backlog_steps | hold_primary_under_premium_pressure_rate | dispatch_primary_under_premium_pressure |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in scenario_summaries:
        report_lines.append(
            f"| {row['scenario_id']} | "
            f"{row.get('premium_pressure_steps', 0)} | "
            f"{row.get('premium_hold_rate')} | "
            f"{row.get('premium_dispatch_rate')} | "
            f"{row.get('premium_missed_useful_dispatch_rate')} | "
            f"{row.get('premium_no_vehicle_steps', 0)} | "
            f"{row.get('premium_already_assigned_steps', 0)} | "
            f"{row.get('premium_no_unassigned_steps', 0)} | "
            f"{row.get('premium_reorder_none_steps', 0)} | "
            f"{row.get('premium_reorder_conservative_steps', 0)} | "
            f"{row.get('premium_reorder_aggressive_steps', 0)} | "
            f"{row.get('premium_reorder_emergency_steps', 0)} | "
            f"{row.get('premium_missed_useful_reorder_rate')} | "
            f"{row.get('premium_service_pressure_steps', 0)} | "
            f"{row.get('premium_late_or_at_risk_backlog_steps', 0)} | "
            f"{row.get('hold_primary_under_premium_pressure_rate')} | "
            f"{row.get('dispatch_primary_under_premium_pressure', 0)} |"
        )
    (output_dir / "real_world_evaluation_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return summary_json
