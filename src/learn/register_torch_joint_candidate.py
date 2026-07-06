"""Safely promote paired Torch joint candidate registry records to active."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

import torch

from src.act.discrete_action_mapper import DISCRETE_ACTION_COUNT
from src.act.observation_builder import OBSERVATION_DIM
from src.eval.scenario_metrics import GLOBAL_FATAL_HARD_BLOCKER_FIELDS
from src.learn.model_registry import Algorithm, ModelRecord, ModelRegistry
from src.orchestration.policy_service import DQN_AGENT_ROLE, PPO_AGENT_ROLE
from src.orchestration.torch_joint_runtime import EXPECTED_MDP_CONTRACT
from src.think.joint_policies import CHECKPOINT_VERSION, HIERARCHICAL_DQN_ARCHITECTURE


EXPECTED_RUNTIME_LOADER = "joint_torch"
EXPECTED_FRAMEWORK = "torch_joint"
PPO_ARTIFACT_KIND = "ppo_final"
DQN_ARTIFACT_KIND = "dqn_final"
JOINT_FINAL_ARTIFACT_KIND = "joint_final"
DEFAULT_SCENARIO_COUNT = 8


@dataclass(frozen=True, slots=True)
class CandidateRegistrationPlan:
    ppo_path: Path
    dqn_path: Path
    ppo_metadata: dict[str, Any]
    dqn_metadata: dict[str, Any]
    metrics: dict[str, float]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.dry_run and args.activate:
        raise ValueError("--dry-run and --activate are mutually exclusive.")
    if args.activate and args.register_candidate:
        raise ValueError("--activate and --register-candidate are mutually exclusive.")
    if args.replace_active and args.register_candidate:
        raise ValueError("--replace-active applies only to candidate activation, not candidate registration.")

    registry_dir = Path(args.registry_dir)
    if not registry_dir.exists():
        raise FileNotFoundError(f"registry directory does not exist: {registry_dir}")
    registry = ModelRegistry(registry_dir)

    if _uses_artifact_candidate_mode(args):
        if args.replace_active:
            raise ValueError("--replace-active applies only to candidate activation mode.")
        plan = _build_candidate_registration_plan(args)
        if args.activate:
            raise ValueError("--activate requires --ppo-candidate-id and --dqn-candidate-id, not artifact paths.")
        if args.register_candidate:
            _register_candidate_rows(registry, plan)
            print("CANDIDATE_REGISTRATION_COMPLETE")
        else:
            _print_candidate_dry_run(plan, registry)
            print("DRY_RUN_OK")
        return 0

    if not args.ppo_candidate_id or not args.dqn_candidate_id:
        raise ValueError(
            "--ppo-candidate-id and --dqn-candidate-id are required unless --joint-checkpoint/--ppo-path/--dqn-path "
            "candidate artifact mode is used."
        )

    ppo_record = _find_record_by_id(registry, args.ppo_candidate_id, algorithm="ppo")
    dqn_record = _find_record_by_id(registry, args.dqn_candidate_id, algorithm="dqn")
    pair = _validate_candidate_pair(
        ppo_record,
        dqn_record,
        logical_model_id=args.logical_model_id,
        registry=registry,
        replace_active=args.replace_active,
    )

    if args.activate:
        _activate_pair(registry, pair)
        print("ACTIVATION_COMPLETE")
    else:
        _print_dry_run(pair, registry)
        print("DRY_RUN_OK")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a reviewed Torch joint PPO/DQN candidate pair to active.")
    parser.add_argument("--logical-model-id", required=True)
    parser.add_argument("--ppo-candidate-id")
    parser.add_argument("--dqn-candidate-id")
    parser.add_argument("--registry-dir", default="models/registry")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print intended writes without mutation.")
    parser.add_argument("--activate", action="store_true", help="Append active records and update active_models.json.")
    parser.add_argument("--register-candidate", action="store_true", help="Append candidate rows without activation.")
    parser.add_argument(
        "--replace-active",
        action="store_true",
        help="Allow an explicit active registry handoff when PPO/DQN active mappings already exist.",
    )
    parser.add_argument("--joint-checkpoint", help="Joint checkpoint path for candidate-row registration mode.")
    parser.add_argument("--ppo-path", help="Final PPO artifact path for candidate-row registration mode.")
    parser.add_argument("--dqn-path", help="Final DQN artifact path for candidate-row registration mode.")
    parser.add_argument("--eval-output-dir", help="Offline scenario eval directory for candidate-row registration mode.")
    parser.add_argument("--gate-result", help="Long-run gate JSON path for candidate-row registration mode.")
    parser.add_argument("--dqn-architecture", help="Expected DQN architecture recorded in the joint checkpoint.")
    parser.add_argument("--hierarchical-init-method", help="Expected hierarchical initialization method, when applicable.")
    return parser.parse_args(argv)


def _uses_artifact_candidate_mode(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, field) is not None
        for field in (
            "joint_checkpoint",
            "ppo_path",
            "dqn_path",
            "eval_output_dir",
            "gate_result",
            "dqn_architecture",
            "hierarchical_init_method",
        )
    )


def _build_candidate_registration_plan(args: argparse.Namespace) -> CandidateRegistrationPlan:
    missing = [
        option
        for option, value in (
            ("--joint-checkpoint", args.joint_checkpoint),
            ("--ppo-path", args.ppo_path),
            ("--dqn-path", args.dqn_path),
            ("--eval-output-dir", args.eval_output_dir),
            ("--gate-result", args.gate_result),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"candidate artifact mode requires: {', '.join(missing)}.")

    joint_checkpoint = Path(str(args.joint_checkpoint))
    ppo_path = Path(str(args.ppo_path))
    dqn_path = Path(str(args.dqn_path))
    eval_output_dir = Path(str(args.eval_output_dir))
    gate_result = Path(str(args.gate_result))

    _require_existing_path(joint_checkpoint, "joint_checkpoint")
    _require_existing_path(ppo_path, "PPO artifact")
    _require_existing_path(dqn_path, "DQN artifact")
    _require_existing_path(eval_output_dir, "eval_output_dir")
    _require_existing_path(gate_result, "gate_result")

    checkpoint_metadata = _validate_joint_checkpoint_for_candidate(
        joint_checkpoint,
        logical_model_id=str(args.logical_model_id),
        expected_dqn_architecture=args.dqn_architecture,
        expected_hierarchical_init_method=args.hierarchical_init_method,
    )
    eval_metadata = _validate_eval_output_dir(eval_output_dir)
    gate_metadata = _validate_gate_result(gate_result)
    training_step = int(checkpoint_metadata["training_step"])
    exact_resume_capable = bool(checkpoint_metadata["exact_resume_capable"])

    common_metadata: dict[str, Any] = {
        "logical_model_id": str(args.logical_model_id),
        "framework": EXPECTED_FRAMEWORK,
        "runtime_loader": EXPECTED_RUNTIME_LOADER,
        "dqn_architecture": checkpoint_metadata["dqn_architecture"],
        "joint_checkpoint": str(joint_checkpoint),
        "contract": checkpoint_metadata["contract"],
        "obs_dim": int(checkpoint_metadata["obs_dim"]),
        "action_count": int(checkpoint_metadata["action_count"]),
        "external_discrete_action_count": int(checkpoint_metadata["external_discrete_action_count"]),
        "training_step": training_step,
        "exact_resume_capable": exact_resume_capable,
        "eval_output_dir": str(eval_output_dir),
        "eval_verdict": "PASS",
        "hard_blocker_status": "zero",
        "long_run_gate_verdict": gate_metadata["long_run_gate_verdict"],
        "registry_scope": "candidate_only",
        "production_ready": False,
        "baseline_update": False,
    }
    if checkpoint_metadata.get("hierarchical_init_method"):
        common_metadata["hierarchical_init_method"] = checkpoint_metadata["hierarchical_init_method"]

    return CandidateRegistrationPlan(
        ppo_path=ppo_path,
        dqn_path=dqn_path,
        ppo_metadata={**common_metadata, "artifact_kind": PPO_ARTIFACT_KIND},
        dqn_metadata={**common_metadata, "artifact_kind": DQN_ARTIFACT_KIND},
        metrics={"scenario_pass_rate": 1.0, "scenario_count": float(eval_metadata["scenario_count"])},
    )


def _find_record_by_id(registry: ModelRegistry, model_id: str, *, algorithm: Algorithm) -> ModelRecord:
    matches = [record for record in registry.records() if record.model_id == model_id]
    if not matches:
        raise ValueError(f"{algorithm} candidate id not found: {model_id}")
    if len(matches) != 1:
        raise ValueError(f"{algorithm} candidate id is not unique: {model_id}")
    [record] = matches
    if record.algorithm != algorithm:
        raise ValueError(f"{algorithm} candidate id {model_id} points to algorithm {record.algorithm!r}.")
    return record


def _validate_candidate_pair(
    ppo_record: ModelRecord,
    dqn_record: ModelRecord,
    *,
    logical_model_id: str,
    registry: ModelRegistry,
    replace_active: bool = False,
) -> tuple[ModelRecord, ModelRecord]:
    _validate_record(
        ppo_record,
        algorithm="ppo",
        agent_role=PPO_AGENT_ROLE,
        artifact_kind=PPO_ARTIFACT_KIND,
        logical_model_id=logical_model_id,
    )
    _validate_record(
        dqn_record,
        algorithm="dqn",
        agent_role=DQN_AGENT_ROLE,
        artifact_kind=DQN_ARTIFACT_KIND,
        logical_model_id=logical_model_id,
    )

    if _metadata_text(ppo_record, "logical_model_id") != _metadata_text(dqn_record, "logical_model_id"):
        raise ValueError("PPO/DQN candidate logical_model_id mismatch.")
    if _metadata_text(ppo_record, "joint_checkpoint") != _metadata_text(dqn_record, "joint_checkpoint"):
        raise ValueError("PPO/DQN candidate joint_checkpoint mismatch.")

    joint_checkpoint = Path(_metadata_text(ppo_record, "joint_checkpoint"))
    if not joint_checkpoint.exists():
        raise FileNotFoundError(f"joint_checkpoint artifact path does not exist: {joint_checkpoint}")
    _require_existing_path(Path(ppo_record.path), "PPO artifact")
    _require_existing_path(Path(dqn_record.path), "DQN artifact")

    eval_output_dir = ppo_record.metadata.get("eval_output_dir")
    if eval_output_dir is not None and str(eval_output_dir) != "":
        _require_existing_path(Path(str(eval_output_dir)), "eval_output_dir")
    dqn_eval_output_dir = dqn_record.metadata.get("eval_output_dir")
    if dqn_eval_output_dir is not None and str(dqn_eval_output_dir) != "":
        _require_existing_path(Path(str(dqn_eval_output_dir)), "eval_output_dir")
    if str(eval_output_dir or "") != str(dqn_eval_output_dir or ""):
        raise ValueError("PPO/DQN candidate eval_output_dir mismatch.")

    active_records = [
        record
        for record in registry.records(status="active")
        if record.metadata.get("logical_model_id") == logical_model_id
    ]
    if active_records:
        raise ValueError(f"duplicate active records already exist for logical_model_id={logical_model_id!r}.")

    active = _read_active_json(registry.active_path)
    occupied = [
        key
        for key in (
            "ppo",
            "dqn",
            _active_key("ppo", PPO_AGENT_ROLE),
            _active_key("dqn", DQN_AGENT_ROLE),
        )
        if key in active
    ]
    if occupied and not replace_active:
        raise ValueError(
            f"active registry mapping already active for {', '.join(occupied)}; "
            "pass --replace-active after dry-run review to plan an explicit handoff."
        )

    return ppo_record, dqn_record


def _validate_record(
    record: ModelRecord,
    *,
    algorithm: Algorithm,
    agent_role: str,
    artifact_kind: str,
    logical_model_id: str,
) -> None:
    if record.status != "candidate":
        raise ValueError(f"{algorithm} candidate status must be 'candidate', got {record.status!r}.")
    if record.algorithm != algorithm:
        raise ValueError(f"{algorithm} candidate algorithm mismatch: got {record.algorithm!r}.")
    _require_metadata_value(record, "agent_role", agent_role, algorithm=algorithm)
    _require_metadata_value(record, "logical_model_id", logical_model_id, algorithm=algorithm)
    _require_metadata_value(record, "framework", EXPECTED_FRAMEWORK, algorithm=algorithm)
    _require_metadata_value(record, "runtime_loader", EXPECTED_RUNTIME_LOADER, algorithm=algorithm)
    _require_metadata_value(record, "artifact_kind", artifact_kind, algorithm=algorithm)
    _require_metadata_value(record, "contract", EXPECTED_MDP_CONTRACT, algorithm=algorithm)
    _require_metadata_int(record, "obs_dim", OBSERVATION_DIM, algorithm=algorithm)
    _require_metadata_int(record, "action_count", DISCRETE_ACTION_COUNT, algorithm=algorithm)
    _require_positive_metadata_int(record, "training_step", algorithm=algorithm)
    _require_metadata_value(record, "eval_verdict", "PASS", algorithm=algorithm)
    _require_metadata_value(record, "hard_blocker_status", "zero", algorithm=algorithm)
    dqn_architecture = str(record.metadata.get("dqn_architecture", "")).lower()
    if dqn_architecture == HIERARCHICAL_DQN_ARCHITECTURE:
        _require_metadata_value(
            record,
            "hierarchical_init_method",
            "flat_teacher_distillation_v1",
            algorithm=algorithm,
        )
        _require_metadata_int(record, "external_discrete_action_count", DISCRETE_ACTION_COUNT, algorithm=algorithm)
        _require_metadata_value(record, "exact_resume_capable", "True", algorithm=algorithm)
        _require_metadata_value(record, "long_run_gate_verdict", "PASS", algorithm=algorithm)
        _require_metadata_value(record, "registry_scope", "candidate_only", algorithm=algorithm)
    _require_false_or_absent(record, "production_ready", algorithm=algorithm)
    _require_false_or_absent(record, "baseline_update", algorithm=algorithm)


def _activate_pair(registry: ModelRegistry, pair: tuple[ModelRecord, ModelRecord]) -> None:
    ppo_record, dqn_record = pair
    active_records = [
        _promoted_record(ppo_record),
        _promoted_record(dqn_record),
    ]
    active = _read_active_json(registry.active_path)
    active[_active_key("ppo", PPO_AGENT_ROLE)] = ppo_record.path
    active[_active_key("dqn", DQN_AGENT_ROLE)] = dqn_record.path

    with registry.index_path.open("a", encoding="utf-8") as handle:
        for record in active_records:
            handle.write(json.dumps(asdict(record), separators=(",", ":")) + "\n")

    try:
        _write_active_json_atomic(registry.active_path, active)
    except OSError as exc:
        raise RuntimeError(
            "models.jsonl active rows were appended but active_models.json write failed; manual registry audit required."
        ) from exc


def _promoted_record(candidate: ModelRecord) -> ModelRecord:
    metadata = dict(candidate.metadata)
    metadata.update(
        {
            "promoted_from_candidate_id": candidate.model_id,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "activation_scope": "registry_active_only",
            "production_promotion": False,
            "baseline_update": False,
        }
    )
    return ModelRecord(
        model_id=str(uuid4()),
        algorithm=candidate.algorithm,
        path=candidate.path,
        status="active",
        created_at=datetime.now(timezone.utc).isoformat(),
        metrics=dict(candidate.metrics),
        metadata=metadata,
    )


def _print_dry_run(pair: tuple[ModelRecord, ModelRecord], registry: ModelRegistry) -> None:
    ppo_record, dqn_record = pair
    print("Dry-run: would activate Torch joint candidate pair.")
    print(f"logical_model_id: {_metadata_text(ppo_record, 'logical_model_id')}")
    print(f"ppo:{PPO_AGENT_ROLE} -> {ppo_record.path}")
    print(f"dqn:{DQN_AGENT_ROLE} -> {dqn_record.path}")
    print(f"joint_checkpoint: {_metadata_text(ppo_record, 'joint_checkpoint')}")
    active = _read_active_json(registry.active_path)
    occupied = {
        key: active[key]
        for key in (
            "ppo",
            "dqn",
            _active_key("ppo", PPO_AGENT_ROLE),
            _active_key("dqn", DQN_AGENT_ROLE),
        )
        if key in active
    }
    if occupied:
        print("would replace active mappings:")
        for key, value in occupied.items():
            print(f"  {key} -> {value}")
    print(f"would write: {registry.index_path}")
    print(f"would write: {registry.active_path}")
    print("production_promotion: false")
    print("baseline_update: false")


def _register_candidate_rows(registry: ModelRegistry, plan: CandidateRegistrationPlan) -> None:
    registry.register(
        algorithm="ppo",
        path=plan.ppo_path,
        agent_role=PPO_AGENT_ROLE,
        status="candidate",
        metrics=plan.metrics,
        metadata=plan.ppo_metadata,
    )
    registry.register(
        algorithm="dqn",
        path=plan.dqn_path,
        agent_role=DQN_AGENT_ROLE,
        status="candidate",
        metrics=plan.metrics,
        metadata=plan.dqn_metadata,
    )


def _print_candidate_dry_run(plan: CandidateRegistrationPlan, registry: ModelRegistry) -> None:
    print("Dry-run: would register Torch joint candidate rows.")
    print(f"logical_model_id: {plan.ppo_metadata['logical_model_id']}")
    print(f"ppo:{PPO_AGENT_ROLE} -> {plan.ppo_path}")
    print(f"dqn:{DQN_AGENT_ROLE} -> {plan.dqn_path}")
    print(f"joint_checkpoint: {plan.ppo_metadata['joint_checkpoint']}")
    print(f"dqn_architecture: {plan.ppo_metadata['dqn_architecture']}")
    if "hierarchical_init_method" in plan.ppo_metadata:
        print(f"hierarchical_init_method: {plan.ppo_metadata['hierarchical_init_method']}")
    print(f"training_step: {plan.ppo_metadata['training_step']}")
    print(f"exact_resume_capable: {str(plan.ppo_metadata['exact_resume_capable']).lower()}")
    print(f"eval_output_dir: {plan.ppo_metadata['eval_output_dir']}")
    print(f"eval_verdict: {plan.ppo_metadata['eval_verdict']}")
    print(f"hard_blocker_status: {plan.ppo_metadata['hard_blocker_status']}")
    print(f"long_run_gate_verdict: {plan.ppo_metadata['long_run_gate_verdict']}")
    print(f"registry_scope: {plan.ppo_metadata['registry_scope']}")
    print(f"would append candidate rows to: {registry.index_path}")
    print("would not write: active_models.json")
    print("production_ready: false")
    print("baseline_update: false")


def _validate_joint_checkpoint_for_candidate(
    path: Path,
    *,
    logical_model_id: str,
    expected_dqn_architecture: str | None,
    expected_hierarchical_init_method: str | None,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("joint checkpoint must be a mapping.")
    _require_checkpoint_value(checkpoint, "checkpoint_version", CHECKPOINT_VERSION)
    _require_checkpoint_value(checkpoint, "artifact_kind", JOINT_FINAL_ARTIFACT_KIND)
    training_step = _require_checkpoint_int(checkpoint, "global_step", minimum=1)
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("joint checkpoint config metadata must be a mapping.")
    contract = str(config.get("mdp_contract_version", ""))
    if contract != EXPECTED_MDP_CONTRACT:
        raise ValueError(f"contract mismatch: expected {EXPECTED_MDP_CONTRACT!r}, got {contract!r}.")
    obs_dim = _require_checkpoint_int(checkpoint, "observation_dim", expected=OBSERVATION_DIM)
    action_count = _require_checkpoint_int(checkpoint, "discrete_action_count", expected=DISCRETE_ACTION_COUNT)
    external_count = _require_checkpoint_int(
        checkpoint,
        "external_discrete_action_count",
        expected=DISCRETE_ACTION_COUNT,
    )

    dqn_architecture = str(checkpoint.get("dqn_architecture", ""))
    if expected_dqn_architecture is not None and dqn_architecture != expected_dqn_architecture:
        raise ValueError(
            f"dqn_architecture mismatch: expected {expected_dqn_architecture!r}, got {dqn_architecture!r}."
        )
    if _looks_hierarchical_candidate(logical_model_id, dqn_architecture):
        if dqn_architecture != HIERARCHICAL_DQN_ARCHITECTURE:
            raise ValueError(
                f"dqn_architecture must be {HIERARCHICAL_DQN_ARCHITECTURE!r} for hierarchical candidate, "
                f"got {dqn_architecture!r}."
            )
        init_method = str(checkpoint.get("hierarchical_init_method", ""))
        expected_init = expected_hierarchical_init_method or "flat_teacher_distillation_v1"
        if init_method != expected_init:
            raise ValueError(f"hierarchical_init_method mismatch: expected {expected_init!r}, got {init_method!r}.")
        internal_heads = checkpoint.get("internal_heads")
        if not isinstance(internal_heads, Mapping) or not internal_heads:
            raise ValueError("internal_heads metadata is required for hierarchical_v1 candidate.")
    else:
        init_method = str(checkpoint.get("hierarchical_init_method", ""))

    resume_state = checkpoint.get("resume_state")
    if not isinstance(resume_state, Mapping) or resume_state.get("exact_resume_capable") is not True:
        raise ValueError("resume_state.exact_resume_capable must be true.")
    if not _has_replay_state(checkpoint):
        raise ValueError("DQN replay state is required for candidate registration.")
    if not _has_rng_state(checkpoint):
        raise ValueError("RNG state is required for candidate registration.")

    return {
        "training_step": training_step,
        "dqn_architecture": dqn_architecture,
        "hierarchical_init_method": init_method,
        "contract": contract,
        "obs_dim": obs_dim,
        "action_count": action_count,
        "external_discrete_action_count": external_count,
        "exact_resume_capable": True,
    }


def _validate_eval_output_dir(path: Path) -> dict[str, Any]:
    summary_path = path / "scenario_summary.json"
    episode_path = path / "episode_metrics.jsonl"
    _require_existing_path(summary_path, "scenario_summary.json")
    _require_existing_path(episode_path, "episode_metrics.jsonl")
    summary = _load_json(summary_path)
    scenarios = summary.get("scenarios") if isinstance(summary, Mapping) else None
    if not isinstance(scenarios, list):
        raise ValueError("scenario_summary.json must contain a scenarios list.")
    if len(scenarios) != DEFAULT_SCENARIO_COUNT:
        raise ValueError(f"expected {DEFAULT_SCENARIO_COUNT} scenarios, got {len(scenarios)}.")
    expected_episode_rows = 0
    for scenario in scenarios:
        if not isinstance(scenario, Mapping):
            raise ValueError("scenario entries must be mappings.")
        scenario_name = str(scenario.get("scenario_id") or scenario.get("name") or "unknown")
        if scenario.get("verdict") != "PASS":
            raise ValueError(f"eval_verdict for scenario {scenario_name!r} must be PASS.")
        if scenario.get("hard_blocker_verdict") != "PASS":
            raise ValueError(f"hard_blocker_verdict for scenario {scenario_name!r} must be PASS.")
        expected_episode_rows += int(scenario.get("episodes", 0) or 0)
        for field_info in GLOBAL_FATAL_HARD_BLOCKER_FIELDS:
            field = field_info[0] if isinstance(field_info, tuple) else str(field_info)
            value = scenario.get(field, 0)
            if isinstance(value, bool):
                blocked = value
            else:
                blocked = int(value or 0) != 0
            if blocked:
                raise ValueError(f"hard blocker {field} is nonzero in scenario {scenario_name!r}.")

    episode_rows = [line for line in episode_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(episode_rows) != expected_episode_rows:
        raise ValueError(f"expected {expected_episode_rows} episode rows, got {len(episode_rows)}.")
    return {"scenario_count": len(scenarios), "episode_rows": len(episode_rows)}


def _validate_gate_result(path: Path) -> dict[str, str]:
    data = _load_json(path)
    if not isinstance(data, Mapping):
        raise ValueError("gate result must be a mapping.")
    decision = str(data.get("decision", ""))
    if decision != "PASS":
        raise ValueError(f"long_run_gate_verdict must be PASS, got {decision!r}.")
    fatal_failures = data.get("fatal_failures", [])
    if fatal_failures:
        raise ValueError(f"long-run gate fatal_failures must be empty, got {fatal_failures!r}.")
    return {"long_run_gate_verdict": decision}


def _load_json(path: Path) -> Any:
    data = path.read_bytes()
    for encoding in ("utf-8", "utf-16", "utf-16-le"):
        try:
            return json.loads(data.decode(encoding))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError(f"could not decode JSON file: {path}")


def _require_checkpoint_value(checkpoint: Mapping[str, Any], key: str, expected: str) -> None:
    actual = str(checkpoint.get(key, ""))
    if actual != expected:
        raise ValueError(f"{key} mismatch: expected {expected!r}, got {actual!r}.")


def _require_checkpoint_int(
    checkpoint: Mapping[str, Any],
    key: str,
    *,
    expected: int | None = None,
    minimum: int | None = None,
) -> int:
    if key not in checkpoint:
        raise ValueError(f"{key} metadata is required.")
    actual = int(checkpoint[key])
    if expected is not None and actual != expected:
        raise ValueError(f"{key} mismatch: expected {expected}, got {actual}.")
    if minimum is not None and actual < minimum:
        raise ValueError(f"{key} must be >= {minimum}, got {actual}.")
    return actual


def _looks_hierarchical_candidate(logical_model_id: str, dqn_architecture: str) -> bool:
    return "hierarchical" in logical_model_id.lower() or dqn_architecture == HIERARCHICAL_DQN_ARCHITECTURE


def _has_replay_state(checkpoint: Mapping[str, Any]) -> bool:
    return any(key in checkpoint and checkpoint[key] is not None for key in ("dqn_replay_buffer_state", "replay_buffer_state", "dqn_replay_state"))


def _has_rng_state(checkpoint: Mapping[str, Any]) -> bool:
    return any(key in checkpoint and checkpoint[key] is not None for key in ("rng_state", "rng_states"))


def _require_existing_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} artifact path does not exist: {path}")


def _require_metadata_value(record: ModelRecord, key: str, expected: str, *, algorithm: Algorithm) -> None:
    actual = _metadata_text(record, key)
    if actual.lower() != expected.lower():
        raise ValueError(f"{algorithm} metadata {key} mismatch: expected {expected!r}, got {actual!r}.")


def _require_metadata_int(record: ModelRecord, key: str, expected: int, *, algorithm: Algorithm) -> None:
    if key not in record.metadata:
        raise ValueError(f"{algorithm} metadata field {key!r} is required.")
    actual = int(record.metadata[key])
    if actual != expected:
        raise ValueError(f"{algorithm} metadata {key} mismatch: expected {expected}, got {actual}.")


def _require_positive_metadata_int(record: ModelRecord, key: str, *, algorithm: Algorithm) -> None:
    if key not in record.metadata:
        raise ValueError(f"{algorithm} metadata field {key!r} is required.")
    actual = int(record.metadata[key])
    if actual <= 0:
        raise ValueError(f"{algorithm} metadata {key} must be positive, got {actual}.")


def _require_false_or_absent(record: ModelRecord, key: str, *, algorithm: Algorithm) -> None:
    if key not in record.metadata:
        return
    value = record.metadata[key]
    if value is not False:
        raise ValueError(f"{algorithm} metadata {key} must be false or absent.")


def _metadata_text(record: ModelRecord, key: str) -> str:
    value = record.metadata.get(key)
    if value is None or str(value) == "":
        raise ValueError(f"{record.algorithm} metadata field {key!r} is required.")
    return str(value)


def _read_active_json(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"active registry file must contain an object: {path}")
    return {str(key): str(value) for key, value in data.items()}


def _write_active_json_atomic(path: Path, active: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temp_path = Path(handle.name)
        json.dump(active, handle, indent=2)
        handle.write("\n")
    temp_path.replace(path)


def _active_key(algorithm: Algorithm, agent_role: str) -> str:
    return f"{algorithm}:{agent_role}"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - exercised by direct CLI use.
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
