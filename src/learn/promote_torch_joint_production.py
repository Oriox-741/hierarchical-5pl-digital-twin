"""Copy-only production promotion for reviewed Torch joint active models."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from tempfile import NamedTemporaryFile
from typing import Any

from src.act.action_projector import CONTINUOUS_ACTION_DIM
from src.act.discrete_action_mapper import DISCRETE_ACTION_COUNT
from src.act.observation_builder import OBSERVATION_DIM
from src.learn.model_registry import Algorithm, ModelRecord, ModelRegistry
from src.orchestration.policy_service import DQN_AGENT_ROLE, PPO_AGENT_ROLE
from src.orchestration.torch_joint_runtime import EXPECTED_MDP_CONTRACT


EXPECTED_LOGICAL_MODEL_ID = "joint_torch_v5_balanced_retention_ft_200k_20260608"
EXPECTED_FRAMEWORK = "torch_joint"
EXPECTED_RUNTIME_LOADER = "joint_torch"
EXPECTED_TRAINING_STEP = 200000
EXPECTED_PARENT_STEP = 1000000
PPO_ARTIFACT_KIND = "ppo_final"
DQN_ARTIFACT_KIND = "dqn_final"

RESIDUAL_WATCHES = (
    "route_disruption_congestion.high_resilience_selected_when_not_best_count=411",
    "demand_spike_volatility.true_lateness_pressure=0.17686320800109925",
    "mixed_success_route_failure_steps=1 in baseline_normal, lead_time_volatility, mixed_stress",
)


@dataclass(frozen=True, slots=True)
class ProductionPromotionPlan:
    logical_model_id: str
    ppo_record: ModelRecord
    dqn_record: ModelRecord
    joint_checkpoint: Path
    ppo_artifact: Path
    dqn_artifact: Path
    eval_output_dir: Path
    production_dir: Path

    @property
    def production_joint_checkpoint(self) -> Path:
        return self.production_dir / self.joint_checkpoint.name

    @property
    def production_ppo_artifact(self) -> Path:
        return self.production_dir / self.ppo_artifact.name

    @property
    def production_dqn_artifact(self) -> Path:
        return self.production_dir / self.dqn_artifact.name

    @property
    def production_manifest(self) -> Path:
        return self.production_dir / "production_manifest.json"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.dry_run and args.promote:
        raise ValueError("--dry-run and --promote are mutually exclusive.")

    promote = bool(args.promote)
    if promote:
        _require_promotion_approval(args)

    registry_dir = Path(args.registry_dir)
    if not registry_dir.exists():
        raise FileNotFoundError(f"registry directory does not exist: {registry_dir}")
    registry = ModelRegistry(registry_dir)
    plan = _build_plan(
        registry,
        logical_model_id=args.logical_model_id,
        production_root=Path(args.production_root),
        production_dir_name=args.production_dir_name,
    )
    _validate_target_absent(plan.production_dir)

    if promote:
        _execute_copy_only_promotion(
            plan,
            human_approver=args.human_approver.strip(),
        )
        print("PRODUCTION_PROMOTION_COMPLETE")
    else:
        _print_dry_run(plan, args)
        print("DRY_RUN_OK")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy active Torch joint artifacts into a production directory.")
    parser.add_argument("--logical-model-id", required=True)
    parser.add_argument("--registry-dir", default="models/registry")
    parser.add_argument("--production-root", default="models/production")
    parser.add_argument("--production-dir-name")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print planned copy without writes.")
    parser.add_argument("--promote", action="store_true", help="Copy production artifacts and write manifest.")
    parser.add_argument("--human-approver", default="")
    parser.add_argument("--accept-residual-watches", action="store_true")
    return parser.parse_args(argv)


def _require_promotion_approval(args: argparse.Namespace) -> None:
    if not str(args.human_approver).strip():
        raise ValueError("--promote requires a non-empty human approver.")
    if not args.accept_residual_watches:
        raise ValueError("--promote requires residual watch acceptance.")


def _build_plan(
    registry: ModelRegistry,
    *,
    logical_model_id: str,
    production_root: Path,
    production_dir_name: str | None,
) -> ProductionPromotionPlan:
    if logical_model_id != EXPECTED_LOGICAL_MODEL_ID:
        raise ValueError(
            f"logical_model_id mismatch: expected {EXPECTED_LOGICAL_MODEL_ID!r}, got {logical_model_id!r}."
        )
    active = _read_json_object(registry.active_path)
    _validate_active_keys(active)
    ppo_path = Path(str(active[_active_key("ppo", PPO_AGENT_ROLE)]))
    dqn_path = Path(str(active[_active_key("dqn", DQN_AGENT_ROLE)]))
    ppo_record = _active_record_for_mapping(
        registry,
        algorithm="ppo",
        agent_role=PPO_AGENT_ROLE,
        path=ppo_path,
        logical_model_id=logical_model_id,
    )
    dqn_record = _active_record_for_mapping(
        registry,
        algorithm="dqn",
        agent_role=DQN_AGENT_ROLE,
        path=dqn_path,
        logical_model_id=logical_model_id,
    )
    _validate_active_record(
        ppo_record,
        algorithm="ppo",
        agent_role=PPO_AGENT_ROLE,
        artifact_kind=PPO_ARTIFACT_KIND,
        logical_model_id=logical_model_id,
    )
    _validate_active_record(
        dqn_record,
        algorithm="dqn",
        agent_role=DQN_AGENT_ROLE,
        artifact_kind=DQN_ARTIFACT_KIND,
        logical_model_id=logical_model_id,
    )
    if _metadata_text(ppo_record, "joint_checkpoint") != _metadata_text(dqn_record, "joint_checkpoint"):
        raise ValueError("active PPO/DQN joint_checkpoint mismatch.")
    if _metadata_text(ppo_record, "eval_output_dir") != _metadata_text(dqn_record, "eval_output_dir"):
        raise ValueError("active PPO/DQN eval_output_dir mismatch.")

    joint_checkpoint = Path(_metadata_text(ppo_record, "joint_checkpoint"))
    eval_output_dir = Path(_metadata_text(ppo_record, "eval_output_dir"))
    ppo_artifact = Path(ppo_record.path)
    dqn_artifact = Path(dqn_record.path)
    _require_existing_path(joint_checkpoint, "joint checkpoint")
    _require_existing_path(ppo_artifact, "PPO artifact")
    _require_existing_path(dqn_artifact, "DQN artifact")
    _require_existing_path(eval_output_dir, "eval output dir")
    _validate_copy_sources(joint_checkpoint, ppo_artifact, dqn_artifact)

    return ProductionPromotionPlan(
        logical_model_id=logical_model_id,
        ppo_record=ppo_record,
        dqn_record=dqn_record,
        joint_checkpoint=joint_checkpoint,
        ppo_artifact=ppo_artifact,
        dqn_artifact=dqn_artifact,
        eval_output_dir=eval_output_dir,
        production_dir=production_root / _safe_production_dir_name(production_dir_name or logical_model_id),
    )


def _validate_active_keys(active: Mapping[str, Any]) -> None:
    expected = {
        _active_key("ppo", PPO_AGENT_ROLE),
        _active_key("dqn", DQN_AGENT_ROLE),
    }
    actual = set(active)
    missing = expected - actual
    if missing:
        missing_text = ", ".join(sorted(missing))
        if _active_key("dqn", DQN_AGENT_ROLE) in missing:
            raise FileNotFoundError(f"active dqn mapping for role {DQN_AGENT_ROLE} is required.")
        raise FileNotFoundError(f"active registry mapping is required: {missing_text}.")
    extra = actual - expected
    if extra:
        extra_text = ", ".join(sorted(extra))
        raise ValueError(f"active registry must contain only intended role mappings; extra keys: {extra_text}.")


def _active_record_for_mapping(
    registry: ModelRegistry,
    *,
    algorithm: Algorithm,
    agent_role: str,
    path: Path,
    logical_model_id: str,
) -> ModelRecord:
    matches = [
        record
        for record in registry.records(algorithm=algorithm, agent_role=agent_role, status="active")
        if Path(record.path) == path and record.metadata.get("logical_model_id") == logical_model_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one active {algorithm} record for {logical_model_id!r} at {path}, got {len(matches)}."
        )
    return matches[0]


def _validate_active_record(
    record: ModelRecord,
    *,
    algorithm: Algorithm,
    agent_role: str,
    artifact_kind: str,
    logical_model_id: str,
) -> None:
    if record.status != "active":
        raise ValueError(f"{algorithm} record status must be active, got {record.status!r}.")
    if record.algorithm != algorithm:
        raise ValueError(f"{algorithm} record algorithm mismatch: got {record.algorithm!r}.")
    _require_metadata_value(record, "logical_model_id", logical_model_id)
    _require_metadata_value(record, "agent_role", agent_role)
    _require_metadata_value(record, "framework", EXPECTED_FRAMEWORK)
    _require_metadata_value(record, "runtime_loader", EXPECTED_RUNTIME_LOADER)
    _require_metadata_value(record, "artifact_kind", artifact_kind)
    _require_metadata_value(record, "contract", EXPECTED_MDP_CONTRACT)
    _require_metadata_int(record, "obs_dim", OBSERVATION_DIM)
    _require_metadata_int(record, "action_count", DISCRETE_ACTION_COUNT)
    _require_metadata_int(record, "training_step", EXPECTED_TRAINING_STEP)
    _require_metadata_int(record, "parent_step", EXPECTED_PARENT_STEP)
    _require_metadata_value(record, "eval_verdict", "PASS")
    _require_metadata_value(record, "hard_blocker_status", "zero")
    _require_false_or_absent(record, "production_promotion")
    _require_false_or_absent(record, "production_ready")
    _require_false_or_absent(record, "baseline_update")


def _execute_copy_only_promotion(plan: ProductionPromotionPlan, *, human_approver: str) -> None:
    staging_dir = _staging_dir_for(plan.production_dir)
    if staging_dir.exists():
        raise FileExistsError(f"production staging target already exists: {staging_dir}")
    plan.production_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir()
    try:
        copy_targets = _copy_targets(plan, target_dir=staging_dir)
        for source, destination in copy_targets:
            shutil.copy2(source, destination)
        _write_manifest_atomic(
            staging_dir / "production_manifest.json",
            _manifest(plan, human_approver=human_approver),
        )
        staging_dir.replace(plan.production_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise


def _copy_targets(plan: ProductionPromotionPlan, *, target_dir: Path) -> tuple[tuple[Path, Path], ...]:
    return (
        (plan.joint_checkpoint, target_dir / plan.joint_checkpoint.name),
        (plan.ppo_artifact, target_dir / plan.ppo_artifact.name),
        (plan.dqn_artifact, target_dir / plan.dqn_artifact.name),
    )


def _validate_copy_sources(joint_checkpoint: Path, ppo_artifact: Path, dqn_artifact: Path) -> None:
    sources = (joint_checkpoint, ppo_artifact, dqn_artifact)
    for source in sources:
        if source.suffix.lower() != ".pt":
            raise ValueError(f"production promotion sources must be .pt artifacts: {source}")
    if len({source.name for source in sources}) != len(sources):
        raise ValueError("production artifact destination basenames must be unique.")
    if len({str(source.resolve()) for source in sources}) != len(sources):
        raise ValueError("production promotion requires three distinct source artifacts.")


def _manifest(plan: ProductionPromotionPlan, *, human_approver: str) -> dict[str, Any]:
    return {
        "logical_model_id": plan.logical_model_id,
        "active_ppo_model_id": plan.ppo_record.model_id,
        "active_dqn_model_id": plan.dqn_record.model_id,
        "source_joint_checkpoint": str(plan.joint_checkpoint),
        "source_ppo_artifact": str(plan.ppo_artifact),
        "source_dqn_artifact": str(plan.dqn_artifact),
        "production_joint_checkpoint": str(plan.production_joint_checkpoint),
        "production_ppo_artifact": str(plan.production_ppo_artifact),
        "production_dqn_artifact": str(plan.production_dqn_artifact),
        "contract": _metadata_text(plan.ppo_record, "contract"),
        "observation_dim": _metadata_int(plan.ppo_record, "obs_dim"),
        "continuous_action_dim": CONTINUOUS_ACTION_DIM,
        "discrete_action_count": _metadata_int(plan.ppo_record, "action_count"),
        "training_step": _metadata_int(plan.ppo_record, "training_step"),
        "parent_step": _metadata_int(plan.ppo_record, "parent_step"),
        "eval_output_dir": str(plan.eval_output_dir),
        "eval_verdict": _metadata_text(plan.ppo_record, "eval_verdict"),
        "hard_blocker_status": _metadata_text(plan.ppo_record, "hard_blocker_status"),
        "residual_watches": list(RESIDUAL_WATCHES),
        "residual_watches_accepted": True,
        "human_approver": human_approver,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "registry_active_scope": True,
        "production_promotion_scope": "copy_only",
        "baseline_update": False,
        "db_mutation": False,
    }


def _write_manifest_atomic(path: Path, manifest: dict[str, Any]) -> None:
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temp_path = Path(handle.name)
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    temp_path.replace(path)


def _print_dry_run(plan: ProductionPromotionPlan, args: argparse.Namespace) -> None:
    print("Dry-run: would copy active Torch joint artifacts into production.")
    print(f"logical_model_id: {plan.logical_model_id}")
    for source, destination in _copy_targets(plan, target_dir=plan.production_dir):
        print(f"planned copy: {source} -> {destination}")
    print(f"planned manifest: {plan.production_manifest}")
    print("residual watches:")
    for watch in RESIDUAL_WATCHES:
        print(f"- {watch}")
    print(f"human_approver_missing: {str(not bool(str(args.human_approver).strip())).lower()}")
    print(f"residual_watches_accepted: {str(bool(args.accept_residual_watches)).lower()}")
    print("registry_mutation: false")
    print("baseline_update: false")
    print("db_mutation: false")


def _validate_target_absent(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"production target already exists: {path}")


def _safe_production_dir_name(name: str) -> str:
    if not name or name in {".", ".."}:
        raise ValueError("--production-dir-name must be a single safe directory name.")
    posix = PurePosixPath(name)
    windows = PureWindowsPath(name)
    if posix.is_absolute() or windows.is_absolute():
        raise ValueError("--production-dir-name must be relative.")
    if len(posix.parts) != 1 or len(windows.parts) != 1:
        raise ValueError("--production-dir-name must not contain path separators or parent traversal.")
    if posix.parts[0] in {".", ".."} or windows.parts[0] in {".", ".."}:
        raise ValueError("--production-dir-name must be a single safe directory name.")
    return name


def _require_existing_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return {str(key): value for key, value in data.items()}


def _require_metadata_value(record: ModelRecord, key: str, expected: str) -> None:
    actual = _metadata_text(record, key)
    if actual.lower() != expected.lower():
        raise ValueError(f"{record.algorithm} metadata {key} mismatch: expected {expected!r}, got {actual!r}.")


def _require_metadata_int(record: ModelRecord, key: str, expected: int) -> None:
    actual = _metadata_int(record, key)
    if actual != expected:
        raise ValueError(f"{record.algorithm} metadata {key} mismatch: expected {expected}, got {actual}.")


def _require_false_or_absent(record: ModelRecord, key: str) -> None:
    if key not in record.metadata:
        return
    if record.metadata[key] is not False:
        raise ValueError(f"{record.algorithm} metadata {key} must be false or absent.")


def _metadata_text(record: ModelRecord, key: str) -> str:
    value = record.metadata.get(key)
    if value is None or str(value) == "":
        raise ValueError(f"{record.algorithm} metadata field {key!r} is required.")
    return str(value)


def _metadata_int(record: ModelRecord, key: str) -> int:
    if key not in record.metadata:
        raise ValueError(f"{record.algorithm} metadata field {key!r} is required.")
    return int(record.metadata[key])


def _staging_dir_for(production_dir: Path) -> Path:
    return production_dir.parent / f".{production_dir.name}.tmp"


def _active_key(algorithm: Algorithm, agent_role: str) -> str:
    return f"{algorithm}:{agent_role}"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - direct CLI behavior.
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
