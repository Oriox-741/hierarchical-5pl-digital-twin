from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Sequence


EXPECTED_LOGICAL_MODEL_ID = "joint_torch_v5_prod_hierarchical_v1_1m_20260611"
EXPECTED_JOINT_FILENAME = "joint_torch_latest.pt"
EXPECTED_PPO_FILENAME = "ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt"
EXPECTED_DQN_FILENAME = "dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt"

DEFAULT_ACTIVE_MODELS_PATH = Path("models/registry/active_models.json")
DEFAULT_MODELS_JSONL_PATH = Path("models/registry/models.jsonl")
DEFAULT_PRODUCTION_DIR = Path("models/production") / EXPECTED_LOGICAL_MODEL_ID
DEFAULT_PRODUCTION_MANIFEST_PATH = DEFAULT_PRODUCTION_DIR / "production_manifest.json"
DEFAULT_DASHBOARD_PATH = Path("docs/00_PROJECT_DASHBOARD.md")

DEFAULT_ACTIVE_PPO_PATH = (
    r"models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611"
    r"\ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt"
)
DEFAULT_ACTIVE_DQN_PATH = (
    r"models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611"
    r"\dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt"
)

DEFAULT_ACTIVE_MODELS_SHA256 = "B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A"
DEFAULT_MODELS_JSONL_SHA256 = "935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1"
DEFAULT_PRODUCTION_MANIFEST_SHA256 = "FDAF8DE495D90BDFD69BCC2EE3772EB6A09661E104A5127EA3CB01C6D02CF99A"
DEFAULT_PRODUCTION_FILE_SHA256 = {
    EXPECTED_JOINT_FILENAME: "C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE",
    EXPECTED_PPO_FILENAME: "2D4CD9130492F7E15E9925177A9B82DF959ECF09534ADE649DC0D81D39410996",
    EXPECTED_DQN_FILENAME: "9DB010EB89ADB7262F3B85902507A13DB45AD7523940B65C3769CEFC96CA470D",
    "production_manifest.json": DEFAULT_PRODUCTION_MANIFEST_SHA256,
}
DEFAULT_PRODUCTION_DIR_FILES = frozenset(DEFAULT_PRODUCTION_FILE_SHA256)

FORBIDDEN_PROCESS_PATTERNS = (
    "train_joint",
    "evaluate_real_world_scenarios",
    "check_long_run_gate",
    "offline_scenarios",
    "aws s3 sync",
    "aws s3 cp",
)

CLASSIFICATION_CLEAN = "ARTIFACT_HEALTH_CLEAN"
CLASSIFICATION_WARNINGS = "ARTIFACT_HEALTH_WARNINGS_ONLY"
CLASSIFICATION_ACTIVE_REGISTRY = "ARTIFACT_HEALTH_ACTIVE_REGISTRY_MISMATCH"
CLASSIFICATION_MODELS_JSONL = "ARTIFACT_HEALTH_MODELS_JSONL_MISMATCH"
CLASSIFICATION_MANIFEST = "ARTIFACT_HEALTH_PRODUCTION_MANIFEST_MISMATCH"
CLASSIFICATION_HASH = "ARTIFACT_HEALTH_PRODUCTION_HASH_MISMATCH"
CLASSIFICATION_RUNTIME = "ARTIFACT_HEALTH_RUNTIME_SMOKE_FAILED"
CLASSIFICATION_PROTECTED = "ARTIFACT_HEALTH_PROTECTED_PATH_DRIFT"
CLASSIFICATION_PROCESS = "ARTIFACT_HEALTH_FORBIDDEN_PROCESS_FOUND"
CLASSIFICATION_BLOCKED = "ARTIFACT_HEALTH_BLOCKED"


@dataclass(frozen=True, slots=True)
class ExpectedProductionIdentity:
    logical_model_id: str
    active_ppo_id: str
    active_dqn_id: str
    candidate_ppo_id: str
    candidate_dqn_id: str
    active_ppo_path: str
    active_dqn_path: str
    dqn_architecture: str
    hierarchical_init_method: str
    contract: str
    observation_dim: int
    action_count: int
    training_step: int


@dataclass(frozen=True, slots=True)
class PathProfile:
    path: Path
    file_count: int
    total_bytes: int


@dataclass(frozen=True, slots=True)
class ProcessEntry:
    pid: int
    name: str
    command_line: str


@dataclass(frozen=True, slots=True)
class HealthReportConfig:
    active_models_path: Path
    models_jsonl_path: Path
    production_dir: Path
    production_manifest_path: Path
    dashboard_path: Path
    output_path: Path | None
    expected_identity: ExpectedProductionIdentity
    expected_active_models_sha256: str
    expected_models_jsonl_sha256: str
    expected_production_manifest_sha256: str
    expected_production_file_sha256: dict[str, str]
    expected_production_dir_files: set[str]
    expected_protected_profiles: dict[str, PathProfile]


DEFAULT_EXPECTED_IDENTITY = ExpectedProductionIdentity(
    logical_model_id=EXPECTED_LOGICAL_MODEL_ID,
    active_ppo_id="17ba1d28-0054-4f7c-ae9a-34cd305ebb89",
    active_dqn_id="f87e10d6-479f-44fc-99d1-6925bc9cb346",
    candidate_ppo_id="3befa11c-e853-4d0f-a951-c2fbbb2a9898",
    candidate_dqn_id="71af6701-ac3a-40f2-bdd9-cc80868d5a1c",
    active_ppo_path=DEFAULT_ACTIVE_PPO_PATH,
    active_dqn_path=DEFAULT_ACTIVE_DQN_PATH,
    dqn_architecture="hierarchical_v1",
    hierarchical_init_method="flat_teacher_distillation_v1",
    contract="physical_reality_v5_route_candidate_visibility",
    observation_dim=73,
    action_count=48,
    training_step=1_000_000,
)

DEFAULT_PROTECTED_PROFILES = {
    "models/baselines": PathProfile(Path("models/baselines"), 2692, 7_392_576_274),
    "db": PathProfile(Path("db"), 7, 28_330),
    "models/checkpoints": PathProfile(Path("models/checkpoints"), 787, 4_920_830_666),
    "models/production": PathProfile(Path("models/production"), 8, 328_265_276),
    "models/eval": PathProfile(Path("models/eval"), 128, 388_267_659),
}


def build_artifact_health_report(
    config: HealthReportConfig,
    *,
    process_entries: Sequence[ProcessEntry] | None = None,
    run_runtime_smoke: bool = True,
) -> dict[str, Any]:
    if process_entries is None:
        process_entries = collect_process_entries()

    report: dict[str, Any] = {
        "report_metadata": {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "script": str(Path(__file__).as_posix()),
            "runtime_smoke_requested": bool(run_runtime_smoke),
        },
        "expected_production_identity": _jsonable(config.expected_identity),
        "active_registry_checks": check_active_registry(config),
        "models_jsonl_checks": check_models_jsonl(config),
        "production_manifest_checks": check_production_manifest(config),
        "production_file_hash_checks": check_production_file_hashes(config),
        "protected_path_profile_checks": check_protected_path_profiles(config),
        "dashboard_consistency_checks": check_dashboard_consistency(config),
        "process_scan_result": check_processes(process_entries),
        "runtime_smoke_result": run_runtime_smoke_check(config) if run_runtime_smoke else _skipped_runtime_smoke(),
    }
    report["final_classification"] = classify_report(report)
    return report


def check_active_registry(config: HealthReportConfig) -> dict[str, Any]:
    issues: list[str] = []
    expected = {
        "ppo:continuous_control": config.expected_identity.active_ppo_path,
        "dqn:tactical_dispatch": config.expected_identity.active_dqn_path,
    }
    try:
        active_models = _read_json(config.active_models_path)
        file_hash = _sha256(config.active_models_path)
    except Exception as exc:  # pragma: no cover - exercised through classification behavior.
        return _blocked_check(str(exc))

    if file_hash != config.expected_active_models_sha256.upper():
        issues.append(
            f"active_models_sha256 expected {config.expected_active_models_sha256.upper()} got {file_hash}"
        )
    if set(active_models) != set(expected):
        issues.append(f"active role keys expected {sorted(expected)} got {sorted(active_models)}")
    for role, expected_path in expected.items():
        actual_path = active_models.get(role)
        if actual_path != expected_path:
            issues.append(f"{role} expected {expected_path!r} got {actual_path!r}")

    return {
        "status": _pass_fail(issues),
        "path": str(config.active_models_path),
        "sha256": file_hash,
        "expected_sha256": config.expected_active_models_sha256.upper(),
        "expected_active_models": expected,
        "actual_active_models": active_models,
        "issues": issues,
    }


def check_models_jsonl(config: HealthReportConfig) -> dict[str, Any]:
    issues: list[str] = []
    missing_rows: list[str] = []
    row_issues: dict[str, list[str]] = {}
    try:
        rows = _read_jsonl(config.models_jsonl_path)
        file_hash = _sha256(config.models_jsonl_path)
    except Exception as exc:  # pragma: no cover - exercised through classification behavior.
        return _blocked_check(str(exc), missing_rows=missing_rows, row_issues=row_issues)

    if file_hash != config.expected_models_jsonl_sha256.upper():
        issues.append(f"models_jsonl_sha256 expected {config.expected_models_jsonl_sha256.upper()} got {file_hash}")

    identity = config.expected_identity
    expected_rows = [
        (identity.candidate_ppo_id, "ppo", "candidate", identity.active_ppo_path, "ppo_final", "continuous_control"),
        (identity.candidate_dqn_id, "dqn", "candidate", identity.active_dqn_path, "dqn_final", "tactical_dispatch"),
        (identity.active_ppo_id, "ppo", "active", identity.active_ppo_path, "ppo_final", "continuous_control"),
        (identity.active_dqn_id, "dqn", "active", identity.active_dqn_path, "dqn_final", "tactical_dispatch"),
    ]
    by_id = {str(row.get("model_id")): row for row in rows if isinstance(row, dict)}
    for model_id, algorithm, status, path, artifact_kind, agent_role in expected_rows:
        row = by_id.get(model_id)
        if not row:
            missing_rows.append(model_id)
            continue
        problems = _validate_registry_row(
            row,
            algorithm=algorithm,
            status=status,
            path=path,
            artifact_kind=artifact_kind,
            agent_role=agent_role,
            identity=identity,
        )
        if problems:
            row_issues[model_id] = problems

    if missing_rows:
        issues.append(f"missing expected rows: {missing_rows}")
    if row_issues:
        issues.append(f"row metadata mismatches: {sorted(row_issues)}")
    if not _has_old_production_active_rows(rows, identity.logical_model_id):
        issues.append("old production active PPO/DQN rows are not both present for audit")

    return {
        "status": _pass_fail(issues),
        "path": str(config.models_jsonl_path),
        "sha256": file_hash,
        "expected_sha256": config.expected_models_jsonl_sha256.upper(),
        "row_count": len(rows),
        "missing_rows": missing_rows,
        "row_issues": row_issues,
        "old_production_active_rows_present": _has_old_production_active_rows(rows, identity.logical_model_id),
        "issues": issues,
    }


def check_production_manifest(config: HealthReportConfig) -> dict[str, Any]:
    issues: list[str] = []
    try:
        manifest = _read_json(config.production_manifest_path)
        file_hash = _sha256(config.production_manifest_path)
    except Exception as exc:  # pragma: no cover - exercised through classification behavior.
        return _blocked_check(str(exc))

    identity = config.expected_identity
    expected_scalars: dict[str, Any] = {
        "logical_model_id": identity.logical_model_id,
        "active_ppo_id": identity.active_ppo_id,
        "active_dqn_id": identity.active_dqn_id,
        "candidate_ppo_id": identity.candidate_ppo_id,
        "candidate_dqn_id": identity.candidate_dqn_id,
        "dqn_architecture": identity.dqn_architecture,
        "hierarchical_init_method": identity.hierarchical_init_method,
        "contract": identity.contract,
        "observation_dim": identity.observation_dim,
        "action_dim": identity.action_count,
        "training_step": identity.training_step,
        "eval_verdict": "PASS",
        "hard_blocker_status": "zero",
        "long_run_gate_verdict": "PASS",
        "residual_watches_accepted": True,
        "baseline_update": False,
        "db_mutation": False,
        "registry_mutation": False,
    }
    if file_hash != config.expected_production_manifest_sha256.upper():
        issues.append(
            f"production_manifest_sha256 expected {config.expected_production_manifest_sha256.upper()} got {file_hash}"
        )
    for field, expected_value in expected_scalars.items():
        actual_value = manifest.get(field)
        if actual_value != expected_value:
            issues.append(f"{field} expected {expected_value!r} got {actual_value!r}")

    for field in ("source_paths", "production_paths"):
        value = manifest.get(field)
        if not isinstance(value, dict) or not all(name in value for name in _model_filenames()):
            issues.append(f"{field} must contain all model artifact filenames")

    sha_section = manifest.get("sha256", {})
    source_hashes = sha_section.get("source") if isinstance(sha_section, dict) else None
    production_hashes = sha_section.get("production") if isinstance(sha_section, dict) else None
    if not isinstance(source_hashes, dict) or not isinstance(production_hashes, dict):
        issues.append("sha256.source and sha256.production must be mappings")
    else:
        for filename in _model_filenames():
            expected_hash = config.expected_production_file_sha256.get(filename, "").upper()
            source_hash = str(source_hashes.get(filename, "")).upper()
            production_hash = str(production_hashes.get(filename, "")).upper()
            if source_hash != expected_hash:
                issues.append(f"sha256.source.{filename} expected {expected_hash} got {source_hash}")
            if production_hash != expected_hash:
                issues.append(f"sha256.production.{filename} expected {expected_hash} got {production_hash}")
            if source_hash != production_hash:
                issues.append(f"sha256 source/production mismatch for {filename}")

    return {
        "status": _pass_fail(issues),
        "path": str(config.production_manifest_path),
        "sha256": file_hash,
        "expected_sha256": config.expected_production_manifest_sha256.upper(),
        "issues": issues,
    }


def check_production_file_hashes(config: HealthReportConfig) -> dict[str, Any]:
    issues: list[str] = []
    actual_files: set[str] = set()
    actual_hashes: dict[str, str | None] = {}
    if not config.production_dir.exists():
        return _blocked_check(f"production dir missing: {config.production_dir}", actual_files=[], actual_hashes={})

    actual_files = {path.name for path in config.production_dir.iterdir() if path.is_file()}
    if actual_files != config.expected_production_dir_files:
        issues.append(
            f"production dir files expected {sorted(config.expected_production_dir_files)} got {sorted(actual_files)}"
        )

    for filename, expected_hash in sorted(config.expected_production_file_sha256.items()):
        path = config.production_dir / filename
        if not path.exists():
            actual_hashes[filename] = None
            issues.append(f"{filename} missing")
            continue
        actual_hash = _sha256(path)
        actual_hashes[filename] = actual_hash
        if actual_hash != expected_hash.upper():
            issues.append(f"{filename} expected {expected_hash.upper()} got {actual_hash}")

    return {
        "status": _pass_fail(issues),
        "production_dir": str(config.production_dir),
        "expected_files": sorted(config.expected_production_dir_files),
        "actual_files": sorted(actual_files),
        "expected_sha256": {name: value.upper() for name, value in sorted(config.expected_production_file_sha256.items())},
        "actual_sha256": actual_hashes,
        "issues": issues,
    }


def check_protected_path_profiles(config: HealthReportConfig) -> dict[str, Any]:
    issues: list[str] = []
    profiles: dict[str, dict[str, Any]] = {}
    for label, expected in sorted(config.expected_protected_profiles.items()):
        try:
            actual = _path_profile(expected.path)
        except Exception as exc:
            profiles[label] = {
                "path": str(expected.path),
                "status": "BLOCKED",
                "error": str(exc),
                "expected_file_count": expected.file_count,
                "expected_total_bytes": expected.total_bytes,
            }
            issues.append(f"{label} blocked: {exc}")
            continue
        profile_issues: list[str] = []
        if actual.file_count != expected.file_count:
            profile_issues.append(f"file_count expected {expected.file_count} got {actual.file_count}")
        if actual.total_bytes != expected.total_bytes:
            profile_issues.append(f"total_bytes expected {expected.total_bytes} got {actual.total_bytes}")
        if profile_issues:
            issues.extend(f"{label}: {issue}" for issue in profile_issues)
        profiles[label] = {
            "path": str(expected.path),
            "status": _pass_fail(profile_issues),
            "expected_file_count": expected.file_count,
            "actual_file_count": actual.file_count,
            "expected_total_bytes": expected.total_bytes,
            "actual_total_bytes": actual.total_bytes,
            "issues": profile_issues,
        }

    return {"status": _pass_fail(issues), "profiles": profiles, "issues": issues}


def check_dashboard_consistency(config: HealthReportConfig) -> dict[str, Any]:
    warnings: list[str] = []
    try:
        text = config.dashboard_path.read_text(encoding="utf-8")
    except Exception as exc:
        return {"status": "WARN", "path": str(config.dashboard_path), "warnings": [str(exc)]}

    expected_tokens = [
        config.expected_identity.logical_model_id,
        "docs/releases/20260611_hierarchical_v1_1m_production_handoff",
        "docs/runbooks/20260611_hierarchical_v1_production_monitoring_runbook",
        "docs/runs/20260611_hierarchical_v1_equal_budget_residual_watch_eval",
        "docs/runbooks/20260611_hierarchical_v1_monitoring_kpi_dictionary",
        "docs/runbooks/20260611_codex_model_lifecycle_governance_runbook",
    ]
    for token in expected_tokens:
        if token not in text:
            warnings.append(f"dashboard missing token {token!r}")

    checkpoint_tokens = {
        str(config.production_dir / EXPECTED_JOINT_FILENAME),
        f"{config.production_dir.as_posix()}/{EXPECTED_JOINT_FILENAME}",
    }
    if not any(token in text for token in checkpoint_tokens):
        warnings.append(f"dashboard missing production checkpoint path for {EXPECTED_JOINT_FILENAME}")

    forbidden_implications = (
        "baseline update complete",
        "db cleanup complete",
        "3m run started",
        "5m run started",
        "10m run started",
        "100m run started",
    )
    lower_text = text.lower()
    for token in forbidden_implications:
        if token in lower_text:
            warnings.append(f"dashboard contains forbidden implication {token!r}")

    return {
        "status": "PASS" if not warnings else "WARN",
        "path": str(config.dashboard_path),
        "warnings": warnings,
    }


def check_processes(process_entries: Sequence[ProcessEntry]) -> dict[str, Any]:
    matches = [entry for entry in process_entries if _is_forbidden_process(entry)]
    return {
        "status": "PASS" if not matches else "FAIL",
        "patterns": list(FORBIDDEN_PROCESS_PATTERNS),
        "matches": [_jsonable(entry) for entry in matches],
        "message": "NO_TRAIN_EVAL_GATE_OR_AWS_DOWNLOAD_PROCESS" if not matches else "FORBIDDEN_PROCESS_FOUND",
    }


def run_runtime_smoke_check(config: HealthReportConfig) -> dict[str, Any]:
    try:
        _ensure_repo_root_on_sys_path()
        import numpy as np
        import torch

        from src.orchestration.torch_joint_runtime import load_torch_joint_policy
    except Exception as exc:
        return {"status": "FAIL", "error": f"runtime import failed: {exc}"}

    try:
        joint_path = config.production_dir / EXPECTED_JOINT_FILENAME
        policy = load_torch_joint_policy(joint_path, device=torch.device("cpu"))
        checkpoint = policy.checkpoint
        config_metadata = checkpoint.get("config", {})
        action = policy.predict_joint(
            np.zeros(config.expected_identity.observation_dim, dtype=np.float32),
            deterministic=True,
        )

        issues: list[str] = []
        _expect_field(issues, checkpoint, "checkpoint_version", "torch_joint_policy_v1")
        _expect_field(issues, checkpoint, "artifact_kind", "joint_final")
        _expect_field(issues, checkpoint, "dqn_architecture", config.expected_identity.dqn_architecture)
        _expect_field(issues, checkpoint, "hierarchical_init_method", config.expected_identity.hierarchical_init_method)
        step = checkpoint.get("global_step", checkpoint.get("training_step"))
        if int(step) != config.expected_identity.training_step:
            issues.append(f"global_step/training_step expected {config.expected_identity.training_step} got {step!r}")
        exact_resume = _exact_resume_capable(checkpoint)
        if exact_resume is not True:
            issues.append(f"exact_resume_capable expected True got {exact_resume!r}")
        contract = config_metadata.get("mdp_contract_version") if isinstance(config_metadata, dict) else None
        if contract != config.expected_identity.contract:
            issues.append(f"contract expected {config.expected_identity.contract!r} got {contract!r}")
        if int(checkpoint.get("observation_dim")) != config.expected_identity.observation_dim:
            issues.append("observation_dim metadata mismatch")
        if int(checkpoint.get("discrete_action_count")) != config.expected_identity.action_count:
            issues.append("discrete_action_count metadata mismatch")
        continuous = np.asarray(action.continuous)
        if continuous.shape != (5,):
            issues.append(f"continuous length expected 5 got {continuous.shape}")
        if not np.all(np.isfinite(continuous)):
            issues.append("continuous contains non-finite values")
        if np.any(continuous < -1.0) or np.any(continuous > 1.0):
            issues.append("continuous action outside [-1, 1]")
        if not 0 <= int(action.discrete) < config.expected_identity.action_count:
            issues.append(f"discrete action outside 0..{config.expected_identity.action_count - 1}: {action.discrete}")

        return {
            "status": _pass_fail(issues),
            "checkpoint_path": str(joint_path),
            "dqn_architecture": checkpoint.get("dqn_architecture"),
            "hierarchical_init_method": checkpoint.get("hierarchical_init_method"),
            "global_step": checkpoint.get("global_step"),
            "exact_resume_capable": exact_resume,
            "continuous_length": int(continuous.shape[0]) if continuous.ndim == 1 else list(continuous.shape),
            "continuous_finite": bool(np.all(np.isfinite(continuous))),
            "continuous_bounded": bool(np.all(continuous >= -1.0) and np.all(continuous <= 1.0)),
            "discrete": int(action.discrete),
            "issues": issues,
        }
    except Exception as exc:
        return {"status": "FAIL", "checkpoint_path": str(config.production_dir / EXPECTED_JOINT_FILENAME), "error": str(exc)}


def classify_report(report: dict[str, Any]) -> str:
    if _check_status(report, "BLOCKED"):
        return CLASSIFICATION_BLOCKED
    if report["process_scan_result"]["status"] == "FAIL":
        return CLASSIFICATION_PROCESS
    if report["active_registry_checks"]["status"] == "FAIL":
        return CLASSIFICATION_ACTIVE_REGISTRY
    if report["models_jsonl_checks"]["status"] == "FAIL":
        return CLASSIFICATION_MODELS_JSONL
    if report["production_manifest_checks"]["status"] == "FAIL":
        return CLASSIFICATION_MANIFEST
    if report["production_file_hash_checks"]["status"] == "FAIL":
        return CLASSIFICATION_HASH
    if report["protected_path_profile_checks"]["status"] == "FAIL":
        return CLASSIFICATION_PROTECTED
    if report["runtime_smoke_result"]["status"] == "FAIL":
        return CLASSIFICATION_RUNTIME
    if report["dashboard_consistency_checks"]["status"] == "WARN":
        return CLASSIFICATION_WARNINGS
    return CLASSIFICATION_CLEAN


def collect_process_entries() -> list[ProcessEntry]:
    if sys.platform == "win32":
        return _collect_windows_process_entries()
    return _collect_posix_process_entries()


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    config = _config_from_args(args)
    report = build_artifact_health_report(config, run_runtime_smoke=not args.skip_runtime_smoke)
    if config.output_path is None:
        raise ValueError("--output is required")
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report["final_classification"])
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only hierarchical v1 production artifact health report.")
    parser.add_argument("--active-models", type=Path, default=DEFAULT_ACTIVE_MODELS_PATH)
    parser.add_argument("--models-jsonl", type=Path, default=DEFAULT_MODELS_JSONL_PATH)
    parser.add_argument("--production-dir", type=Path, default=DEFAULT_PRODUCTION_DIR)
    parser.add_argument("--production-manifest", type=Path, default=DEFAULT_PRODUCTION_MANIFEST_PATH)
    parser.add_argument("--dashboard", type=Path, default=DEFAULT_DASHBOARD_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-runtime-smoke", action="store_true")
    parser.add_argument("--protected-profile", action="append", default=[])
    parser.add_argument("--expected-active-models-sha256", default=DEFAULT_ACTIVE_MODELS_SHA256)
    parser.add_argument("--expected-models-jsonl-sha256", default=DEFAULT_MODELS_JSONL_SHA256)
    parser.add_argument("--expected-production-manifest-sha256", default=DEFAULT_PRODUCTION_MANIFEST_SHA256)
    parser.add_argument("--expected-production-file-sha256", action="append", default=[])
    parser.add_argument("--expected-production-dir-files", default=",".join(sorted(DEFAULT_PRODUCTION_DIR_FILES)))
    parser.add_argument("--expected-active-ppo-id", default=DEFAULT_EXPECTED_IDENTITY.active_ppo_id)
    parser.add_argument("--expected-active-dqn-id", default=DEFAULT_EXPECTED_IDENTITY.active_dqn_id)
    parser.add_argument("--expected-candidate-ppo-id", default=DEFAULT_EXPECTED_IDENTITY.candidate_ppo_id)
    parser.add_argument("--expected-candidate-dqn-id", default=DEFAULT_EXPECTED_IDENTITY.candidate_dqn_id)
    parser.add_argument("--expected-active-ppo-path", default=DEFAULT_EXPECTED_IDENTITY.active_ppo_path)
    parser.add_argument("--expected-active-dqn-path", default=DEFAULT_EXPECTED_IDENTITY.active_dqn_path)
    return parser


def _config_from_args(args: argparse.Namespace) -> HealthReportConfig:
    expected_hashes = dict(DEFAULT_PRODUCTION_FILE_SHA256)
    if args.expected_production_file_sha256:
        expected_hashes = {}
        for item in args.expected_production_file_sha256:
            filename, value = item.split("=", 1)
            expected_hashes[filename] = value.upper()

    expected_files = {item for item in args.expected_production_dir_files.split(",") if item}
    protected_profiles = dict(DEFAULT_PROTECTED_PROFILES)
    if args.protected_profile:
        protected_profiles = {}
        for item in args.protected_profile:
            label, path, file_count, total_bytes = item.split("|", 3)
            protected_profiles[label] = PathProfile(Path(path), int(file_count), int(total_bytes))

    expected_identity = ExpectedProductionIdentity(
        logical_model_id=DEFAULT_EXPECTED_IDENTITY.logical_model_id,
        active_ppo_id=args.expected_active_ppo_id,
        active_dqn_id=args.expected_active_dqn_id,
        candidate_ppo_id=args.expected_candidate_ppo_id,
        candidate_dqn_id=args.expected_candidate_dqn_id,
        active_ppo_path=args.expected_active_ppo_path,
        active_dqn_path=args.expected_active_dqn_path,
        dqn_architecture=DEFAULT_EXPECTED_IDENTITY.dqn_architecture,
        hierarchical_init_method=DEFAULT_EXPECTED_IDENTITY.hierarchical_init_method,
        contract=DEFAULT_EXPECTED_IDENTITY.contract,
        observation_dim=DEFAULT_EXPECTED_IDENTITY.observation_dim,
        action_count=DEFAULT_EXPECTED_IDENTITY.action_count,
        training_step=DEFAULT_EXPECTED_IDENTITY.training_step,
    )
    return HealthReportConfig(
        active_models_path=args.active_models,
        models_jsonl_path=args.models_jsonl,
        production_dir=args.production_dir,
        production_manifest_path=args.production_manifest,
        dashboard_path=args.dashboard,
        output_path=args.output,
        expected_identity=expected_identity,
        expected_active_models_sha256=args.expected_active_models_sha256.upper(),
        expected_models_jsonl_sha256=args.expected_models_jsonl_sha256.upper(),
        expected_production_manifest_sha256=args.expected_production_manifest_sha256.upper(),
        expected_production_file_sha256={key: value.upper() for key, value in expected_hashes.items()},
        expected_production_dir_files=expected_files,
        expected_protected_profiles=protected_profiles,
    )


def _validate_registry_row(
    row: dict[str, Any],
    *,
    algorithm: str,
    status: str,
    path: str,
    artifact_kind: str,
    agent_role: str,
    identity: ExpectedProductionIdentity,
) -> list[str]:
    problems: list[str] = []
    expected_row_fields = {"algorithm": algorithm, "status": status, "path": path}
    for field, expected_value in expected_row_fields.items():
        if row.get(field) != expected_value:
            problems.append(f"{field} expected {expected_value!r} got {row.get(field)!r}")

    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return problems + ["metadata must be a mapping"]

    expected_metadata: dict[str, Any] = {
        "logical_model_id": identity.logical_model_id,
        "dqn_architecture": identity.dqn_architecture,
        "hierarchical_init_method": identity.hierarchical_init_method,
        "training_step": identity.training_step,
        "exact_resume_capable": True,
        "eval_verdict": "PASS",
        "hard_blocker_status": "zero",
        "long_run_gate_verdict": "PASS",
        "production_ready": False,
        "baseline_update": False,
        "artifact_kind": artifact_kind,
        "agent_role": agent_role,
    }
    for field, expected_value in expected_metadata.items():
        if metadata.get(field) != expected_value:
            problems.append(f"metadata.{field} expected {expected_value!r} got {metadata.get(field)!r}")
    return problems


def _has_old_production_active_rows(rows: Iterable[dict[str, Any]], current_logical_model_id: str) -> bool:
    old_active_roles: set[str] = set()
    for row in rows:
        metadata = row.get("metadata", {}) if isinstance(row, dict) else {}
        if (
            row.get("status") == "active"
            and isinstance(metadata, dict)
            and metadata.get("logical_model_id") != current_logical_model_id
        ):
            if row.get("algorithm") == "ppo":
                old_active_roles.add("ppo")
            if row.get("algorithm") == "dqn":
                old_active_roles.add("dqn")
    return old_active_roles == {"ppo", "dqn"}


def _path_profile(path: Path) -> PathProfile:
    if not path.exists():
        raise FileNotFoundError(path)
    files = [item for item in path.rglob("*") if item.is_file()]
    return PathProfile(path=path, file_count=len(files), total_bytes=sum(item.stat().st_size for item in files))


def _collect_windows_process_entries() -> list[ProcessEntry]:
    command = (
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Depth 3"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    parsed = json.loads(result.stdout)
    if isinstance(parsed, dict):
        parsed = [parsed]
    entries: list[ProcessEntry] = []
    for item in parsed:
        command_line = item.get("CommandLine") or ""
        entries.append(
            ProcessEntry(
                pid=int(item.get("ProcessId") or 0),
                name=str(item.get("Name") or ""),
                command_line=str(command_line),
            )
        )
    return entries


def _collect_posix_process_entries() -> list[ProcessEntry]:
    result = subprocess.run(["ps", "-eo", "pid=,comm=,args="], check=False, capture_output=True, text=True)
    entries: list[ProcessEntry] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 2:
            continue
        pid = int(parts[0])
        name = parts[1]
        command_line = parts[2] if len(parts) > 2 else name
        entries.append(ProcessEntry(pid=pid, name=name, command_line=command_line))
    return entries


def _is_forbidden_process(entry: ProcessEntry) -> bool:
    command_line = entry.command_line.lower()
    return any(pattern in command_line for pattern in FORBIDDEN_PROCESS_PATTERNS)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_repo_root_on_sys_path() -> None:
    repo_root = _repo_root()
    if not any(Path(entry or ".").resolve() == repo_root for entry in sys.path):
        sys.path.insert(0, str(repo_root))


def _exact_resume_capable(checkpoint: dict[str, Any]) -> bool | None:
    if "exact_resume_capable" in checkpoint:
        return bool(checkpoint["exact_resume_capable"])
    resume_state = checkpoint.get("resume_state")
    if isinstance(resume_state, dict) and "exact_resume_capable" in resume_state:
        return bool(resume_state["exact_resume_capable"])
    return None


def _pass_fail(issues: Sequence[str]) -> str:
    return "FAIL" if issues else "PASS"


def _blocked_check(message: str, **extra: Any) -> dict[str, Any]:
    return {"status": "BLOCKED", "issues": [message], **extra}


def _skipped_runtime_smoke() -> dict[str, Any]:
    return {"status": "SKIPPED", "message": "runtime smoke skipped by request"}


def _check_status(report: dict[str, Any], status: str) -> bool:
    return any(isinstance(value, dict) and value.get("status") == status for value in report.values())


def _expect_field(issues: list[str], mapping: Any, field: str, expected: Any) -> None:
    actual = mapping.get(field) if isinstance(mapping, dict) else None
    if actual != expected:
        issues.append(f"{field} expected {expected!r} got {actual!r}")


def _model_filenames() -> tuple[str, str, str]:
    return EXPECTED_JOINT_FILENAME, EXPECTED_PPO_FILENAME, EXPECTED_DQN_FILENAME


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
