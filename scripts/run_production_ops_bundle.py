from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any, Callable, Sequence


GENERATOR_VERSION = "production_ops_bundle_v1"

EXPECTED_LOGICAL_MODEL_ID = "joint_torch_v5_prod_hierarchical_v1_1m_20260611"
EXPECTED_RUNTIME_FAMILY = "torch_joint"
EXPECTED_DQN_ARCHITECTURE = "hierarchical_v1"
EXPECTED_HIERARCHICAL_INIT_METHOD = "flat_teacher_distillation_v1"

ARTIFACT_CLEAN = "ARTIFACT_HEALTH_CLEAN"
MONITORING_CLEAN = "MONITORING_REPORT_CLEAN"
MONITORING_WARNINGS = {
    "MONITORING_REPORT_WARNINGS_ONLY",
    "MONITORING_REPORT_NEEDS_INVESTIGATION",
}
MONITORING_RISK = {
    "MONITORING_REPORT_PRODUCTION_RISK_FOUND",
    "MONITORING_REPORT_PROTECTED_STATE_DRIFT",
    "MONITORING_REPORT_BLOCKED_BY_DATA_QUALITY",
}
COMPANY_READY = "COMPANY_DATA_SCHEMA_READY"
COMPANY_WARNINGS = {
    "COMPANY_DATA_SCHEMA_WARNINGS_ONLY",
    "COMPANY_DATA_SCHEMA_NEEDS_FIXES",
}
COMPANY_BLOCKED = {"COMPANY_DATA_SCHEMA_BLOCKED"}

OPS_CLEAN = "OPS_BUNDLE_CLEAN"
OPS_WARNINGS = "OPS_BUNDLE_WARNINGS_ONLY"
OPS_DATA_BLOCKED = "OPS_BUNDLE_DATA_SCHEMA_BLOCKED"
OPS_PRODUCTION_RISK = "OPS_BUNDLE_PRODUCTION_RISK_FOUND"
OPS_PROTECTED_DRIFT = "OPS_BUNDLE_PROTECTED_STATE_DRIFT"

WATCHED_ACTIONS = (24, 32)

PROTECTED_OUTPUT_DIRS = (
    Path("models/registry"),
    Path("models/production"),
    Path("models/baselines"),
    Path("models/checkpoints"),
    Path("models/eval"),
    Path("db"),
    Path("configs"),
    Path("data"),
)


@dataclass(frozen=True, slots=True)
class ProcessEntry:
    pid: int
    name: str
    command_line: str


CommandRunner = Callable[[list[str]], Any]


def build_ops_bundle_report(
    *,
    artifact_health: dict[str, Any],
    monitoring_report: dict[str, Any],
    company_data_schema: dict[str, Any],
    process_entries: Sequence[ProcessEntry] | None = None,
) -> dict[str, Any]:
    if process_entries is None:
        process_entries = collect_process_entries()

    artifact_classification = _artifact_classification(artifact_health)
    monitoring_classification = _decision_classification(monitoring_report)
    company_classification = _decision_classification(company_data_schema)
    process_scan = scan_processes(process_entries)
    decision = classify_bundle(
        artifact_classification=artifact_classification,
        monitoring_classification=monitoring_classification,
        company_classification=company_classification,
        process_scan=process_scan,
    )
    metadata = _monitoring_metadata(monitoring_report)

    return {
        "bundle_metadata": {
            "generator_version": GENERATOR_VERSION,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "logical_model_id": metadata.get("logical_model_id", EXPECTED_LOGICAL_MODEL_ID),
            "runtime_family": metadata.get("runtime_family", EXPECTED_RUNTIME_FAMILY),
            "dqn_architecture": metadata.get("dqn_architecture", EXPECTED_DQN_ARCHITECTURE),
            "hierarchical_init_method": metadata.get(
                "hierarchical_init_method",
                EXPECTED_HIERARCHICAL_INIT_METHOD,
            ),
            "synthetic_only": _all_synthetic(monitoring_report, company_data_schema),
            "private_data_ingested": False,
            "training_run": False,
            "offline_eval_run": False,
            "long_run_gate_run": False,
        },
        "artifact_health": _artifact_health_summary(artifact_health, artifact_classification),
        "monitoring_report": _monitoring_summary(monitoring_report, monitoring_classification),
        "company_data_schema": _company_schema_summary(company_data_schema, company_classification),
        "protected_state": _protected_state_summary(artifact_health, artifact_classification),
        "process_scan": process_scan,
        "decision": decision,
    }


def classify_bundle(
    *,
    artifact_classification: str | None,
    monitoring_classification: str | None,
    company_classification: str | None,
    process_scan: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if artifact_classification != ARTIFACT_CLEAN:
        reasons.append(f"artifact-health classification is {artifact_classification or 'missing/malformed'}")
    if _process_scan_has_forbidden_work(process_scan):
        reasons.append("process scan detected train/eval/gate/AWS/private-data process patterns")
    if monitoring_classification in MONITORING_RISK or monitoring_classification is None:
        reasons.append(f"monitoring classification is {monitoring_classification or 'missing/malformed'}")
    if company_classification in COMPANY_BLOCKED or company_classification is None:
        reasons.append(f"company schema classification is {company_classification or 'missing/malformed'}")
    if monitoring_classification in MONITORING_WARNINGS:
        reasons.append(f"monitoring classification is {monitoring_classification}")
    if company_classification in COMPANY_WARNINGS:
        reasons.append(f"company schema classification is {company_classification}")

    if artifact_classification != ARTIFACT_CLEAN or _process_scan_has_forbidden_work(process_scan):
        classification = OPS_PROTECTED_DRIFT
        recommended_action = "Investigate protected state/process drift before any production operation."
    elif monitoring_classification in MONITORING_RISK or monitoring_classification is None:
        classification = OPS_PRODUCTION_RISK
        recommended_action = "Open a production-risk investigation; do not train automatically."
    elif company_classification in COMPANY_BLOCKED or company_classification is None:
        classification = OPS_DATA_BLOCKED
        recommended_action = "Fix synthetic company-schema readiness before relying on the bundle."
    elif monitoring_classification in MONITORING_WARNINGS or company_classification in COMPANY_WARNINGS:
        classification = OPS_WARNINGS
        recommended_action = "Record warning-only bundle and continue monitoring; no model action authorized."
    else:
        classification = OPS_CLEAN
        recommended_action = "Record clean ops bundle; no model action authorized."

    return {
        "classification": classification,
        "reasons": reasons or ["artifact health, synthetic monitoring, and synthetic company schema are clean/ready"],
        "recommended_action": recommended_action,
    }


def run_bundle(
    output_path: Path | str,
    *,
    repo_root: Path | str | None = None,
    command_runner: CommandRunner | None = None,
    process_entries: Sequence[ProcessEntry] | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve() if repo_root is not None else _repo_root()
    output = _resolve_output_path(output_path, root)
    _ensure_safe_output_path(output, root)
    runner = command_runner or _default_command_runner

    with TemporaryDirectory(prefix="codex_step5_ops_bundle_") as temp_dir:
        temp_root = Path(temp_dir)
        artifact_output = temp_root / "artifact_health.json"
        monitoring_output = temp_root / "monitoring_report.json"
        company_output = temp_root / "company_schema_report.json"

        _run_checked(
            runner,
            [
                sys.executable,
                str(root / "scripts" / "production_artifact_health_report.py"),
                "--output",
                str(artifact_output),
            ],
        )
        _run_checked(
            runner,
            [
                sys.executable,
                str(root / "scripts" / "monitoring_report_generator.py"),
                "--artifact-health",
                str(artifact_output),
                "--use-built-in-synthetic-fixture",
                "clean",
                "--output",
                str(monitoring_output),
            ],
        )
        _run_checked(
            runner,
            [
                sys.executable,
                str(root / "scripts" / "company_data_intake_validator.py"),
                "--use-built-in-synthetic-fixture",
                "valid_minimal",
                "--output",
                str(company_output),
            ],
        )

        report = build_ops_bundle_report(
            artifact_health=_load_json_or_marker(artifact_output),
            monitoring_report=_load_json_or_marker(monitoring_output),
            company_data_schema=_load_json_or_marker(company_output),
            process_entries=process_entries,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def collect_process_entries() -> list[ProcessEntry]:
    _ensure_repo_root_on_path(_repo_root())
    from scripts import production_artifact_health_report as health

    return [
        ProcessEntry(pid=entry.pid, name=entry.name, command_line=entry.command_line)
        for entry in health.collect_process_entries()
    ]


def scan_processes(process_entries: Sequence[ProcessEntry]) -> dict[str, Any]:
    raw_matches: list[dict[str, Any]] = []
    flags = {
        "train_process_detected": False,
        "offline_eval_process_detected": False,
        "long_run_gate_process_detected": False,
        "aws_download_process_detected": False,
        "private_data_process_detected": False,
    }
    patterns = {
        "train_process_detected": ("train_joint", "train_joint_torch"),
        "offline_eval_process_detected": ("offline_scenarios", "evaluate_real_world_scenarios"),
        "long_run_gate_process_detected": ("long_run_gate", "check_long_run_gate"),
        "aws_download_process_detected": ("aws s3 cp", "aws s3 sync"),
        "private_data_process_detected": ("private", "private_extract", "company_extract"),
    }
    for entry in process_entries:
        command_line = entry.command_line.lower()
        matched_flags = [
            flag
            for flag, flag_patterns in patterns.items()
            if any(pattern in command_line for pattern in flag_patterns)
        ]
        if not matched_flags:
            continue
        for flag in matched_flags:
            flags[flag] = True
        raw_matches.append({**asdict(entry), "matched_flags": matched_flags})

    return {**flags, "raw_matches": raw_matches}


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    report = run_bundle(args.output)
    print(report["decision"]["classification"])
    return 0


def _artifact_health_summary(report: dict[str, Any], classification: str | None) -> dict[str, Any]:
    metadata = report.get("report_metadata") if isinstance(report.get("report_metadata"), dict) else {}
    return {
        "source_tool": "scripts/production_artifact_health_report.py",
        "classification": classification or "MISSING_OR_MALFORMED",
        "report_generated_at_utc": metadata.get("generated_at_utc"),
        "active_registry_status": _status(report, "active_registry_checks"),
        "production_manifest_status": _status(report, "production_manifest_checks"),
        "runtime_smoke_status": _status(report, "runtime_smoke_result"),
        "dashboard_consistency_status": _status(report, "dashboard_consistency_checks"),
    }


def _monitoring_summary(report: dict[str, Any], classification: str | None) -> dict[str, Any]:
    metadata = _monitoring_metadata(report)
    return {
        "source_tool": "scripts/monitoring_report_generator.py",
        "classification": classification or "MISSING_OR_MALFORMED",
        "synthetic_data": bool(metadata.get("synthetic_data")),
        "watched_actions": list(WATCHED_ACTIONS),
        "alert_count": len(report.get("alerts") or []),
        "residual_watch_count": len(report.get("residual_watches") or []),
    }


def _company_schema_summary(report: dict[str, Any], classification: str | None) -> dict[str, Any]:
    metadata = report.get("report_metadata") if isinstance(report.get("report_metadata"), dict) else {}
    return {
        "source_tool": "scripts/company_data_intake_validator.py",
        "classification": classification or "MISSING_OR_MALFORMED",
        "synthetic_data": bool(metadata.get("synthetic_data")),
        "table_count": int(metadata.get("table_count") or len(report.get("table_summaries") or [])),
        "data_quality_check_count": len(report.get("data_quality_checks") or []),
        "join_key_check_count": len(report.get("join_key_checks") or []),
        "warning_count": len(report.get("warnings") or []),
    }


def _protected_state_summary(report: dict[str, Any], classification: str | None) -> dict[str, Any]:
    clean = classification == ARTIFACT_CLEAN
    profile_status = _status(report, "protected_path_profile_checks")
    return {
        "registry_status": "clean" if _status(report, "active_registry_checks") == "PASS" else "drift",
        "production_status": "clean" if _production_status_clean(report) else "drift",
        "baseline_status": "clean" if clean and profile_status == "PASS" else "unknown_or_drift",
        "db_status": "clean" if clean and profile_status == "PASS" else "unknown_or_drift",
        "checkpoint_status": "clean" if clean and profile_status == "PASS" else "unknown_or_drift",
        "eval_output_status": "clean" if clean and profile_status == "PASS" else "unknown_or_drift",
        "artifact_health_report_status": "clean" if clean else "drift",
    }


def _artifact_classification(report: dict[str, Any]) -> str | None:
    value = report.get("final_classification")
    return value if isinstance(value, str) and value else None


def _decision_classification(report: dict[str, Any]) -> str | None:
    decision = report.get("decision")
    if not isinstance(decision, dict):
        return None
    value = decision.get("classification")
    return value if isinstance(value, str) and value else None


def _monitoring_metadata(report: dict[str, Any]) -> dict[str, Any]:
    metadata = report.get("report_metadata")
    return metadata if isinstance(metadata, dict) else {}


def _all_synthetic(monitoring_report: dict[str, Any], company_data_schema: dict[str, Any]) -> bool:
    monitoring_metadata = _monitoring_metadata(monitoring_report)
    company_metadata = company_data_schema.get("report_metadata")
    if not isinstance(company_metadata, dict):
        company_metadata = {}
    return bool(monitoring_metadata.get("synthetic_data")) and bool(company_metadata.get("synthetic_data"))


def _status(report: dict[str, Any], key: str) -> str | None:
    section = report.get(key)
    if not isinstance(section, dict):
        return None
    value = section.get("status")
    return value if isinstance(value, str) else None


def _production_status_clean(report: dict[str, Any]) -> bool:
    return all(
        status in {None, "PASS"}
        for status in (
            _status(report, "production_manifest_checks"),
            _status(report, "production_file_hash_checks"),
            _status(report, "runtime_smoke_result"),
        )
    )


def _process_scan_has_forbidden_work(process_scan: dict[str, Any]) -> bool:
    return any(
        bool(process_scan.get(key))
        for key in (
            "train_process_detected",
            "offline_eval_process_detected",
            "long_run_gate_process_detected",
            "aws_download_process_detected",
            "private_data_process_detected",
        )
    )


def _load_json_or_marker(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"_ops_bundle_load_error": f"{type(exc).__name__}: {exc}"}
    return data if isinstance(data, dict) else {"_ops_bundle_load_error": "JSON root is not an object"}


def _run_checked(command_runner: CommandRunner, command: list[str]) -> None:
    result = command_runner(command)
    returncode = getattr(result, "returncode", 0)
    if returncode != 0:
        stdout = getattr(result, "stdout", "")
        stderr = getattr(result, "stderr", "")
        raise RuntimeError(
            f"Subtool failed with exit code {returncode}: {' '.join(command)}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )


def _default_command_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def _resolve_output_path(output_path: Path | str, repo_root: Path) -> Path:
    output = Path(output_path)
    if not output.is_absolute():
        output = repo_root / output
    return output.resolve()


def _ensure_safe_output_path(output: Path, repo_root: Path) -> None:
    for protected in PROTECTED_OUTPUT_DIRS:
        protected_path = (repo_root / protected).resolve()
        if _is_relative_to(output, protected_path):
            raise ValueError(f"protected output path is not allowed: {output}")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_repo_root_on_path(repo_root: Path) -> None:
    repo_root_text = str(repo_root)
    for entry in sys.path:
        try:
            if Path(entry or ".").resolve() == repo_root:
                return
        except OSError:
            continue
    sys.path.insert(0, repo_root_text)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local-only hierarchical v1 production ops bundle.")
    parser.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
