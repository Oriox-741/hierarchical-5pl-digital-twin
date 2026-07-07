"""Validate and scaffold planned ablation studies without fabricating results."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "ablation"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "ablation"
REQUIRED_FIELDS = {
    "name",
    "type",
    "purpose",
    "required_artifacts",
    "expected_outputs",
    "claim_status",
    "safety_notes",
    "scenario_config_reference",
    "output_dir",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Single ablation config to validate or scaffold.")
    parser.add_argument("--all", action="store_true", help="Process every config under configs/ablation.")
    parser.add_argument("--dry-run", action="store_true", help="Validate configs and print planned steps only.")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/ablation"))
    parser.add_argument("--allow-missing-artifacts", action="store_true")
    parser.add_argument(
        "--allow-unsafe-diagnostic",
        action="store_true",
        default=False,
        help="Explicit guardrail flag for unsafe diagnostic modes. Normal use leaves safety enabled.",
    )
    args = parser.parse_args(argv)
    if not args.all and args.config is None:
        parser.error("provide --config or --all")
    if args.all and args.config is not None:
        parser.error("use either --config or --all, not both")
    return args


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    missing = sorted(REQUIRED_FIELDS - data.keys())
    if missing:
        raise ValueError(f"{path} missing required fields: {missing}")
    if data["type"] not in {"evaluation_time", "training_time_plan"}:
        raise ValueError(f"{path} has unsupported type {data['type']!r}")
    if data["claim_status"] not in {"planned", "completed"}:
        raise ValueError(f"{path} has unsupported claim_status {data['claim_status']!r}")
    if not str(data["output_dir"]).startswith("reports/ablation/"):
        raise ValueError(f"{path} output_dir must stay under reports/ablation/")
    return data


def _config_paths(args: argparse.Namespace) -> list[Path]:
    if args.all:
        return sorted(CONFIG_DIR.glob("*.json"))
    config = args.config
    if not config.is_absolute():
        config = REPO_ROOT / config
    return [config]


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _artifact_path(raw: str) -> Path | None:
    if raw.startswith("future "):
        return None
    if raw.startswith("private "):
        return None
    normalized = raw.replace("\\", "/")
    if "/" not in normalized and not Path(raw).suffix:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _missing_artifacts(config: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for artifact in config["required_artifacts"]:
        path = _artifact_path(str(artifact))
        if path is not None and not path.exists():
            missing.append(_repo_relative(path))
    return missing


def _discover_route_candidate_indices() -> dict[str, int] | None:
    try:
        source_path = REPO_ROOT / "src" / "act" / "observation_builder.py"
        module = ast.parse(source_path.read_text(encoding="utf-8"))
        values: dict[str, tuple[str, ...]] = {}

        def eval_node(node: ast.AST) -> tuple[str, ...]:
            if isinstance(node, ast.Tuple):
                items: list[str] = []
                for element in node.elts:
                    if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
                        raise ValueError("tuple contains non-string feature item")
                    items.append(element.value)
                return tuple(items)
            if isinstance(node, ast.Name):
                return values[node.id]
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                return eval_node(node.left) + eval_node(node.right)
            raise ValueError(f"unsupported feature expression: {ast.dump(node)}")

        for statement in module.body:
            target_name = None
            value = None
            if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                target_name = statement.target.id
                value = statement.value
            elif isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                target_name = statement.targets[0].id
                value = statement.value
            if target_name and value is not None:
                try:
                    values[target_name] = eval_node(value)
                except Exception:
                    continue

        observation_features = values["OBSERVATION_FEATURES"]
        route_features = values["ROUTE_CANDIDATE_OBSERVATION_FEATURES"]
        observation_feature_index = {name: index for index, name in enumerate(observation_features)}
    except Exception:
        return None
    return {name: observation_feature_index[name] for name in route_features}


def _resolve_output_dir(base_output_dir: Path, config: dict[str, Any]) -> Path:
    base = base_output_dir if base_output_dir.is_absolute() else REPO_ROOT / base_output_dir
    return base / str(config["name"])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_readme(path: Path, config: dict[str, Any], status: str, missing: list[str]) -> None:
    lines = [
        f"# {config['name']}",
        "",
        config["purpose"],
        "",
        f"- Status: `{status}`",
        f"- Claim status: `{config['claim_status']}`",
        "- Metrics generated: none",
    ]
    if missing:
        lines.append("- Missing private artifacts:")
        lines.extend(f"  - `{item}`" for item in missing)
    lines.append("")
    lines.append("This scaffold must not be cited as an ablation result.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_dry_run(config: dict[str, Any]) -> None:
    print(f"[DRY RUN] {config['name']}: {config['purpose']}")
    if config["name"] == "mask_route_candidates_eval":
        indices = _discover_route_candidate_indices()
        if indices is None:
            print("[DRY RUN] route-candidate feature indices: not discoverable from observation builder")
        else:
            print(f"[DRY RUN] route-candidate feature indices: {indices}")
    if config["name"] == "safety_projection_diagnostic":
        print("[DRY RUN] safety layer remains enabled by default")


def _run_config(config: dict[str, Any], args: argparse.Namespace) -> int:
    if args.dry_run:
        _print_dry_run(config)
        return 0

    missing = _missing_artifacts(config) if config["type"] == "evaluation_time" else []
    for artifact in missing:
        print(
            f"Missing private artifact: {artifact}. This is expected for sanitized public repo. No result generated.",
            file=sys.stderr,
        )
    if missing and not args.allow_missing_artifacts:
        return 2

    status = "planned_training_time_only"
    if config["type"] == "evaluation_time":
        status = "missing_artifacts" if missing else "scaffolded_not_executed"

    output_dir = _resolve_output_dir(args.output_dir, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    route_candidate_indices = _discover_route_candidate_indices() if config["name"] == "mask_route_candidates_eval" else None
    metadata = {
        "ablation_name": config["name"],
        "claim_status": config["claim_status"],
        "config_type": config["type"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "missing_artifacts": missing,
        "metrics_generated": False,
        "purpose": config["purpose"],
        "route_candidate_indices": route_candidate_indices,
        "safety_disabled": False,
        "unsafe_diagnostic_requested": bool(args.allow_unsafe_diagnostic),
        "status": status,
    }
    _write_json(output_dir / "run_metadata.json", metadata)
    _write_readme(output_dir / "README.md", config, status, missing)
    print(f"Wrote ablation scaffold metadata to {_repo_relative(output_dir)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = _config_paths(args)
    if not paths:
        print("No ablation config files found.", file=sys.stderr)
        return 2

    exit_code = 0
    for path in paths:
        try:
            config = _load_config(path)
        except Exception as exc:
            print(f"Config validation failed for {_repo_relative(path)}: {exc}", file=sys.stderr)
            exit_code = max(exit_code, 2)
            continue
        exit_code = max(exit_code, _run_config(config, args))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
