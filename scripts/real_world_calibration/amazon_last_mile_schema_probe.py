"""Probe local Amazon Last Mile Routing Challenge files.

This script never downloads data. It checks a user-provided local data root for
the expected challenge JSON files and prints a compact schema-oriented summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_FILES: tuple[str, ...] = (
    "route_data.json",
    "actual_sequences.json",
    "package_data.json",
    "travel_times.json",
    "invalid_sequence_scores.json",
    "new_route_data.json",
    "new_actual_sequences.json",
    "new_package_data.json",
    "new_travel_times.json",
    "new_invalid_sequence_scores.json",
)


def _find_named_file(data_root: Path, filename: str) -> Path | None:
    direct = data_root / filename
    if direct.exists():
        return direct
    matches = list(data_root.rglob(filename))
    return matches[0] if matches else None


def _load_json_head(path: Path, *, max_routes: int) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        return {"_non_mapping_type": type(payload).__name__}
    return dict(list(payload.items())[:max_routes])


def _summarize_mapping_shape(payload: Mapping[str, Any]) -> dict[str, Any]:
    top_keys = list(payload.keys())
    first_value = payload[top_keys[0]] if top_keys else None
    summary: dict[str, Any] = {
        "sample_top_level_keys": top_keys[:3],
        "sample_top_level_count": len(top_keys),
        "first_value_type": type(first_value).__name__ if first_value is not None else None,
    }
    if isinstance(first_value, Mapping):
        summary["first_value_keys"] = list(first_value.keys())[:10]
        nested_key = next(iter(first_value), None)
        nested_value = first_value.get(nested_key) if nested_key is not None else None
        summary["first_nested_key"] = nested_key
        summary["first_nested_value_type"] = (
            type(nested_value).__name__ if nested_value is not None else None
        )
        if isinstance(nested_value, Mapping):
            summary["first_nested_value_keys"] = list(nested_value.keys())[:10]
    return summary


def probe_amazon_last_mile_schema(data_root: Path, *, max_routes: int = 3) -> dict[str, Any]:
    """Return a compact schema summary for local challenge files."""

    if not data_root.exists():
        return {
            "data_root": str(data_root),
            "exists": False,
            "files": {},
            "message": "data_root does not exist; no download attempted",
        }

    files: dict[str, Any] = {}
    for filename in EXPECTED_FILES:
        path = _find_named_file(data_root, filename)
        if path is None:
            files[filename] = {"present": False}
            continue
        try:
            head = _load_json_head(path, max_routes=max_routes)
            files[filename] = {
                "present": True,
                "path": str(path),
                "bytes": path.stat().st_size,
                "shape": _summarize_mapping_shape(head),
            }
        except (OSError, json.JSONDecodeError) as exc:
            files[filename] = {
                "present": True,
                "path": str(path),
                "error": f"{type(exc).__name__}: {exc}",
            }

    return {"data_root": str(data_root), "exists": True, "files": files}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe local Amazon Last Mile schema files without downloading data."
    )
    parser.add_argument("data_root", type=Path, help="Local Amazon Last Mile data root.")
    parser.add_argument("--max-routes", type=int, default=3, help="Routes to inspect per file.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = probe_amazon_last_mile_schema(args.data_root, max_routes=max(args.max_routes, 0))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
