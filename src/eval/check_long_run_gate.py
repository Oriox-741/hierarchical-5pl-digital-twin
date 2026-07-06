"""CLI wrapper for the read-only long-run evaluation gate checker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.eval.long_run_gate import evaluate_gate


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only long-run eval gate checker.")
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--production-summary", type=Path, required=True)
    parser.add_argument("--previous-summary", type=Path, default=None)
    parser.add_argument("--candidate-label", default="candidate")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate_gate(
        candidate_summary=args.candidate_summary,
        production_summary=args.production_summary,
        previous_summary=args.previous_summary,
    )
    report["candidate_label"] = str(args.candidate_label)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

