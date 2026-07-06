# Public Replay Action Tendency Deep Dive Audit - 2026-06-14

## Classification

`PUBLIC_REPLAY_ACTION_TENDENCY_DEEP_DIVE_READY`

## Inputs Read

- `docs/reports/20260614_final_advisor_master_package_tr.md`
- `docs/reports/20260614_final_integrated_benchmark_scorecard_tr.md`
- `docs/runs/20260614_final_evidence_integration_sprint_audit.md`
- `reports/benchmarks/final_evidence_20260614/replay_large_samples/lade_large_replay_report.json`
- `reports/benchmarks/final_evidence_20260614/replay_large_samples/nyc_hvfhs_large_replay_report.json`
- `reports/benchmarks/final_evidence_20260614/replay_large_samples/olist_large_replay_report.json`
- `scripts/public_historical_replay_adapter.py`

## Why Fresh Diagnostic Was Needed

Existing large replay JSON reports have `rows=[]` and aggregate summaries only. Segmenting by dispatch/demand/fleet/lateness/congestion/disruption proxy required recomputing aggregate counters from the same public data mappings. This was a non-protected diagnostic, not training, offline eval, or a long-run gate.

## Outputs Written

- `docs/reports/20260614_public_replay_action_tendency_deep_dive_tr.md`
- `reports/benchmarks/final_evidence_20260614/replay_large_samples/public_replay_action_tendency_deep_dive.json`
- `docs/runs/20260614_public_replay_action_tendency_deep_dive_audit.md`

## Sample Sizes

- LaDe: `31415`
- NYC HVFHS: `100000`
- Olist: `96476`

## No-Overclaim

No training, offline eval, long-run gate, registry mutation, production mutation, baseline mutation, DB mutation, checkpoint mutation, existing eval overwrite, causal superiority claim, or company-data validation claim was performed.
