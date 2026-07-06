# Company Data Validation Harness Report - 2026-06-14

## Classification

`COMPANY_REPLAY_SCHEMA_READY`

## Evidence

Command:

```powershell
python scripts\company_data_replay_validator.py --use-built-in-fixture valid_minimal --output reports\benchmarks\public_data_expansion_20260614\company_data_validation_harness\company_data_harness_report.json
```

Output:

`COMPANY_REPLAY_SCHEMA_READY`

## Synthetic Fixture Result

- Dispatch attempt rows: 2
- Input mode: `built_in_synthetic_fixture`
- Synthetic-only metadata: `true`
- Action 24 count: 1
- Action 32 count: 1
- Action 24 failures: 0
- Action 32 failures: 0
- Secondary cost premium mean: 14.0
- No-reorder stockout rate: 0.0
- Route failure rate: 0.0
- Actual/planned route time ratio mean: 1.04198

## Interpretation

The harness is ready as a schema/replay validator for future approved company data. It is not evidence that action 24/32 are economically optimal; it proves only that the required validation questions have an executable synthetic harness.

Caller-supplied `--fixture-root` inputs are not reported as synthetic by default. Report outputs also reject protected paths before any write attempt.
