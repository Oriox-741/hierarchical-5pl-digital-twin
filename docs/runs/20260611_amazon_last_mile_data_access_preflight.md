# Amazon Last Mile Data Access Preflight

Date: 2026-06-11

Final classification: `AMAZON_LAST_MILE_AWS_CLI_MISSING`

## Scope

This preflight checked whether the Amazon Last Mile Routing Research Challenge
dataset can be accessed safely for route-choice proxy validation without
downloading the full dataset.

Hard constraints followed:

- no training
- no offline eval
- no registry mutation
- no production mutation
- no baseline mutation
- no DB mutation
- no checkpoint mutation
- no edits to existing eval outputs
- no full dataset download
- no large data directory creation

## Source Context Read

- `docs/runs/20260611_public_route_proxy_validation_readiness.md`
- `docs/plans/20260611_real_world_calibration_and_validation_plan.md`
- `docs/runs/20260611_hierarchical_v1_equal_budget_residual_watch_eval.md`
- `docs/00_PROJECT_DASHBOARD.md`

## AWS CLI Check

Command:

```powershell
Get-Command aws -ErrorAction SilentlyContinue | Select-Object Source,Version,CommandType | Format-List
```

Result:

- AWS CLI was not found.
- Because `aws` is missing, the metadata-only S3 listing was not run.
- No fallback download command was attempted.

Required listing command once AWS CLI is installed:

```powershell
aws s3 ls --summarize --human-readable --recursive --no-sign-request s3://amazon-last-mile-challenges/almrrc2021/
```

## Dataset Size And File Count

Not available from this machine during this preflight.

Reason:

- AWS CLI is missing.
- The task explicitly forbids full dataset download.
- No alternate command was used that would fetch more than metadata/listing.

## Likely Relevant Files

Expected files from the prior route-proxy readiness package and public schema
docs:

- `route_data.json`
- `actual_sequences.json`
- `package_data.json`
- `travel_times.json`
- `invalid_sequence_scores.json`
- `new_route_data.json`
- `new_actual_sequences.json`
- `new_package_data.json`
- `new_travel_times.json`

Route-choice proxy validation needs:

- `route_data.json` for `route_score`, station, departure, stop metadata, and
  zone information
- `actual_sequences.json` for actual driver stop order
- `package_data.json` for time-window and service-time stress proxies
- `travel_times.json` for directed travel-time matrix, fastest-route proxy, and
  travel-time asymmetry / low-congestion proxy

## Small Schema Or Sample Files

No small dataset files were identified locally because S3 listing could not be
queried.

Small public documentation that remains useful without dataset download:

- MIT-CAVE schema docs:
  `https://github.com/MIT-CAVE/rc-cli/blob/main/templates/data_structures.md`
- AWS Open Data Registry page:
  `https://registry.opendata.aws/amazon-last-mile-challenges/`
- Waterloo Amazon Challenge data page:
  `https://www.math.uwaterloo.ca/tsp/amz/data.html`

Those docs can confirm schema and access commands, but they are not a substitute
for a file-level S3 listing.

## Local Data Directory Check

Checked paths:

- `data/public/almrrc2021`: absent
- `almrrc2021`: absent
- `data`: absent
- `data/public`: absent

No Amazon Last Mile data directory was created.

## Recommended Manual Access Path

1. Install AWS CLI.
2. Run only the metadata/listing command:

   ```powershell
   aws s3 ls --summarize --human-readable --recursive --no-sign-request s3://amazon-last-mile-challenges/almrrc2021/
   ```

3. Review total size, file count, and per-file sizes.
4. If small schema/sample files are visible, approve selective download by exact
   key only.
5. If only large route JSON files are visible, choose a manual subset strategy
   before download:
   - download one small metadata/schema-like file if present;
   - or use the dataset's documented training/evaluation file names and request
     explicit approval for one exact file;
   - or download outside this workspace and copy in a small curated sample.

Do not run:

```powershell
aws s3 sync --no-sign-request s3://amazon-last-mile-challenges/almrrc2021/ data\public\almrrc2021\
```

without a separate approval after the metadata listing.

## Protected No-Mutation Proof

This task only wrote this run report and optionally the dashboard pointer.

No model, registry, production, baseline, DB, checkpoint, or existing eval-output
path was intentionally written.

Process scan found no matching AWS S3, training, offline-eval, long-run-gate, or
3M/5M/10M/100M process.

Known protected hashes are verified in the final response.

## Decision

`AMAZON_LAST_MILE_AWS_CLI_MISSING`

Rationale: dataset access is likely feasible through public S3, but this
workspace cannot perform the required metadata-only listing because AWS CLI is
not installed. The next safe action is installing AWS CLI and rerunning only the
listing command above.
