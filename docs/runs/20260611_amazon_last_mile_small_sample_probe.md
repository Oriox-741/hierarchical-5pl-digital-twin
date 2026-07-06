# Amazon Last Mile Small-Sample Route Proxy Probe

Date: 2026-06-11

Final classification: `AMAZON_SMALL_SAMPLE_BLOCKED`

## Scope

This task attempted to start a small-sample Amazon Last Mile route-choice proxy
probe using only the explicitly approved small training sample files.

Hard constraints:

- do not download the full 3.1 GiB dataset
- do not download unapproved training/evaluation bulk files
- stop before download if expected download exceeds 10 MiB
- use a fresh `data/public/almrrc2021_small/` directory only
- do not train
- do not run project offline eval
- do not mutate registry, production, baselines, DB, checkpoints, or existing
  eval outputs

## Source Context Read

- `docs/runs/20260611_amazon_last_mile_data_access_preflight.md`
- `docs/runs/20260611_public_route_proxy_validation_readiness.md`
- `docs/plans/20260611_real_world_calibration_and_validation_plan.md`
- `docs/runs/20260611_hierarchical_v1_equal_budget_residual_watch_eval.md`
- `docs/00_PROJECT_DASHBOARD.md`
- `scripts/real_world_calibration/README.md`
- `scripts/real_world_calibration/amazon_last_mile_schema_probe.py`
- `scripts/real_world_calibration/route_proxy_metrics.py`

## Known S3 Listing From User

- Total objects: `42`
- Total size: `3.1 GiB`

Approved small files:

- `s3://amazon-last-mile-challenges/almrrc2021/almrrc2021-data-training/model_apply_inputs/new_route_data.json`
- `s3://amazon-last-mile-challenges/almrrc2021/almrrc2021-data-training/model_apply_inputs/new_package_data.json`
- `s3://amazon-last-mile-challenges/almrrc2021/almrrc2021-data-training/model_apply_inputs/new_travel_times.json`
- `s3://amazon-last-mile-challenges/almrrc2021/almrrc2021-data-training/model_score_inputs/new_actual_sequences.json`
- `s3://amazon-last-mile-challenges/almrrc2021/almrrc2021-data-training/model_score_inputs/new_invalid_sequence_scores.json`
- `s3://amazon-last-mile-challenges/almrrc2021/License.txt`
- `s3://amazon-last-mile-challenges/almrrc2021/Readme.txt`

## Pre-Download Checks

Target directory check:

- `data/public/almrrc2021_small`: absent

AWS CLI check:

```powershell
Get-Command aws -ErrorAction SilentlyContinue | Select-Object Source,Version,CommandType | Format-List
```

Result:

- AWS CLI was not found.
- The approved `aws s3 cp --no-sign-request` download path cannot be executed.
- No alternate downloader was used.
- No data directory was created.

## Download Decision

No files were downloaded.

Reason:

- The task requested selective download with `aws s3 cp --no-sign-request`.
- AWS CLI is missing in this workspace.
- Expected small-file download size could not be independently verified by local
  S3 metadata commands before download.
- Replacing `aws s3 cp` with another downloader would change the approved access
  path and could blur the no-full-download guarantee.

## Schema Probe Result

Not run against data.

Reason:

- The small sample files are not present locally.
- Running the probe against an absent data root would only confirm absence, not
  inspect Amazon schema.

## Route Proxy Metrics Result

Not run against data.

Reason:

- `new_actual_sequences.json` and `new_travel_times.json` are required for the
  current route proxy summary.
- The approved small files were not downloaded.

## Required Output Status

| Requirement | Status | Notes |
| --- | --- | --- |
| route count | blocked | sample files absent |
| stop/package field summary | blocked | `new_package_data.json` absent |
| time window availability | blocked | `new_package_data.json` absent |
| travel time matrix shape/sample | blocked | `new_travel_times.json` absent |
| actual sequence availability | blocked | `new_actual_sequences.json` absent |
| route score availability | blocked | `new_route_data.json` absent |
| invalid sequence score availability | blocked | `new_invalid_sequence_scores.json` absent |
| route proxy metrics | blocked | sample files absent |

## Route-Side Validation Answers

Can this public sample test shortest-like route behavior?

- Not yet in this workspace because the small sample is absent.
- Once downloaded, it should partially test shortest-like behavior through actual
  sequence cost versus travel-time-matrix alternatives. Literal geometric
  shortest route remains limited because Amazon locations are obfuscated.

Can this public sample test low-congestion/reliability proxy behavior?

- Not yet in this workspace because the small sample is absent.
- Once downloaded, it should partially test low-congestion/reliability proxies
  through directed travel-time asymmetry and actual route quality. It cannot
  provide true live congestion labels.

Can it test route concentration under stress buckets?

- Not yet in this workspace because the small sample is absent.
- Once downloaded, it should partially test route concentration by grouping
  route proxies under tight time-window and travel-time-asymmetry buckets.

What remains untestable without company data?

- `secondary_fleet` cost, availability, acceptance, and reliability
- primary versus secondary carrier economics
- reorder, stockout, backlog, and holding-cost correctness
- company-specific dispatch locks, no-vehicle semantics, already-assigned
  semantics, and SLA penalty costs

## Safe Resume Path

1. Install AWS CLI.
2. Re-run exact object metadata listing for the approved keys or a recursive
   listing filtered to those keys.
3. If the approved-file total is at or below 10 MiB, create:

   ```powershell
   New-Item -ItemType Directory -Force data\public\almrrc2021_small | Out-Null
   ```

4. Download only the approved files with exact `aws s3 cp --no-sign-request`
   commands.
5. Verify file sizes and SHA256 hashes.
6. Run:

   ```powershell
   python scripts\real_world_calibration\amazon_last_mile_schema_probe.py data\public\almrrc2021_small
   python scripts\real_world_calibration\route_proxy_metrics.py data\public\almrrc2021_small --route-limit 25
   ```

## Protected No-Mutation Proof

No data directory was created:

- `data/public/almrrc2021_small`: absent

No model, registry, production, baseline, DB, checkpoint, or existing eval-output
path was intentionally written.

No train, offline-eval, long-run-gate, AWS S3, 3M/5M/10M/100M process was
running at verification time.

Known protected hashes are verified in the final response.

## Reviewer Verdict

Independent read-only reviewer verdict:

`AMAZON_SMALL_SAMPLE_BLOCKED`

Reviewer scope covered the missing AWS CLI block, no-download/no-mutation
constraints, route-only framing, approved S3 key preservation, and safe resume
path.

## Decision

`AMAZON_SMALL_SAMPLE_BLOCKED`

Rationale: the approved small-sample route proxy probe is technically ready, but
this workspace still lacks AWS CLI, so the exact approved selective download path
cannot be executed and the sample schema cannot be inspected locally.
