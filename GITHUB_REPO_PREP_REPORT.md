# GitHub Repository Preparation Report

Generated UTC: `2026-07-06T20:22:24.262163+00:00`

## Target

- Local sanitized repository directory: `reports/github_publish/codex-proje-sanitized`
- Intended GitHub visibility: private first
- GitHub owner/repository placeholders in the request were not concrete.
- GitHub CLI status: unavailable (`gh` command not found).
- Remote push performed: `False`
- Push instructions: `PUSH_INSTRUCTIONS.md`

## Included / Excluded Counts

- Included files during initial copy: `1286`
- Excluded files during initial copy: `436`
- Ambiguous exclusions recorded: `9`
- Current sanitized file count before Git metadata: `668`
- Current sanitized total size: `43255053` bytes

## Largest Included Files

| Path | Size bytes |
| --- | ---: |
| `reports/demo_control_room_v8/traces/route_disruption_congestion_trace.json` | 3581939 |
| `reports/demo_control_room_v8/traces/mixed_stress_trace.json` | 3491655 |
| `reports/demo_control_room_v8/traces/baseline_normal_trace.json` | 3337741 |
| `reports/thesis_handoff/chapter3_method_evidence/models/eval/equal_budget_oldprod_20ep_seed42_20260611/scenario_summary.json` | 1902834 |
| `reports/demo_control_room_v6/traces/route_disruption_congestion_trace.json` | 1898036 |
| `reports/demo_control_room_v6/traces/mixed_stress_trace.json` | 1881220 |
| `reports/demo_control_room_v6/traces/baseline_normal_trace.json` | 1817544 |
| `reports/benchmarks/full_completion_20260613/rule_based_full_20ep/rule_based_full_report.json` | 1768690 |
| `reports/benchmarks/full_completion_20260613/amazon_full_route_proxy/amazon_full_route_proxy_summary.csv` | 1083988 |
| `reports/thesis_handoff/chapter3_method_evidence/verification/post_protected_fingerprint.json` | 977997 |
| `reports/thesis_handoff/chapter3_method_evidence/verification/pre_protected_fingerprint.json` | 977997 |
| `reports/benchmarks/full_completion_20260613/multiseed_robustness/seed_44/scenario_summary.json` | 759887 |
| `reports/thesis_handoff/chapter3_method_evidence/models/eval/joint_torch_v5_prod_hierarchical_v1_1m_20260611_offline_scenarios/scenario_summary.json` | 759322 |
| `reports/benchmarks/full_completion_20260613/multiseed_robustness/seed_42/scenario_summary.json` | 759321 |
| `reports/thesis_handoff/chapter3_method_evidence/models/eval/equal_budget_hierarchical_v1_20ep_seed42_20260611/scenario_summary.json` | 759321 |
| `reports/benchmarks/full_completion_20260613/multiseed_robustness/seed_43/scenario_summary.json` | 759241 |
| `reports/benchmarks/full_completion_20260613/multiseed_robustness/seed_46/scenario_summary.json` | 751637 |
| `reports/benchmarks/full_completion_20260613/multiseed_robustness/seed_45/scenario_summary.json` | 750682 |
| `reports/benchmarks/full_completion_20260613/multiseed_robustness/seed_44/scenario_summary.csv` | 601254 |
| `reports/benchmarks/full_completion_20260613/multiseed_robustness/seed_42/scenario_summary.csv` | 600822 |


## Tests and Checks Run

| Check | Result |
| --- | --- |
| `python -m py_compile` over `src`, `scripts`, `tests` | PASS; 143 Python files compiled |
| `python -m unittest discover -s tests` | SKIPPED; not run because broad discovery may trigger optional-dependency or artifact-dependent workflows |
| Secret/artifact scan | PASS; blocker count `0` |
| Files over 5 MB | `0` |
| Files over 50 MB | `0` |
| Checkpoint/model binary inclusion | PASS; none included |
| Raw data / DB inclusion | PASS; none included |
| DevSpace/tunnel config inclusion | PASS; none included |
| README local links | PASS; missing link count `0` |
| `.gitignore` required protected patterns | PASS; missing pattern count `0` |
| CI safety scan | PASS; forbidden term count `0` |
| Protected original artifact comparison | PASS; mismatch count `0` |
| Train/eval/gate process scan | PASS; no matching process found |

## Secret Scan Result

`SECRET_SCAN_REVIEWED.md` records harmless explanatory matches. No concrete token/key/private-key/risky-endpoint pattern remains. `SECRET_SCAN_BLOCKERS.md` is absent because there are no blockers.

## Protected Artifact Result

The sanitized repository does not include checkpoint binaries, production model directories, registry directories, baseline directories, DB files, raw public/private data bodies, existing eval output directories, DevSpace material, or agent-loop state. Original protected path counts/sizes/hashes were checked in `protected_no_mutation_check.json`.

## GitHub Publishing Result

Remote publishing was not performed because GitHub CLI is unavailable and the request provided placeholder owner/repository values. A local Git repository will be initialized and committed inside this sanitized directory only. Use `PUSH_INSTRUCTIONS.md` after selecting the private GitHub owner/repository.

Commit hash: recorded in final assistant response after local commit. Embedding the final commit hash inside this committed file would change the commit itself.

## Commands Used

- `git --version`
- `gh --version (failed: GitHub CLI not installed)`
- `python --version`
- `python -m py_compile <143 files under src/scripts/tests>`
- `custom secret/artifact scan over sanitized publish directory`
- `custom protected no-mutation comparison for registry/production/baselines/checkpoints/eval/db/data roots`
- `git init; git add .; git commit -m "Initial sanitized research artifact repository" (pending at report-write time)`

## Remaining Manual Decisions

- Choose final GitHub owner.
- Confirm final repository name if different from `codex-proje-sanitized`.
- Select a license before granting reuse rights; `LICENSE_PENDING.md` is included.
- Confirm whether any additional docs should be made public after private review.

## Final Local Decision

The sanitized local repository is ready for local Git initialization and private-first manual push instructions, subject to independent read-only review.

## Windows Git Long-Path Validation

Deep benchmark `chunks/` directories were removed from the sanitized copy after `git add` reported filename-too-long errors. Aggregate benchmark summaries remain included; raw/chunk-level bodies are intentionally excluded.
