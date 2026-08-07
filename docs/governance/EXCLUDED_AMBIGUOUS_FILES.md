# Excluded and Ambiguous Files

This file records files skipped during sanitized repository preparation. It intentionally lists paths and reasons only, not secret values.

## Ambiguous / Reviewed Exclusions

- `docs/reports/20260612_full_repository_file_inventory.md`: local_inventory_excluded
- `docs/runs/20260611_hierarchical_dqn_1m_eval.log`: blocked_suffix_or_raw_body
- `docs/runs/20260611_hierarchical_dqn_1m_train.log`: blocked_suffix_or_raw_body
- `docs/runs/20260611_hierarchical_dqn_250k_eval.log`: blocked_suffix_or_raw_body
- `docs/runs/20260611_hierarchical_dqn_250k_train.log`: blocked_suffix_or_raw_body
- `docs/runs/20260611_hierarchical_dqn_500k_eval.log`: blocked_suffix_or_raw_body
- `docs/runs/20260611_hierarchical_dqn_500k_train.log`: blocked_suffix_or_raw_body
- `reports/demo_control_room_v8/smoke/server_stderr.log`: blocked_suffix_or_raw_body
- `reports/demo_control_room_v8/smoke/server_stdout.log`: blocked_suffix_or_raw_body

## Exclusion Counts by Reason

- `blocked_suffix_or_raw_body`: 8
- `cache_or_environment`: 54
- `heavy_raw_or_install_process_artifact_excluded`: 332
- `internal_agent_plan_history_excluded`: 11
- `local_inventory_excluded`: 1
- `not_summary_or_too_large`: 1
- `thesis_binary_or_zip_excluded`: 29

## Removed After Git Long-Path Validation

- $(chunks.FullName.Replace( + '\','').Replace('\','/')): removed from sanitized copy because Windows Git could not index deeply nested benchmark chunk paths; aggregate benchmark summaries remain included.
- $(chunks.FullName.Replace( + '\','').Replace('\','/')): removed from sanitized copy because Windows Git could not index deeply nested benchmark chunk paths; aggregate benchmark summaries remain included.
