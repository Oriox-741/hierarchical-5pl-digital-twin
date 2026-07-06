# Continuous-Aware Rule Baseline Skeleton Report

Date: 2026-06-13

Final classification: CONTINUOUS_AWARE_BASELINE_SKELETON_READY.

## Scope

Implemented skeleton/library support for explicit continuous-aware rule baseline modes. This task did not run any benchmark, offline eval, or long-run gate, and did not mutate protected artifacts.

Allowed writes used:

- `src/eval/rule_based_baseline_arena.py`
- `tests/eval/test_rule_based_baseline_arena.py`
- `docs/runs/20260613_continuous_aware_baseline_skeleton_report.md`
- `docs/00_PROJECT_DASHBOARD.md`

## Architecture

The continuous-mode seam lives in `src/eval/rule_based_baseline_arena.py`.

Added:

- `ContinuousBaselineMode`
  - `neutral_continuous`
  - `heuristic_continuous`
  - `ppo_assisted_continuous`
  - `oracle_diagnostic_continuous`
- `continuous_mode_metadata(mode)`
- `build_continuous_action(...)`
- `normalized_action_from_physical_targets(...)`
- bounded vector validation for shape `(5,)`, finite values, and normalized `[-1, 1]`
- optional injected `continuous_provider` seam for PPO-assisted mode
- explicit oracle diagnostic opt-in guard

Default behavior remains backwards-compatible:

- `build_joint_action_from_rule_decision(decision)` still returns the existing normalized-zero continuous vector plus the validated rule discrete action id.

## Continuous Action Strategy

`neutral_continuous`:

- returns `np.zeros((5,), dtype=np.float32)`;
- preserves current benchmark behavior;
- now documents the source-truth caveat that normalized zero projects to midpoint physical controls, not physical no-op.

`heuristic_continuous`:

- uses current rule context only;
- no future leakage;
- derives inventory, lead-time, flow/SLA, route pressure, feasibility, vehicle scarcity, and capacity-shock signals from rule context;
- converts physical targets back to normalized `[-1, 1]` using `ActionProjector` defaults/config.

`ppo_assisted_continuous`:

- provider injection only;
- does not load production checkpoints;
- validates provider output strictly.

`oracle_diagnostic_continuous`:

- metadata marks `deployable=false`;
- raises unless `allow_oracle_diagnostic=True`;
- remains a diagnostic skeleton, not a deployable baseline.

## TDD Evidence

RED phase:

```text
python -m unittest tests.eval.test_rule_based_baseline_arena -v
```

Expected failure was observed before implementation:

- 18 tests ran.
- 10 errors.
- Failures were missing-symbol/missing-API errors for `ContinuousBaselineMode`, `build_continuous_action`, `continuous_mode_metadata`, and the new rollout parameter.
- Existing arena tests still passed.

GREEN phase first focused test:

```text
python -m unittest tests.eval.test_rule_based_baseline_arena -v
```

Result:

- 18 tests ran.
- OK.

Second RED phase for optional non-route continuous signals:

```text
python -m unittest tests.eval.test_rule_based_baseline_arena.RuleBasedBaselineArenaTests.test_heuristic_capacity_buffer_increases_under_optional_capacity_shock_signal tests.eval.test_rule_based_baseline_arena.RuleBasedBaselineArenaTests.test_heuristic_inventory_posture_responds_to_optional_lead_time_pressure -v
```

Expected failure was observed before the optional-signal patch:

- 2 tests ran.
- 2 failures.
- Capacity-shock pressure did not increase capacity buffer.
- Lead-time volatility pressure did not increase reorder/safety-stock posture.

Second GREEN focused test:

```text
python -m unittest tests.eval.test_rule_based_baseline_arena.RuleBasedBaselineArenaTests.test_heuristic_capacity_buffer_increases_under_optional_capacity_shock_signal tests.eval.test_rule_based_baseline_arena.RuleBasedBaselineArenaTests.test_heuristic_inventory_posture_responds_to_optional_lead_time_pressure -v
```

Result:

- 2 tests ran.
- OK.

Required combined rule-baseline test:

```text
python -m unittest tests.eval.test_rule_based_baselines tests.eval.test_rule_based_baseline_arena -v
```

Result:

- 31 tests ran.
- OK.

Required py_compile:

```text
python -m py_compile src/eval/rule_based_baselines.py src/eval/rule_based_baseline_arena.py tests/eval/test_rule_based_baselines.py tests/eval/test_rule_based_baseline_arena.py
```

Result:

- exit code 0.

## Requirement Coverage

- `neutral_continuous` returns existing zero vector: covered by test.
- Neutral projection midpoint physical controls documented and tested: covered.
- `heuristic_continuous` returns finite bounded 5-dim vector: covered.
- High stock pressure increases reorder/safety-stock posture: covered.
- Optional lead-time volatility pressure increases reorder/safety-stock posture: covered.
- High holding pressure with low inventory risk suppresses reorder posture: covered.
- Useful work/SLA pressure increases dispatch intensity and feasibility caps it: covered.
- Vehicle scarcity increases capacity buffer posture: covered.
- Optional capacity-shock pressure increases capacity buffer posture: covered.
- Route disruption/congestion changes continuous posture without changing discrete route/action id: covered.
- `ppo_assisted_continuous` uses injected provider and rejects bad output: covered.
- `oracle_diagnostic_continuous` non-deployable and explicitly gated: covered.
- Action 24 and action 32 remain legal external actions: existing rule-baseline tests still pass.
- Fake-env rollout feeds `EpisodeMetricAccumulator.observe` without output writes: covered.
- Protected output dir refusals remain intact: existing arena tests still pass.
- No `env_5pl` physics change: no edit made.
- No `DiscreteActionMapper` change: no edit made.

## Protected No-Mutation Proof

Pre-edit read-only snapshot:

- `models/registry/active_models.json`: `b7c6e79749044b92639b8a27ef38483022fc21d9f0ee08743af88725e42aa60a`
- `models/registry/models.jsonl`: `935d8bf34adc8dd956a68fb57581ec860b02d74ec26434dd6c084e63055c38e1`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json`: `fdaf8de495d90bdfd69bcc2ee3772eb6a09661e104a5127ea3cb01c6d02cf99a`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt`: `c1e756bdf7cd6b824ca134dbe6fb021612367ab18b12f2a24e13bf2167e51cde`
- `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`: `7a6e4eaac7cb9cdbf5926a544cf1cab7812940586c26e4c01cc757f3fd81b52e`
- `models/baselines` profile: `bc787edba875e123bfb3dc2deaa7c22d4789db9160e0818b22678bc635d3a375`
- `db` profile: `f6bd34478b1071a1296a1412314486a18ccacd390f73c2e0f37c826b06b2f277`
- `models/checkpoints` profile: `59320d2465d0faabfff77ff5592715a7a4bce68e1c8e5157519ee78b8c98af11`
- `models/eval` profile: `60d72954ad9dd8cf53b67aa94bda2d817d664ffa326eac91a6962c3b84b587cd`
- `models/production` profile: `998d09e1828a2e7417ce5d9b0491e10d1df5c01ed31ef9aa441fdf784dc77f19`

Post-implementation read-only checks:

- key protected file sizes and SHA256 hashes matched all file values above;
- protected directory file counts and byte totals matched the pre-edit snapshot:
  - `models/baselines`: 2692 files, 7392576274 bytes
  - `db`: 7 files, 28330 bytes
  - `models/checkpoints`: 787 files, 4920830666 bytes
  - `models/eval`: 128 files, 388267659 bytes
  - `models/production`: 8 files, 328265276 bytes
- directory profile strings were not used as final mutation proof because the original profile command depended on a PowerShell/.NET relative-path method that was unavailable in the final shell runtime; compatible reruns preserved counts/bytes but did not reproduce the stored profile hash strings.

Additional proof:

- `models/eval/not_allowed_for_rule_baseline` does not exist.
- No train/eval/gate/AWS process was present on final process recheck before writing this report.

## No-Run Statement

This task did not run:

- training;
- offline eval;
- `python -m src.eval.evaluate_real_world_scenarios`;
- long-run gate;
- `python -m src.eval.long_run_gate`;
- benchmark commands;
- dependency installation;
- dataset downloads.

## Independent Reviewer

Reviewer verdict: `CONTINUOUS_AWARE_BASELINE_SKELETON_APPROVED`.

Allowed verdicts:

- `CONTINUOUS_AWARE_BASELINE_SKELETON_APPROVED`
- `CONTINUOUS_AWARE_BASELINE_SKELETON_NEEDS_FIXES`
- `CONTINUOUS_AWARE_BASELINE_SKELETON_BLOCKED`

## Final Classification

CONTINUOUS_AWARE_BASELINE_SKELETON_READY
