# Rule-Based Baseline Implementation Plan No Run

Date: 2026-06-13

Scope: implementation plan only. No code is implemented by this document. No benchmark, offline eval, long-run gate, training, dependency install, dataset download, registry mutation, production mutation, baseline mutation, DB mutation, checkpoint mutation, existing eval-output edit, training-config creation, private company-data ingest, or firm-facing data request is authorized.

Final planned classification: `RULE_BASED_BASELINE_SPEC_READY`

## Executive Summary

The safest later implementation path is a small pure-Python rule policy module that emits existing discrete action ids under the current `physical_reality_v5_route_candidate_visibility` contract. It should not depend on model checkpoints and should not change `env_5pl`, reward code, scenario files, registry, production artifacts, or eval outputs.

The first executable future goal should build a skeleton and tests only. It should not run scenario eval or benchmark comparisons.

## Proposed Future Files

Implementation skeleton, only after explicit approval:

- `src/eval/rule_based_baselines.py`
- `tests/eval/test_rule_based_baselines.py`
- `docs/runs/20260613_rule_based_baseline_skeleton_report.md`

Possible future benchmark runner, not part of the first skeleton goal:

- `scripts/rule_based_baseline_eval.py`
- `docs/runs/future_rule_based_baseline_report.md`

The first implementation should avoid a runner unless the user explicitly approves benchmark execution later.

## Recommended Implementation Style

Use deterministic pure policy objects/functions:

- no model checkpoint dependency;
- no PPO/DQN network loading;
- no training;
- no optimizer, replay, or RNG checkpoint state;
- no registry reads unless a future comparison report needs model identity;
- no protected writes;
- deterministic action id output for the same state fixture;
- explicit encode/decode helpers using the existing action semantics.

Recommended object shape:

```python
@dataclass(frozen=True)
class RuleBasedDecisionContext:
    dispatchable_orders: tuple[OrderView, ...]
    primary_vehicle_available: bool
    secondary_vehicle_available: bool
    route_disruption_pressure: float
    congestion_pressure: float
    stockout_risk: float
    holding_cost_pressure: float
    safety_stock_gap: float

class RuleBasedBaseline(Protocol):
    name: str
    def select_action(self, context: RuleBasedDecisionContext) -> int: ...
```

The implementation should keep context extraction separate from decision rules. Unit tests can then validate rule logic with synthetic fixtures without constructing `FivePLDigitalTwinEnv`.

## Integration Boundary

Preferred first skeleton:

- implement pure policies only;
- do not plug into `real_world_scenario_arena`;
- do not run scenarios;
- prove action ids and no-work holds with tests.

Later benchmark integration, if approved:

- add a policy adapter that can be used by `real_world_scenario_arena` or a separate approved evaluator;
- construct `RuleBasedDecisionContext` from `env_5pl` state/info;
- run in fresh output dirs only;
- compare against existing hierarchical production summaries only when the comparator path is read-only and approved.

Avoid modifying `real_world_scenario_arena` unless a small adapter hook is clearly cleaner than a separate approved evaluator.

## Rule Inventory For Skeleton

The skeleton should expose these named rules:

- `fifo_shortest_primary_none`
- `earliest_due_shortest_primary_none`
- `premium_first_shortest_primary_secondary_none`
- `low_congestion_under_disruption`
- `conservative_stock_threshold_reorder`
- `emergency_stockout_prevention_reorder`
- `vehicle_scarcity_primary_first_secondary_fallback`
- `high_holding_no_overstock`

Each rule should return one of the existing action ids, commonly:

- `0`: hold + shortest + secondary + none
- `24`: dispatch + shortest + secondary + none
- `28`: dispatch + shortest + primary + none
- `29`: dispatch + shortest + primary + conservative
- `31`: dispatch + shortest + primary + emergency
- `32`: dispatch + low_congestion + secondary + none
- `36`: dispatch + low_congestion + primary + none

Any additional ids used by a rule must be documented in code and tests.

## TDD Requirements

Write failing tests before implementation for:

- FIFO picks the oldest eligible order.
- Earliest due date picks the earliest promised/due order.
- Premium-first picks premium/urgent work before normal work.
- Low-congestion rule emits action `32` or `36` under route disruption/congestion pressure.
- Conservative stock-threshold reorder emits action `29` when inventory pressure is moderate.
- Emergency stockout prevention emits action `31` when stockout risk is severe.
- Vehicle scarcity rule uses primary action `28` when primary is available and secondary action `24` when only secondary is useful.
- High-holding no-overstock keeps reorder `none`.
- Invalid/no-work state returns canonical hold `0` rather than fake dispatch.
- Deterministic output for identical context.
- Action encode/decode table includes action `24` and action `32`.

Test fixtures must be synthetic and small. They must not read private data, checkpoint files, registry files, production files, or eval outputs.

## Future Verification Commands

Skeleton-only verification:

```powershell
python -m unittest tests.eval.test_rule_based_baselines -v
python -m py_compile src/eval/rule_based_baselines.py tests/eval/test_rule_based_baselines.py
```

Protected-state verification:

```powershell
Get-FileHash -Algorithm SHA256 models\registry\active_models.json
Get-FileHash -Algorithm SHA256 models\registry\models.jsonl
Get-FileHash -Algorithm SHA256 models\production\joint_torch_v5_prod_hierarchical_v1_1m_20260611\joint_torch_latest.pt
Get-FileHash -Algorithm SHA256 models\production\joint_torch_v5_prod_hierarchical_v1_1m_20260611\production_manifest.json
```

Process scan:

```powershell
Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -match '(train_joint|evaluate_real_world_scenarios|check_long_run_gate|aws\s+s3|run_benchmark|benchmark_eval|private_data|company_data_intake)'
}
```

Do not run scenario eval or gate in the skeleton goal.

## Future Benchmark Output Design

If benchmark execution is later approved, use a fresh output location such as:

- `reports/benchmark/rule_based_baselines/YYYYMMDD_rule_based_baseline_report.json`
- `reports/benchmark/rule_based_baselines/YYYYMMDD_rule_based_baseline_report.csv`
- `docs/runs/YYYYMMDD_rule_based_baseline_benchmark_report.md`

Do not write into existing `models/eval/**` unless a future request explicitly approves a fresh eval directory there.

Expected JSON shape:

```json
{
  "benchmark_metadata": {},
  "production_reference": {},
  "baselines": [],
  "scenario_metrics": [],
  "action_metrics": [],
  "comparison_to_hierarchical_v1": [],
  "decision": {
    "classification": "string",
    "reason": "string"
  }
}
```

Possible future classifications:

- `RULE_BASED_BASELINE_BENCHMARK_CLEAN`
- `RULE_BASED_BASELINE_BENCHMARK_WARNINGS_ONLY`
- `RULE_BASED_BASELINE_BENCHMARK_NEEDS_INVESTIGATION`
- `RULE_BASED_BASELINE_BENCHMARK_BLOCKED`

No benchmark classification should authorize training or protected mutation by itself.

## Non-Goals

- No implementation in this planning task.
- No simulator benchmark in the skeleton goal.
- No offline eval.
- No long-run gate.
- No model training.
- No dependency install.
- No dataset download.
- No registry, production, baseline, DB, checkpoint, or existing eval-output mutation.
- No private company data.
- No claims about real-world optimum.

## Approval Boundary

This document prepares a later implementation. It does not approve that implementation.

Exact future command:

```text
/goal Read docs/goals/20260612_execute_rule_based_baseline_skeleton_goal.txt and execute it exactly.
```

Before any future code:

- user must approve the skeleton goal;
- tests must be written first;
- independent read-only review is required after the patch;
- protected hashes and process scan must be reported;
- final output must state that no benchmark/eval was run.

Before any future benchmark:

- user must approve exact benchmark command, output dir, comparator, scenario count, seed, device, and mutation scope;
- output dirs must be fresh;
- protected hashes must be captured before and after;
- run report must include normalized denominators and no-mutation proof.
