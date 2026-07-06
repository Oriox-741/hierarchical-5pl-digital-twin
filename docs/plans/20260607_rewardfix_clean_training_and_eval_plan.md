# Rewardfix Clean Training and Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare the premium reward guard telemetry reporting patch, then stage a clean v5 rewardfix training and evaluation path without touching registry, production, old checkpoints, DB rows, or invalid artifacts.

**Architecture:** Keep the immediate change small: surface telemetry that already exists in `src/act/env_5pl.py` through the evaluation accumulator, scenario summaries, CSV, and markdown report. Clean rewardfix training is a later cold-start run under the same v5 contract after explicit approval; registry review is blocked until training and offline scenario evaluation both pass.

**Tech Stack:** Python unittest, PyTorch checkpoint metadata, existing `src.eval.evaluate_real_world_scenarios` offline evaluator, existing `src.learn.train_joint_torch` trainer, JSONL/JSON/CSV/Markdown artifacts.

---

## Current State Summary

- Valid previous env id: `joint_curriculum_v5_clean_cold_start_1m_after_blockers`
- Valid previous checkpoint: `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_blockers/joint_torch_latest.pt`
- Current contract: `physical_reality_v5_route_candidate_visibility`
- Observation dim: `73`
- Discrete action count: `48`
- Recent completed work:
  - Offline evaluation readiness completed.
  - Metric-fix evaluator patch completed.
  - Premium hold/zero-work primary-fleet reward leak fixed in reward logic.
  - Global hard-blocker gating patch completed.
  - Rewardfix diagnostic offline scenario evaluation completed.
  - Diagnostic result: `OFFLINE_SCENARIO_EVALUATION_PASS_WITH_WATCHES_NOT_REGISTRY_READY`
- Aborted rewardfix clean-run artifacts exist for `joint_curriculum_v5_clean_cold_start_1m_after_rewardfix`; that identity is stale and must not be used for evaluation, resume, or registry.
- Interpretation: the diagnostic rerun cleared patched-code telemetry for the current evaluator/reward logic, but the checkpoint was trained before the reward exploit fix. It is not registry-ready.

## Invalid Artifacts To Avoid

- Do not use `joint_curriculum_v5_clean_cold_start_1m`.
- Do not use `models/checkpoints/joint_torch_v5_clean_cold_start_1m`.
- Do not use any v3/v4 checkpoint.
- Do not use any old invalid 700k artifact.
- Do not use `joint_curriculum_v5_clean_cold_start_1m_after_rewardfix`; it has partial aborted-run DB/checkpoint history.
- Do not use `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix`; it is a stale partial output dir and is not valid for eval/resume/registry.
- Do not resume from `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_blockers/joint_torch_latest.pt` for the rewardfix clean run. It is valid as a previous reference only, not as a warm-start source.

## Non-Negotiable Safety Constraints

- No registry update unless explicitly approved after Phase 6.
- No production overwrite.
- No DB mutation unless explicitly approved.
- No checkpoint or baseline mutation unless explicitly approved.
- No use of the invalid 700k run.
- No v3/v4 checkpoint.
- No resume from old contaminated checkpoints.
- Training commands require explicit manual approval.
- Full offline evaluation commands require explicit manual approval.
- Phase 1 may edit code/tests only when explicitly executing Phase 1. This plan file creation does not authorize those edits by itself.

## File Structure For Future Work

- Modify: `src/eval/scenario_metrics.py`
  - Responsibility: collect per-step premium guard telemetry from `info["reward_components"]`, expose it in `EpisodeMetrics.to_dict()`, and aggregate it in `summarize_episodes()`.
- Modify: `src/eval/real_world_scenario_arena.py`
  - Responsibility: include premium guard telemetry in `real_world_evaluation_report.md`; JSONL/JSON/CSV output should flow from `EpisodeMetrics.to_dict()` and scenario summary keys.
- Modify: `tests/eval/test_real_world_scenario_evaluator.py`
  - Responsibility: focused tests proving premium guard telemetry is collected, summarized, emitted to JSONL/JSON/CSV, and shown in the markdown report.
- Created for manual use after explicit approval: `configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean.json`
  - Responsibility: cold-start perf-clean rewardfix training config using the same v5 contract, observation dim 73, and action count 48.
- Later write manually after explicit approval: `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608`
  - Responsibility: new perf-clean rewardfix checkpoint directory. Codex should not launch this long run.
- Later write after explicit approval: `models/eval/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_offline_scenarios_20260608`
  - Responsibility: offline scenario evaluation artifacts for the perf-clean rewardfix checkpoint if evaluation is approved on 2026-06-08. Use the actual run date suffix if approval happens on a later date.

---

## Phase 1: Premium Reward Guard Telemetry Reporting Patch

### Expected Classification

`PREMIUM_GUARD_TELEMETRY_REPORTING_PATCH_READY`

### Telemetry Fields To Surface

These fields already exist in `src/act/env_5pl.py` reward components:

- `premium_sla_fleet_credit_blocked_no_current_work`
- `premium_primary_adaptation_credit_blocked_no_current_work`
- `premium_fleet_credit_allowed`

Aggregate each field as a count of steps where the reward component is positive. Because the reward code emits `0.0` or `1.0`, summing positive indicators produces an episode/scenario exposure count.

### Expected Output Changes

- `episode_metrics.jsonl`
  - Each episode row includes:
    - `premium_sla_fleet_credit_blocked_no_current_work`
    - `premium_primary_adaptation_credit_blocked_no_current_work`
    - `premium_fleet_credit_allowed`
- `scenario_summary.json`
  - Each scenario summary includes summed counts for the same three fields.
- `scenario_summary.csv`
  - The same three fields appear as columns through existing sorted summary-column export.
- `real_world_evaluation_report.md`
  - Add a `## Premium Reward Guard Telemetry` table with one row per scenario and the three fields.

### Task 1.1: Write Failing Accumulator Test

**Files:**
- Modify: `tests/eval/test_real_world_scenario_evaluator.py`

- [ ] **Step 1: Add this test to `ScenarioMetricAccumulatorTests` or the nearest existing accumulator test class**

```python
def test_premium_guard_telemetry_counters_increment(self) -> None:
    accumulator = EpisodeMetricAccumulator(
        scenario_id="premium_sla_pressure",
        seed=102,
        episode_index=0,
    )

    accumulator.observe(
        reward=0.0,
        info={
            "projected_action": {
                "discrete": {
                    "dispatch": "hold",
                    "route": "high_resilience",
                    "mode": "primary_fleet",
                    "reorder": "conservative",
                }
            },
            "reward_components": {
                "premium_sla_fleet_credit_blocked_no_current_work": 1.0,
                "premium_primary_adaptation_credit_blocked_no_current_work": 1.0,
                "premium_fleet_credit_allowed": 0.0,
            },
        },
    )
    accumulator.observe(
        reward=0.1,
        info={
            "projected_action": {
                "discrete": {
                    "dispatch": "dispatch",
                    "route": "shortest",
                    "mode": "primary_fleet",
                    "reorder": "conservative",
                }
            },
            "reward_components": {
                "premium_sla_fleet_credit_blocked_no_current_work": 0.0,
                "premium_primary_adaptation_credit_blocked_no_current_work": 0.0,
                "premium_fleet_credit_allowed": 1.0,
            },
        },
    )

    episode = accumulator.finish()

    self.assertEqual(episode.premium_sla_fleet_credit_blocked_no_current_work, 1)
    self.assertEqual(episode.premium_primary_adaptation_credit_blocked_no_current_work, 1)
    self.assertEqual(episode.premium_fleet_credit_allowed, 1)
    self.assertEqual(
        episode.to_dict()["premium_sla_fleet_credit_blocked_no_current_work"],
        1,
    )
    self.assertEqual(
        episode.to_dict()["premium_primary_adaptation_credit_blocked_no_current_work"],
        1,
    )
    self.assertEqual(episode.to_dict()["premium_fleet_credit_allowed"], 1)
```

- [ ] **Step 2: Run the focused test and verify it fails before implementation**

Run:

```powershell
python -m unittest tests.eval.test_real_world_scenario_evaluator -v
```

Expected before implementation:

```text
FAILED
AttributeError: 'EpisodeMetrics' object has no attribute 'premium_sla_fleet_credit_blocked_no_current_work'
```

### Task 1.2: Write Failing Output Wiring Test

**Files:**
- Modify: `tests/eval/test_real_world_scenario_evaluator.py`

- [ ] **Step 1: Add this test near `test_evaluation_outputs_include_delivered_and_hard_blocker_fields`**

```python
def test_evaluation_outputs_include_premium_guard_telemetry_fields(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "reports"
        scenario = ScenarioConfig(
            scenario_id="premium_sla_pressure",
            name="Premium SLA",
            description="Synthetic premium guard telemetry fixture.",
            seeds=[102],
            episode_count=1,
            environment_overrides={},
            expected_behavior="fixture",
            pass_fail_thresholds={},
        )
        episode = EpisodeMetricAccumulator(
            scenario_id="premium_sla_pressure",
            seed=102,
            episode_index=0,
        ).finish()
        episode.premium_sla_fleet_credit_blocked_no_current_work = 2
        episode.premium_primary_adaptation_credit_blocked_no_current_work = 3
        episode.premium_fleet_credit_allowed = 5

        write_evaluation_outputs(
            output_dir=output_dir,
            checkpoint_path=Path("candidate.pt"),
            scenarios=[scenario],
            episode_rows=[episode],
            override_capabilities={},
        )

        episode_json = json.loads((output_dir / "episode_metrics.jsonl").read_text(encoding="utf-8"))
        summary_json = json.loads((output_dir / "scenario_summary.json").read_text(encoding="utf-8"))
        csv_text = (output_dir / "scenario_summary.csv").read_text(encoding="utf-8")
        report_text = (output_dir / "real_world_evaluation_report.md").read_text(encoding="utf-8")
        scenario_summary = summary_json["scenarios"][0]

        self.assertEqual(episode_json["premium_sla_fleet_credit_blocked_no_current_work"], 2)
        self.assertEqual(episode_json["premium_primary_adaptation_credit_blocked_no_current_work"], 3)
        self.assertEqual(episode_json["premium_fleet_credit_allowed"], 5)
        self.assertEqual(scenario_summary["premium_sla_fleet_credit_blocked_no_current_work"], 2)
        self.assertEqual(scenario_summary["premium_primary_adaptation_credit_blocked_no_current_work"], 3)
        self.assertEqual(scenario_summary["premium_fleet_credit_allowed"], 5)
        self.assertIn("premium_sla_fleet_credit_blocked_no_current_work", csv_text)
        self.assertIn("premium_primary_adaptation_credit_blocked_no_current_work", csv_text)
        self.assertIn("premium_fleet_credit_allowed", csv_text)
        self.assertIn("Premium Reward Guard Telemetry", report_text)
        self.assertIn("premium_sla_pressure", report_text)
        self.assertIn("2", report_text)
        self.assertIn("3", report_text)
        self.assertIn("5", report_text)
```

- [ ] **Step 2: Run the focused test and verify it fails before implementation**

Run:

```powershell
python -m unittest tests.eval.test_real_world_scenario_evaluator -v
```

Expected before implementation:

```text
FAILED
AttributeError: 'EpisodeMetrics' object has no attribute 'premium_sla_fleet_credit_blocked_no_current_work'
```

### Task 1.3: Implement Metrics Aggregation

**Files:**
- Modify: `src/eval/scenario_metrics.py`

- [ ] **Step 1: Add fields to `EpisodeMetrics`**

Add after `emergency_zero_useful_positive_credit: int = 0`:

```python
    premium_sla_fleet_credit_blocked_no_current_work: int = 0
    premium_primary_adaptation_credit_blocked_no_current_work: int = 0
    premium_fleet_credit_allowed: int = 0
```

- [ ] **Step 2: Add fields to `EpisodeMetrics.to_dict()`**

Add after `"emergency_zero_useful_positive_credit": self.emergency_zero_useful_positive_credit,`:

```python
            "premium_sla_fleet_credit_blocked_no_current_work": (
                self.premium_sla_fleet_credit_blocked_no_current_work
            ),
            "premium_primary_adaptation_credit_blocked_no_current_work": (
                self.premium_primary_adaptation_credit_blocked_no_current_work
            ),
            "premium_fleet_credit_allowed": self.premium_fleet_credit_allowed,
```

- [ ] **Step 3: Add accumulator counters in `EpisodeMetricAccumulator.__init__()`**

Add after `self.emergency_zero_useful_positive_credit = 0`:

```python
        self.premium_sla_fleet_credit_blocked_no_current_work = 0
        self.premium_primary_adaptation_credit_blocked_no_current_work = 0
        self.premium_fleet_credit_allowed = 0
```

- [ ] **Step 4: Add positive-indicator counting in `EpisodeMetricAccumulator.observe()`**

Add after the emergency zero-useful positive-credit check:

```python
        if _component(info, "premium_sla_fleet_credit_blocked_no_current_work") > EPSILON:
            self.premium_sla_fleet_credit_blocked_no_current_work += 1
        if _component(info, "premium_primary_adaptation_credit_blocked_no_current_work") > EPSILON:
            self.premium_primary_adaptation_credit_blocked_no_current_work += 1
        if _component(info, "premium_fleet_credit_allowed") > EPSILON:
            self.premium_fleet_credit_allowed += 1
```

- [ ] **Step 5: Pass fields into `EpisodeMetrics(...)` in `finish()`**

Add after `emergency_zero_useful_positive_credit=self.emergency_zero_useful_positive_credit,`:

```python
            premium_sla_fleet_credit_blocked_no_current_work=(
                self.premium_sla_fleet_credit_blocked_no_current_work
            ),
            premium_primary_adaptation_credit_blocked_no_current_work=(
                self.premium_primary_adaptation_credit_blocked_no_current_work
            ),
            premium_fleet_credit_allowed=self.premium_fleet_credit_allowed,
```

- [ ] **Step 6: Aggregate fields in `summarize_episodes()`**

Add after `"emergency_zero_useful_positive_credit": sum(...),`:

```python
        "premium_sla_fleet_credit_blocked_no_current_work": sum(
            item.premium_sla_fleet_credit_blocked_no_current_work for item in episodes
        ),
        "premium_primary_adaptation_credit_blocked_no_current_work": sum(
            item.premium_primary_adaptation_credit_blocked_no_current_work for item in episodes
        ),
        "premium_fleet_credit_allowed": sum(item.premium_fleet_credit_allowed for item in episodes),
```

- [ ] **Step 7: Run the focused evaluator tests**

Run:

```powershell
python -m unittest tests.eval.test_real_world_scenario_evaluator -v
```

Expected after implementation:

```text
OK
```

### Task 1.4: Implement Markdown Report Table

**Files:**
- Modify: `src/eval/real_world_scenario_arena.py`

- [ ] **Step 1: Add a premium telemetry table after the hard-blocker telemetry table**

Add before `real_world_evaluation_report.md` is written:

```python
    premium_guard_fields = (
        "premium_sla_fleet_credit_blocked_no_current_work",
        "premium_primary_adaptation_credit_blocked_no_current_work",
        "premium_fleet_credit_allowed",
    )
    report_lines.extend(
        [
            "",
            "## Premium Reward Guard Telemetry",
            "",
            "| Scenario | "
            + " | ".join(premium_guard_fields)
            + " |",
            "|---|" + "|".join("---:" for _field in premium_guard_fields) + "|",
        ]
    )
    for row in scenario_summaries:
        report_lines.append(
            f"| {row['scenario_id']} | "
            + " | ".join(str(row.get(field, 0)) for field in premium_guard_fields)
            + " |"
        )
```

- [ ] **Step 2: Run output wiring test**

Run:

```powershell
python -m unittest tests.eval.test_real_world_scenario_evaluator -v
```

Expected:

```text
OK
```

### Task 1.5: Phase 1 Validation

**Files:**
- Modify: no additional files.

- [ ] **Step 1: Run focused eval tests**

```powershell
python -m unittest tests.eval.test_real_world_scenario_evaluator -v
```

Expected:

```text
OK
```

- [ ] **Step 2: Run related eval config/override tests**

```powershell
python -m unittest tests.eval.test_real_world_scenario_configs -v
python -m unittest tests.eval.test_scenario_overrides -v
```

Expected for each command:

```text
OK
```

- [ ] **Step 3: Compile changed Python files**

```powershell
python -m py_compile src\eval\scenario_metrics.py src\eval\real_world_scenario_arena.py tests\eval\test_real_world_scenario_evaluator.py
```

Expected:

```text
<no output, exit code 0>
```

- [ ] **Step 4: Stop condition**

Stop after tests pass. Do not run training. Do not run full evaluation. Do not create or modify training configs. Do not update registry. Report:

```text
PREMIUM_GUARD_TELEMETRY_REPORTING_PATCH_READY
```

---

## Phase 2: Perf-Clean Rewardfix Training Readiness

Phase 2 keeps the safety checklist for a new clean run trained under patched reward logic. The config has a fresh perf-clean identity because the prior `after_rewardfix` identity has partial aborted-run DB/checkpoint history.

### Proposed Identifiers And Paths

- Proposed env id: `joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean`
- Proposed output dir: `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608`
- Proposed config: `configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean.json`
- Proposed final PPO: `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608/ppo_torch_joint_final_v5_clean_cold_start_1m_after_rewardfix_perfclean.pt`
- Proposed final DQN: `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608/dqn_torch_joint_final_v5_clean_cold_start_1m_after_rewardfix_perfclean.pt`

### Required Config Properties

- `schema_version`: `joint_training_v1`
- `mdp_contract_version`: `physical_reality_v5_route_candidate_visibility`
- `shared_global_parameters.experiment_name`: `joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean`
- `shared_global_parameters.environment_id`: `joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean`
- `shared_global_parameters.team_id`: `joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean`
- `shared_global_parameters.observation_dim`: `73`
- `torch_joint_training.checkpoint_interval_steps`: keep existing known-good value from after-blockers config unless the user requests a diagnostic cadence change.
- `cold_start.required`: `true`
- `cold_start.allow_resume_contract`: `physical_reality_v5_route_candidate_visibility`
- `cold_start.reject_stale_contracts`: `true`
- No `resume` key anywhere in the config.
- No `active_checkpoint_dir` key.
- No `models/checkpoints` string anywhere in the config.
- Registry disabled at command time with `--skip-registry`.
- Trace logging disabled at command time with `--disable-trace-logging`.
- The stale `joint_curriculum_v5_clean_cold_start_1m_after_rewardfix` identity and `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix` directory are quarantined and not usable for eval/resume/registry.

### Performance Patch V4 Note

Manual V3 30k perf-clean probing showed replay collation is no longer the primary bottleneck. The remaining slowdown is dominated by recurring DQN compute windows, while CPU usage is under-parallelized: rollout/env stepping is mostly single-core, and DQN updates use only a few CPU core equivalents with the current Torch thread settings.

V4 keeps the canonical `joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean` learning schedule unchanged. It adds CLI-only Torch thread controls for manual sweeps and two explicitly experimental 30k cadence probes, `dqn1024` and `dqn512`, for speed/learning-quality comparison. Those cadence probes may change learning dynamics and must not replace the canonical 1M config or become registry candidates without a separate quality review.

Full CPU utilization would require larger architectural changes, such as vectorized or multiprocess rollout. That remains out of scope because it would touch curriculum scheduling, episode identity, replay collection, trace semantics, and deterministic auditability.

### Read-Only Readiness Checks

- [ ] **Step 1: Confirm no training or evaluation process is running**

```powershell
Get-CimInstance Win32_Process |
  Where-Object {
    $_.CommandLine -match 'train_joint_torch|evaluate_real_world_scenarios'
  } |
  Select-Object ProcessId,Name,CommandLine
```

Expected:

```text
<no train_joint_torch or evaluate_real_world_scenarios rows>
```

- [ ] **Step 2: Confirm target output dir does not already contain a run**

```powershell
Test-Path -LiteralPath 'models\checkpoints\joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608'
```

Expected before a new clean run:

```text
False
```

- [ ] **Step 3: Confirm proposed config exists and has the fresh perf-clean identity**

```powershell
Test-Path -LiteralPath 'configs\training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean.json'
```

Expected after approved config creation:

```text
True
```

- [ ] **Step 4: Confirm invalid artifacts still exist only as disallowed references and are not selected**

```powershell
Test-Path -LiteralPath 'models\checkpoints\joint_torch_v5_clean_cold_start_1m'
Test-Path -LiteralPath 'models\checkpoints\joint_torch_v5_clean_cold_start_1m_after_blockers\joint_torch_latest.pt'
Test-Path -LiteralPath 'models\checkpoints\joint_torch_v5_clean_cold_start_1m_after_rewardfix'
```

Expected:

```text
True
True
True
```

Interpretation: the old invalid v5 clean 1M dir may exist but must not be used. The after-blockers checkpoint may exist and is only a reference artifact, not a resume source. The stale rewardfix dir may exist as a quarantined partial artifact only.

### Manual Training Command For Later Approval

Do not run this in Codex unless the user explicitly approves training.

```powershell
python -m src.learn.train_joint_torch `
  --config configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean.json `
  --output-dir models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608 `
  --final-ppo-path models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608/ppo_torch_joint_final_v5_clean_cold_start_1m_after_rewardfix_perfclean.pt `
  --final-dqn-path models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608/dqn_torch_joint_final_v5_clean_cold_start_1m_after_rewardfix_perfclean.pt `
  --skip-registry `
  --disable-trace-logging
```

Required manual review before running:

- The command has no `--resume`.
- The command includes `--skip-registry`.
- The command includes `--disable-trace-logging`.
- The output dir is the dated perf-clean rewardfix dir, not the stale rewardfix dir or after-blockers dir.
- Final PPO/DQN paths are inside the dated perf-clean rewardfix output dir, not production or baseline folders.
- The config has no references to invalid checkpoints or active checkpoint dirs.
- The stale `joint_curriculum_v5_clean_cold_start_1m_after_rewardfix` identity is not used.

---

## Phase 3: Clean Rewardfix Training Run

Manual approval required. Codex must not execute this phase during planning, Phase 1, or Phase 2 readiness.

- [ ] **Step 1: Ask for explicit training approval**

Required user approval text:

```text
Approved: run perf-clean rewardfix 1M training from cold start with --skip-registry and --disable-trace-logging.
```

- [ ] **Step 2: Re-run Phase 2 safety checks immediately before launch**

Run every read-only check from Phase 2 again.

- [ ] **Step 3: Launch the approved command exactly once**

Use the Phase 2 manual command. Do not add `--resume`. Do not remove `--skip-registry`.

- [ ] **Step 4: Monitor without modifying registry or baselines**

Read-only monitoring commands:

```powershell
Get-Content -LiteralPath 'models\checkpoints\joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608\training_metrics.jsonl' -Tail 5
Get-ChildItem -LiteralPath 'models\checkpoints\joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608' | Select-Object Name,Length,LastWriteTime
```

- [ ] **Step 5: Stop condition**

Training is complete only when the trainer exits successfully and the expected latest/final artifacts exist in the rewardfix output dir. Do not update registry.

---

## Phase 4: Post-Training Audit Plan

Run only after Phase 3 is explicitly approved and completes.

### Artifacts To Inspect

- `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608/joint_torch_latest.pt`
- `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608/ppo_torch_joint_final_v5_clean_cold_start_1m_after_rewardfix_perfclean.pt`
- `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608/dqn_torch_joint_final_v5_clean_cold_start_1m_after_rewardfix_perfclean.pt`
- `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608/training_metrics.jsonl`

### Audit Checks

- Curriculum stages reached and final `global_step` align with the configured run length.
- `training_metrics.jsonl` final step matches checkpoint `global_step`.
- Checkpoint metadata has:
  - contract `physical_reality_v5_route_candidate_visibility`
  - observation dim `73`
  - discrete action count `48`
  - environment id `joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean`
- Global hard blockers remain clean in training metrics where available:
  - fake dispatch credit
  - customer revisit
  - route-failure zero-success positive dispatch/progress credit
  - NaN/Inf
  - dqn_local negative but dqn_train positive
  - no-current-work delivery credit
  - hold delivery leak
  - action8 route/delivery credit leak
  - unsafe 24/25 candidate credit
  - no-work positive dqn_local
  - emergency zero-useful positive credit
- Premium reward guard telemetry is present:
  - blocked no-current-work premium SLA fleet credit
  - blocked no-current-work premium primary adaptation credit
  - allowed premium fleet credit under justified dispatch work
- Route/fleet/reorder distributions are plausible:
  - route distribution does not collapse to one route family without scenario justification
  - premium SLA does not overcorrect into primary-fleet-only behavior without service/lateness need
  - emergency/aggressive reorder usage remains justified by pressure
- Registry remains untouched.

### Read-Only Metadata Check Command

```powershell
@'
from pathlib import Path
import torch

path = Path("models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608/joint_torch_latest.pt")
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
config = checkpoint.get("config", {})
shared = config.get("shared_global_parameters", {})
print({
    "global_step": checkpoint.get("global_step"),
    "contract": config.get("mdp_contract_version"),
    "observation_dim": checkpoint.get("observation_dim"),
    "discrete_action_count": checkpoint.get("discrete_action_count"),
    "environment_id": shared.get("environment_id"),
})
'@ | python -
```

Expected:

```text
{'global_step': 1000000, 'contract': 'physical_reality_v5_route_candidate_visibility', 'observation_dim': 73, 'discrete_action_count': 48, 'environment_id': 'joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean'}
```

---

## Phase 5: Offline Scenario Evaluation Plan For Clean Rewardfix Checkpoint

Run only after Phase 4 passes and the user explicitly approves offline scenario evaluation.

### Proposed Output Dir Naming

If evaluation is approved on 2026-06-08, use:

```text
models/eval/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_offline_scenarios_20260608
```

If approval happens later, use the same format with that run date:

```text
models/eval/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_offline_scenarios_YYYYMMDD
```

### Manual Evaluation Command For Later Approval

Do not run this in Codex unless the user explicitly approves offline evaluation.

```powershell
python -m src.eval.evaluate_real_world_scenarios `
  --checkpoint models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608/joint_torch_latest.pt `
  --scenario-dir configs/eval_scenarios `
  --output-dir models/eval/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_offline_scenarios_20260608 `
  --device cpu `
  --deterministic
```

### Expected Evaluation Outputs

- `episode_metrics.jsonl`
- `scenario_summary.json`
- `scenario_summary.csv`
- `real_world_evaluation_report.md`

### Post-Run Audit Metrics

- All 8 scenarios exist.
- 24 episode rows exist.
- Metadata:
  - checkpoint basename `joint_torch_latest.pt`
  - contract `physical_reality_v5_route_candidate_visibility`
  - observation dim `73`
  - action count `48`
- Scenario threshold verdicts.
- Hard-blocker verdicts and hard-blocker failures.
- `delivered_delta` remains an alias for `mean_step_delivered_delta`.
- `final_step_delivered_delta`, `episode_delivered_delta_sum`, `mean_step_delivered_delta`, and `total_delivered`.
- Service/lateness.
- Dispatch/hold ratio.
- Route distribution.
- Fleet distribution.
- Reorder distribution.
- Delivered counts and backlog behavior.
- Route failures.
- `no_vehicle_available`.
- `already_assigned_context_rows`.
- Premium reward guard telemetry:
  - `premium_sla_fleet_credit_blocked_no_current_work`
  - `premium_primary_adaptation_credit_blocked_no_current_work`
  - `premium_fleet_credit_allowed`
- Global fatal hard blockers:
  - `fake_dispatch_credit`
  - `customer_revisited`
  - `route_failure_positive_dispatch_credit`
  - `nan_inf_detected`
  - `dqn_local_negative_positive_train_rows`
  - `no_current_work_dqn_delivery_credit`
  - `hold_delivery_credit_leak`
  - `action8_route_or_delivery_credit_leak`
  - `unsafe_24_25_candidate_credit`
  - `no_work_positive_dqn_local`
  - `emergency_zero_useful_positive_credit`
- Watches:
  - mixed_stress service/lateness
  - shortest underuse in route/mixed stress
  - high_resilience/low_congestion dominance
  - premium SLA primary/secondary fleet balance
  - demand/backlog behavior
  - lead-time behavior
  - vehicle scarcity behavior
  - high holding cost behavior

---

## Phase 6: Registry Candidate Review

Blocked unless all conditions are true:

- Phase 3 perf-clean rewardfix training completes successfully.
- Phase 4 post-training audit passes.
- Phase 5 offline scenario evaluation passes.
- Global hard blockers are zero.
- Watches are acceptable or explicitly waived by the user.
- The user gives explicit registry approval.

Do not update registry during Phase 6 preparation. Registry candidate review is a decision gate, not an automatic mutation.

Possible final classifications for Phase 6 review:

- `REGISTRY_CANDIDATE_REVIEW_READY`
- `REGISTRY_CANDIDATE_REVIEW_BLOCKED_HARD_BLOCKER`
- `REGISTRY_CANDIDATE_REVIEW_BLOCKED_SCENARIO_FAILURE`
- `REGISTRY_CANDIDATE_REVIEW_BLOCKED_WATCHES`
- `REGISTRY_CANDIDATE_REVIEW_BLOCKED_NO_USER_APPROVAL`

---

## Safe /goal Execution Guidance

### Execute Phase 1 Only

```text
/goal Execute Phase 1 only from docs/plans/20260607_rewardfix_clean_training_and_eval_plan.md. Use superpowers:using-superpowers, superpowers:executing-plans, superpowers:test-driven-development, and superpowers:verification-before-completion. Allowed edits: src/eval/scenario_metrics.py, src/eval/real_world_scenario_arena.py, tests/eval/test_real_world_scenario_evaluator.py. Do not train. Do not run full evaluation. Do not create configs. Do not update registry. Do not mutate DB/checkpoints/baselines/production/model artifacts. Stop after PREMIUM_GUARD_TELEMETRY_REPORTING_PATCH_READY validation.
```

### Execute Read-Only Phase 2 Readiness Only

```text
/goal Execute read-only Phase 2 readiness only from docs/plans/20260607_rewardfix_clean_training_and_eval_plan.md. Use superpowers:using-superpowers, superpowers:systematic-debugging, and superpowers:verification-before-completion. Do not edit files. Do not train. Do not run evaluation. Do not resume. Do not update registry. Do not mutate DB/checkpoints/baselines/production/model artifacts. Report whether perf-clean rewardfix training readiness is blocked or ready for explicit manual approval.
```

No `/goal` in this plan runs training automatically.

---

## Recommended Immediate Next Goal

Manual Phase 3 is the next substantive action after the user reviews the perf-clean config and command.

Rationale: telemetry and reward fixes are already staged, but the prior `after_rewardfix` run was aborted and is stale. The next long run should be launched manually with the fresh perf-clean config, the dated output dir, `--skip-registry`, and `--disable-trace-logging`.

## Self-Review

- Spec coverage: This plan includes title/current state, safety constraints, stale rewardfix quarantine, Phase 1 telemetry patch, Phase 2 perf-clean readiness, Phase 3 approval-gated manual training, Phase 4 post-training audit, Phase 5 offline scenario evaluation, Phase 6 registry candidate gate, safe `/goal` commands, and recommended immediate next goal.
- Placeholder scan: Training and evaluation commands are concrete. The only date pattern is an output naming convention; the concrete 2026-06-08 path is provided for the current run date.
- Type consistency: The three premium guard telemetry fields are integer exposure counts in `EpisodeMetrics`, JSONL episode rows, scenario summaries, CSV columns, and markdown report rows.
