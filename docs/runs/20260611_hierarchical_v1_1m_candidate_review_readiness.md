# Hierarchical V1 1M Candidate Review Readiness

Date: 2026-06-11

## Result

Read-only candidate audit result: `CANDIDATE_NEEDS_RUNTIME_OR_REGISTRY_PATCH`.

Model artifact status: ready for human model review.

Registry dry-run status: blocked until registry tooling is generalized for hierarchical candidate metadata and this 1M logical model id.

No registry update, active model update, production mutation, baseline mutation, DB mutation, checkpoint mutation, training, offline eval, or promotion was performed.

## Candidate

- Candidate checkpoint:
  `models/checkpoints/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt`
- Eval directory:
  `models/eval/joint_torch_v5_prod_hierarchical_v1_1m_20260611_offline_scenarios`
- Suggested logical model id:
  `joint_torch_v5_prod_hierarchical_v1_1m_20260611`

## Checkpoint Metadata

The candidate checkpoint loaded read-only on CPU with:

- `checkpoint_version`: `torch_joint_policy_v1`
- `artifact_kind`: `joint_final`
- `global_step`: `1000000`
- `dqn_architecture`: `hierarchical_v1`
- `hierarchical_init_method`: `flat_teacher_distillation_v1`
- `contract`: `physical_reality_v5_route_candidate_visibility`
- `observation_dim`: `73`
- `continuous_action_dim`: `5`
- `discrete_action_count`: `48`
- optimizer state present: yes
- replay state present: yes
- RNG state present: yes
- `resume_state.exact_resume_capable`: yes
- replay size / cursor / total added: `500000` / `0` / `1000000`

Final split artifacts exist:

- `ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt`: present
- `dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt`: present

## Eval And Gate Evidence

Offline eval evidence:

- Scenarios: `8`
- Episode rows: `160`
- Expected episode rows from scenario summaries: `160`
- Scenario verdicts: `8/8 PASS`
- Hard-blocker verdicts: `8/8 PASS`
- Global fatal hard-blocker totals: zero

Long-run gate:

- Gate JSON: `docs/runs/20260611_hierarchical_dqn_1m_gate_result.json`
- Gate decision: `PASS`
- Fatal failures: none
- Warning count: `18`
- Gate exit code: `0` as recorded in `docs/runs/20260611_hierarchical_dqn_ladder_report.md`

Note: the gate JSON does not include an `exit_code` field; the exit-code evidence is the ladder report generated immediately after the gate command.

## Scenario Comparison

Values are `service / lateness / dispatch / success / no-current+no-unassigned`.

| Scenario | Production | Hierarchical 1M | Verdict |
|---|---:|---:|---|
| baseline_normal | 0.938 / 0.000 / 0.487 / 1.000 / 0 | 0.990 / 0.000 / 0.372 / 1.000 / 0 | PASS |
| demand_spike_volatility | 0.843 / 0.177 / 0.728 / 1.000 / 0 | 0.859 / 0.027 / 0.793 / 1.000 / 0 | PASS |
| high_holding_cost | 0.943 / 0.000 / 0.387 / 1.000 / 0 | 0.966 / 0.000 / 0.449 / 0.999 / 2 | PASS |
| lead_time_volatility | 0.936 / 0.000 / 0.455 / 1.000 / 0 | 1.000 / 0.000 / 0.360 / 0.999 / 1 | PASS |
| mixed_stress | 0.951 / 0.000 / 0.911 / 0.996 / 6 | 0.942 / 0.000 / 0.987 / 0.999 / 8 | PASS |
| premium_sla_pressure | 0.980 / 0.000 / 0.427 / 0.769 / 178 | 1.000 / 0.000 / 0.357 / 0.999 / 1 | PASS |
| route_disruption_congestion | 0.929 / 0.000 / 0.575 / 0.822 / 171 | 0.901 / 0.008 / 0.499 / 0.999 / 2 | PASS |
| vehicle_scarcity_capacity_shock | 0.914 / 0.000 / 0.731 / 1.000 / 0 | 0.949 / 0.001 / 0.806 / 1.000 / 0 | PASS |

## Residual Watches

These are review watches, not gate failures:

- `route_disruption_congestion` service remains below production: `0.901` vs `0.929`, with lateness `0.008` vs `0.000`.
- `mixed_stress` service remains below production: `0.942` vs `0.951`.
- Top-action concentration is higher than production in several scenarios.
- `mixed_success_route_failure_steps` increased in several scenarios, including demand spike `13` vs production `0`, mixed stress `10` vs `1`, and route disruption `6` vs `0`.
- Demand-spike service regressed from the 500k gate but remains above production and passed thresholds.

## Runtime Compatibility

Direct runtime loader:

- `load_torch_joint_policy` loaded the 1M hierarchical checkpoint read-only on CPU.
- Loaded policy architecture: `hierarchical_v1`.
- Deterministic zero-observation prediction returned:
  - continuous action length: `5`
  - finite continuous values: yes
  - continuous range: `[-0.221803, 0.994099]`
  - discrete action: `1`
  - discrete action legal: yes, in `0..47`

PolicyService compatibility:

- A temporary registry with active PPO/DQN records pointing to the hierarchical 1M artifacts resolved through `PolicyService.predict_joint(..., fallback_to_heuristic=False)`.
- The temp-registry smoke returned `algorithm=torch_joint`, continuous length `5`, finite continuous values, and discrete action `1`.
- This proves the runtime path can serve `hierarchical_v1` once registry rows point to the shared joint checkpoint.

## Registry Readiness Finding

Current real registry state:

- `active_models.json` remains the current production active mapping.
- `models.jsonl` contains `6` rows.
- Hierarchical registry rows: `0`.
- `active_models.json` SHA256: `E34DCCA1EA897CBAC4FA8DFD01066D11A0B2604E12FB5FC2CF37B2142975F6CD`
- `models.jsonl` SHA256: `AFB686085AC284748693232E189F472A0F0AD272508E45999C7ACB0C120D62EE`

Existing activation CLI:

- `python -m src.learn.register_torch_joint_candidate --help` supports `--dry-run`.
- It only promotes existing candidate rows by PPO/DQN candidate id.
- It is hardcoded for:
  - `EXPECTED_LOGICAL_MODEL_ID=joint_torch_v5_balanced_retention_ft_200k_20260608`
  - `EXPECTED_TRAINING_STEP=200000`
  - `EXPECTED_PARENT_STEP=1000000`
- It does not inspect or preserve `dqn_architecture`.
- It does not inspect or preserve `hierarchical_init_method`.
- It does not create hierarchical candidate rows.

Therefore no exact safe real dry-run command exists for the hierarchical 1M candidate yet. Running the current CLI against this candidate would require candidate ids that do not exist and would reject the intended logical model/step metadata even if such rows were manually added.

## Required Registry Patch Before Dry-Run

Implement and review one of these before any registry write:

1. Generalize `src.learn.register_torch_joint_candidate` so reviewed candidate metadata is data-driven instead of hardcoded to the old 200k production handoff.
2. Or add a separate hierarchical candidate registration/dry-run CLI.

Required behavior:

- Dry-run default must perform no writes.
- Candidate-row creation and active promotion must remain separate phases.
- Metadata must include and validate:
  - `logical_model_id=joint_torch_v5_prod_hierarchical_v1_1m_20260611`
  - `framework=torch_joint`
  - `runtime_loader=joint_torch`
  - `dqn_architecture=hierarchical_v1`
  - `hierarchical_init_method=flat_teacher_distillation_v1`
  - `contract=physical_reality_v5_route_candidate_visibility`
  - `obs_dim=73`
  - `action_count=48`
  - `training_step=1000000`
  - `joint_checkpoint=models/checkpoints/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt`
  - `eval_output_dir=models/eval/joint_torch_v5_prod_hierarchical_v1_1m_20260611_offline_scenarios`
  - `eval_verdict=PASS`
  - `hard_blocker_status=zero`
  - `production_promotion=false`
  - `baseline_update=false`
- Activation dry-run must print intended writes only to:
  - `models/registry/models.jsonl`
  - `models/registry/active_models.json`
- Runtime smoke after temp activation must prove `PolicyService` serves `algorithm=torch_joint` and does not use SB3 loaders.

Future dry-run shape after the required patch:

```powershell
python -m src.learn.register_torch_joint_candidate `
  --logical-model-id joint_torch_v5_prod_hierarchical_v1_1m_20260611 `
  --ppo-candidate-id <hierarchical-ppo-candidate-id> `
  --dqn-candidate-id <hierarchical-dqn-candidate-id> `
  --registry-dir models/registry `
  --dry-run
```

The placeholder ids are intentional: no hierarchical candidate registry rows exist yet.

## Verification Commands

- `python -m unittest tests.orchestration.test_torch_joint_runtime tests.orchestration.test_policy_service_torch_joint tests.learn.test_register_torch_joint_candidate -v`
  - Result: PASS, `39` tests.
- `python -m py_compile src\orchestration\torch_joint_runtime.py src\orchestration\policy_service.py src\learn\model_registry.py src\learn\register_torch_joint_candidate.py tests\orchestration\test_torch_joint_runtime.py tests\orchestration\test_policy_service_torch_joint.py tests\learn\test_register_torch_joint_candidate.py`
  - Result: PASS.

No training or offline eval command was run.

## Protected No-Mutation Proof

Protected hashes after the read-only audit:

- `models/registry/active_models.json`: `E34DCCA1EA897CBAC4FA8DFD01066D11A0B2604E12FB5FC2CF37B2142975F6CD`
- `models/registry/models.jsonl`: `AFB686085AC284748693232E189F472A0F0AD272508E45999C7ACB0C120D62EE`
- `models/production/joint_torch_v5_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`: `7A6E4EAAC7CB9CDBF5926A544CF1CAB7812940586C26E4C01CC757F3FD81B52E`

Protected root profiles:

- `models/production`: `4` files, `15040339` bytes.
- `models/baselines`: `2692` files, `7392576274` bytes.
- `db`: `7` files, `28330` bytes.

## Registration Plan Only

Do not register or promote now.

Recommended next step:

1. Patch registry tooling to support hierarchical candidate rows and dry-run activation review.
2. Add tests proving old production handoff safety remains intact.
3. Add tests proving hierarchical metadata is required and preserved.
4. Run dry-run only.
5. Request human review before any real registry write.

## Final Classification

`HIERARCHICAL_V1_1M_CANDIDATE_NEEDS_RUNTIME_OR_REGISTRY_PATCH`
