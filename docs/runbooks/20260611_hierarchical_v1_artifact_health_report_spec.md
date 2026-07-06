# Hierarchical V1 Artifact Health Report Spec

Date: 2026-06-11

Final classification: `STEP1_ARTIFACT_HEALTH_REPORT_SPEC_READY`

## Purpose

This spec defines a future read-only artifact-health report for the current
hierarchical v1 production state. It should answer one question: does the
registry, production manifest, production artifact set, runtime smoke evidence,
dashboard, and protected path state still match the approved production handoff?

This document is a specification only. It does not implement a CLI and does not
authorize training, offline eval, long-run gates, registry mutation, production
mutation, baseline mutation, DB mutation, checkpoint mutation, eval-output
edits, dataset downloads, or training configs.

## Current Expected Identity

- logical model id: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- active PPO id: `17ba1d28-0054-4f7c-ae9a-34cd305ebb89`
- active DQN id: `f87e10d6-479f-44fc-99d1-6925bc9cb346`
- candidate PPO id: `3befa11c-e853-4d0f-a951-c2fbbb2a9898`
- candidate DQN id: `71af6701-ac3a-40f2-bdd9-cc80868d5a1c`
- architecture: `hierarchical_v1`
- init method: `flat_teacher_distillation_v1`
- contract: `physical_reality_v5_route_candidate_visibility`
- obs/action: `73 / 48`
- training step: `1000000`

## Active Registry Checks

The report should read `models/registry/active_models.json` and verify:

- `ppo:continuous_control` points to
  `models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt`
- `dqn:tactical_dispatch` points to
  `models\checkpoints\joint_torch_v5_prod_hierarchical_v1_1m_20260611\dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt`
- no unexpected active role is introduced,
- hash equals
  `B7C6E79749044B92639B8A27EF38483022FC21D9F0EE08743AF88725E42AA60A`
  unless a later separately approved activation changes it.

## Candidate/Active Row Checks

The report should read `models/registry/models.jsonl` and verify:

- the two hierarchical candidate rows remain present,
- the two hierarchical active rows remain present,
- candidate rows remain `status=candidate`,
- active rows remain `status=active`,
- old production rows remain present for audit,
- active row metadata includes:
  - `logical_model_id=joint_torch_v5_prod_hierarchical_v1_1m_20260611`,
  - `dqn_architecture=hierarchical_v1`,
  - `hierarchical_init_method=flat_teacher_distillation_v1`,
  - `training_step=1000000`,
  - `exact_resume_capable=true`,
  - `eval_verdict=PASS`,
  - `hard_blocker_status=zero`,
  - `long_run_gate_verdict=PASS`,
  - `production_ready=false`,
  - `baseline_update=false`.

Expected `models.jsonl` hash:

`935D8BF34ADC8DD956A68FB57581EC860B02D74EC26434DD6C084E63055C38E1`

## Production Manifest Checks

The report should read:

`models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json`

Verify:

- logical model id matches,
- active/candidate ids match,
- source paths and production paths are present,
- source and production SHA256 hashes match for the three model artifacts,
- `dqn_architecture=hierarchical_v1`,
- `hierarchical_init_method=flat_teacher_distillation_v1`,
- `contract=physical_reality_v5_route_candidate_visibility`,
- `observation_dim=73`,
- `action_dim=48`,
- `training_step=1000000`,
- `eval_verdict=PASS`,
- `hard_blocker_status=zero`,
- `long_run_gate_verdict=PASS`,
- `residual_watches_accepted=true`,
- `baseline_update=false`,
- `db_mutation=false`,
- `registry_mutation=false`.

Expected production manifest hash:

`FDAF8DE495D90BDFD69BCC2EE3772EB6A09661E104A5127EA3CB01C6D02CF99A`

## Expected Production File Hashes

Production directory:

`models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611`

Expected files and hashes:

| File | SHA256 |
| --- | --- |
| `joint_torch_latest.pt` | `C1E756BDF7CD6B824CA134DBE6FB021612367AB18B12F2A24E13BF2167E51CDE` |
| `ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt` | `2D4CD9130492F7E15E9925177A9B82DF959ECF09534ADE649DC0D81D39410996` |
| `dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt` | `9DB010EB89ADB7262F3B85902507A13DB45AD7523940B65C3769CEFC96CA470D` |
| `production_manifest.json` | `FDAF8DE495D90BDFD69BCC2EE3772EB6A09661E104A5127EA3CB01C6D02CF99A` |

The production directory should contain exactly these four files.

## Runtime Smoke Checklist

Future artifact-health reports should include a read-only smoke check:

- load production `joint_torch_latest.pt` with `load_torch_joint_policy` on CPU,
- verify checkpoint metadata:
  - `checkpoint_version=torch_joint_policy_v1`,
  - `artifact_kind=joint_final`,
  - `dqn_architecture=hierarchical_v1`,
  - `hierarchical_init_method=flat_teacher_distillation_v1`,
  - `global_step=1000000`,
  - `exact_resume_capable=true`,
  - contract/obs/action match `physical_reality_v5_route_candidate_visibility`
    and `73 / 48`,
- deterministic zero-observation prediction:
  - continuous length `5`,
  - continuous finite,
  - continuous bounded in `[-1, 1]`,
  - discrete action in `0..47`,
- `PolicyService.predict_joint(..., fallback_to_heuristic=False)` returns
  `algorithm=torch_joint`,
- SB3 PPO and DQN loader calls remain `0`.

## Protected Path Profile Checks

Current expected path profiles:

| Path | File count | Total bytes |
| --- | ---: | ---: |
| `models/baselines` | `2692` | `7392576274` |
| `db` | `7` | `28330` |
| `models/checkpoints` | `787` | `4920830666` |
| `models/production` | `8` | `328265276` |
| `models/eval` | `128` | `388267659` |

Any drift requires explanation and a link to an explicit approved task.

## Process Scan

The report should scan for forbidden active operations:

- `train_joint`,
- `evaluate_real_world_scenarios`,
- `check_long_run_gate`,
- `offline_scenarios`,
- `aws s3 sync`,
- `aws s3 cp`.

Any active match is at least `ARTIFACT_HEALTH_FORBIDDEN_PROCESS_FOUND`.

## Dashboard Consistency Checks

The report should verify `docs/00_PROJECT_DASHBOARD.md` points to:

- current production checkpoint,
- final production release handoff,
- production monitoring runbook,
- equal-budget residual-watch eval,
- Step 1 monitoring and governance docs after Step 1 exists.

Dashboard status must not imply a new registry activation, production copy,
baseline update, DB write, checkpoint mutation, eval run, or training run unless
there is an explicit approved audit report for that operation.

## Failure Classifications

Future artifact-health report classifications:

- `ARTIFACT_HEALTH_CLEAN`
- `ARTIFACT_HEALTH_WARNINGS_ONLY`
- `ARTIFACT_HEALTH_ACTIVE_REGISTRY_MISMATCH`
- `ARTIFACT_HEALTH_MODELS_JSONL_MISMATCH`
- `ARTIFACT_HEALTH_PRODUCTION_MANIFEST_MISMATCH`
- `ARTIFACT_HEALTH_PRODUCTION_HASH_MISMATCH`
- `ARTIFACT_HEALTH_RUNTIME_SMOKE_FAILED`
- `ARTIFACT_HEALTH_PROTECTED_PATH_DRIFT`
- `ARTIFACT_HEALTH_FORBIDDEN_PROCESS_FOUND`
- `ARTIFACT_HEALTH_BLOCKED`

## Exact Future Read-Only Command Snippets

Protected hashes:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath @(
  'models\registry\active_models.json',
  'models\registry\models.jsonl',
  'models\production\joint_torch_v5_prod_hierarchical_v1_1m_20260611\joint_torch_latest.pt',
  'models\production\joint_torch_v5_prod_hierarchical_v1_1m_20260611\ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt',
  'models\production\joint_torch_v5_prod_hierarchical_v1_1m_20260611\dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt',
  'models\production\joint_torch_v5_prod_hierarchical_v1_1m_20260611\production_manifest.json'
)
```

Protected path profiles:

```powershell
$paths = @('models\baselines','db','models\checkpoints','models\production','models\eval')
foreach ($path in $paths) {
  $items = Get-ChildItem -LiteralPath $path -Recurse -File -Force
  $bytes = ($items | Measure-Object -Property Length -Sum).Sum
  "$path files=$($items.Count) bytes=$bytes"
}
```

Process scan:

```powershell
$self = $PID
$matches = Get-CimInstance Win32_Process | Where-Object {
  $_.ProcessId -ne $self -and $_.CommandLine -and ($_.CommandLine -match 'train_joint|evaluate_real_world_scenarios|check_long_run_gate|offline_scenarios|aws s3 sync|aws s3 cp')
}
if ($matches) { $matches | Select-Object ProcessId,Name,CommandLine } else { 'NO_TRAIN_EVAL_GATE_OR_AWS_DOWNLOAD_PROCESS' }
```

## Source Documents

- `docs/releases/20260611_hierarchical_v1_1m_production_handoff.md`
- `docs/runs/20260611_final_production_state_readonly_audit.md`
- `models/registry/active_models.json`
- `models/registry/models.jsonl`
- `models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/production_manifest.json`

## Final Classification

`STEP1_ARTIFACT_HEALTH_REPORT_SPEC_READY`
