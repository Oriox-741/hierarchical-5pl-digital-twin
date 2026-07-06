# Hierarchical DQN Action Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat internal 48-logit DQN with a metadata-gated hierarchical/factorized DQN option that still emits one external action id in `[0, 47]`.

**Architecture:** Implement `hierarchical_v1` as an additive factorized Q composer with separate dispatch, route, fleet/mode, and reorder heads. Preserve the existing `flat_v1` policy as the default for legacy checkpoints; only explicitly tagged hierarchical checkpoints use the new architecture. For hold actions, route and fleet/mode heads are masked out of the composed external Q value so meaningless route/fleet labels cannot create hold-action pockets like action 16.

**Tech Stack:** Python 3.12, PyTorch, `unittest`, existing `JointPolicyBundle`, existing `DiscreteActionMapper`, existing exact-resume checkpoint payload path.

---

## Evidence And Audit Summary

Teacher retention failed before continuation:

- `docs/runs/20260611_level2_teacher_retention_ladder_report.md`
- 250k teacher-retention gate failed on `baseline_normal`, `high_holding_cost`, `lead_time_volatility`, and `route_disruption_congestion`.
- The failure shifted into action 16 concentration and reduced useful dispatch coverage, which supports an internal action-factorization fix rather than another flat-action reward patch.

Relevant code surfaces:

- `src/act/discrete_action_mapper.py`
  - External action contract is a fixed product: dispatch `2` x route `3` x fleet/mode `2` x reorder `4` = `48`.
  - `DiscreteActionMapper.map()` and `DiscreteActionMapper.encode()` must remain unchanged.
- `src/think/joint_policies.py`
  - Current DQN is `DiscreteQNetwork`, a flat `nn.Linear(..., 48)` head.
  - `JointPolicyBundle.select_dqn_action()` takes `argmax` over external Q values and returns one action id.
  - `build_checkpoint()` and `load_checkpoint_state()` own policy metadata and model state compatibility.
- `src/learn/train_joint_torch.py`
  - Trainer constructs `JointPolicyBundle()` and optimizes `policies.dqn_q_network.parameters()`.
  - `dqn_loss()` expects `policies.dqn_q_network(observations)` to return shape `(batch, 48)`.
  - Exact resume metadata and replay/RNG persistence are already implemented.
- `src/learn/joint_buffers.py`
  - Replay stores external action ids only and validates `[0, 47]`.
  - No replay change is required for a Q-network that still returns external Q values.
- `src/orchestration/torch_joint_runtime.py`
  - Runtime constructs `JointPolicyBundle` from checkpoint metadata and calls `select_dqn_action()`.
  - It must instantiate the same DQN architecture recorded in the checkpoint.
- `src/eval/real_world_scenario_arena.py`
  - Eval loader has the same checkpoint-to-`JointPolicyBundle` construction path as runtime.
- `src/act/env_5pl.py`
  - Env consumes only the external action id through `DiscreteActionMapper`; no env change is required.

Design comparison:

- A. Independent heads without Q composition is not enough because existing trainer/eval code expects a full external `(batch, 48)` Q tensor.
- B. Conditional action sampling is attractive, but it would require a new Bellman backup and replay loss contract.
- C. Additive factorized Q composition is the minimal viable patch because it keeps the existing DQN loss, replay format, runtime API, and evaluator API unchanged while changing the internal representation.

Chosen first implementation: `hierarchical_v1`, an additive factorized Q composer:

```text
Q_external(action) =
  Q_dispatch[dispatch]
  + Q_reorder[reorder]
  + I(dispatch == DISPATCH) * (Q_route[route] + Q_mode[mode])
```

This preserves all 48 external action ids. It also makes hold actions invariant to route and fleet/mode labels, which directly targets the observed action-16 hold label pocket.

## Checkpoint Compatibility Decision

Use explicit architecture metadata:

- `dqn_architecture`: `flat_v1` or `hierarchical_v1`
- `external_discrete_action_count`: `48`
- `internal_heads`:
  - for `flat_v1`: one external Q head of size `48`
  - for `hierarchical_v1`: dispatch `2`, route `3`, mode `2`, reorder `4`, composition `dispatch + reorder + dispatch_mask(route + mode)`

Compatibility rules:

- Legacy checkpoints missing `dqn_architecture` are treated as `flat_v1`.
- `flat_v1` checkpoints load only into a `flat_v1` `JointPolicyBundle`.
- `hierarchical_v1` checkpoints load only into a `hierarchical_v1` `JointPolicyBundle`.
- Flat-to-hierarchical conversion is not implicit and is not an exact resume.
- Any later production-parent run with `hierarchical_v1` needs an explicit partial-initialization or distillation plan. This task must not create that config.

## Implementation Tasks

### Task 1: Add Round-Trip And Head-Composition Tests

**Files:**

- Modify: `tests/act/test_discrete_action_mapper_contract.py`
- Create: `tests/think/test_hierarchical_dqn_policy.py`

- [ ] **Step 1: Add mapper round-trip test for all 48 actions**

Add this test to `tests/act/test_discrete_action_mapper_contract.py`:

```python
def test_all_discrete_actions_round_trip_through_structured_components(self) -> None:
    mapper = DiscreteActionMapper()

    for action_id in range(DISCRETE_ACTION_COUNT):
        with self.subTest(action_id=action_id):
            structured = mapper.map(action_id)
            self.assertEqual(action_id, mapper.encode(structured))
```

- [ ] **Step 2: Add hierarchical policy tests**

Create `tests/think/test_hierarchical_dqn_policy.py` with tests that assert:

- `HierarchicalDQNQNetwork(...)(observation)` returns `(batch, 48)`.
- composed head choices map to the exact same external ids as `DiscreteActionMapper.encode()`.
- deterministic `JointPolicyBundle(dqn_architecture="hierarchical_v1").select_dqn_action()` returns legal ids in `[0, 47]`.
- hold actions with identical dispatch/reorder but different route/mode have equal Q values.
- a dispatch-only auxiliary loss can update the dispatch head without route/mode/reorder head gradients.
- checkpoint metadata records `dqn_architecture`, `external_discrete_action_count`, and `internal_heads`.
- loading a flat checkpoint into a hierarchical bundle raises a clear architecture mismatch error.
- loading a hierarchical checkpoint into a flat bundle raises the same kind of error.

- [ ] **Step 3: Verify RED**

Run:

```powershell
python -m unittest tests.act.test_discrete_action_mapper_contract tests.think.test_hierarchical_dqn_policy -v
```

Expected before implementation: mapper round-trip passes; hierarchical tests fail because `HierarchicalDQNQNetwork` and `dqn_architecture` support are missing.

### Task 2: Implement Hierarchical Q Network And Metadata

**Files:**

- Modify: `src/think/joint_policies.py`

- [ ] **Step 1: Add architecture constants**

Add:

```python
FLAT_DQN_ARCHITECTURE: Final[str] = "flat_v1"
HIERARCHICAL_DQN_ARCHITECTURE: Final[str] = "hierarchical_v1"
SUPPORTED_DQN_ARCHITECTURES: Final[frozenset[str]] = frozenset({FLAT_DQN_ARCHITECTURE, HIERARCHICAL_DQN_ARCHITECTURE})
```

- [ ] **Step 2: Add `HierarchicalDQNQNetwork`**

Implement a PyTorch module that:

- validates observation dim and action count,
- builds a shared trunk,
- has `dispatch_head`, `route_head`, `mode_head`, and `reorder_head`,
- precomputes external action component index tensors using `DiscreteActionMapper`,
- returns composed external Q values with route/mode masked for hold actions,
- exposes `dispatch_q_values(observation)` for dispatch-only auxiliary losses.

- [ ] **Step 3: Add DQN factory**

Add `build_dqn_q_network(..., dqn_architecture=...)` returning either `DiscreteQNetwork` or `HierarchicalDQNQNetwork`.

- [ ] **Step 4: Wire `JointPolicyBundle`**

Modify `JointPolicyBundle.__init__()` to accept `dqn_architecture: str = FLAT_DQN_ARCHITECTURE`, store it, and build both online and target DQN networks through the factory.

- [ ] **Step 5: Add checkpoint metadata**

Modify `build_checkpoint()` to include:

```python
"dqn_architecture": self.dqn_architecture,
"external_discrete_action_count": self.discrete_action_count,
"internal_heads": self.dqn_q_network.architecture_metadata(),
```

- [ ] **Step 6: Enforce architecture compatibility on load**

Modify `load_checkpoint_state()` so:

- missing `dqn_architecture` means `flat_v1`,
- actual architecture must match `self.dqn_architecture`,
- `external_discrete_action_count` must match `self.discrete_action_count` when present,
- missing/invalid `model_state_dicts` still raises as before.

- [ ] **Step 7: Verify GREEN**

Run:

```powershell
python -m unittest tests.act.test_discrete_action_mapper_contract tests.think.test_hierarchical_dqn_policy -v
```

Expected: all tests pass.

### Task 3: Wire Runtime And Eval Loader To Checkpoint Architecture

**Files:**

- Modify: `src/orchestration/torch_joint_runtime.py`
- Modify: `src/eval/real_world_scenario_arena.py`
- Modify: `tests/orchestration/test_torch_joint_runtime.py`
- Modify: `tests/eval/test_real_world_scenario_evaluator.py`

- [ ] **Step 1: Add RED runtime/eval tests**

Add tests that build a hierarchical checkpoint and verify:

- `load_torch_joint_policy()` loads it into a hierarchical bundle.
- runtime prediction returns one legal external discrete action.
- `load_policy_checkpoint()` in the evaluator loads the hierarchical bundle and sets modules to eval mode.

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m unittest tests.orchestration.test_torch_joint_runtime tests.eval.test_real_world_scenario_evaluator -v
```

Expected before loader patch: hierarchical checkpoint load fails or is instantiated as flat.

- [ ] **Step 3: Read architecture from checkpoint**

In both loaders, pass:

```python
dqn_architecture=str(checkpoint.get("dqn_architecture", FLAT_DQN_ARCHITECTURE))
```

when constructing `JointPolicyBundle`.

- [ ] **Step 4: Verify GREEN**

Run:

```powershell
python -m unittest tests.orchestration.test_torch_joint_runtime tests.eval.test_real_world_scenario_evaluator -v
```

Expected: all tests pass.

### Task 4: Wire Trainer Construction And Exact-Resume Metadata

**Files:**

- Modify: `src/learn/train_joint_torch.py`
- Modify: `tests/learn/test_train_joint_curriculum.py`

- [ ] **Step 1: Add RED trainer tests**

Add tests that assert:

- a training config with `torch_training.dqn_architecture = "hierarchical_v1"` constructs a hierarchical `JointPolicyBundle`,
- checkpoint payloads include the DQN architecture metadata,
- exact-resume payloads still include replay/RNG state with hierarchical policy,
- loading a hierarchical checkpoint into a default flat bundle rejects with a DQN architecture mismatch,
- loading a flat legacy checkpoint into a hierarchical bundle rejects with the same mismatch.

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m unittest tests.learn.test_train_joint_curriculum -v
```

Expected before trainer patch: config architecture construction is missing or metadata is incomplete.

- [ ] **Step 3: Add trainer config helper**

Implement a small helper in `train_joint_torch.py`:

```python
def _dqn_architecture_from_config(config: Mapping[str, Any]) -> str:
    torch_cfg = config.get("torch_training", {})
    value = torch_cfg.get("dqn_architecture", FLAT_DQN_ARCHITECTURE) if isinstance(torch_cfg, Mapping) else FLAT_DQN_ARCHITECTURE
    if value not in SUPPORTED_DQN_ARCHITECTURES:
        raise ValueError(...)
    return str(value)
```

- [ ] **Step 4: Use helper in `main()`**

Construct:

```python
policies = JointPolicyBundle(dqn_architecture=_dqn_architecture_from_config(config)).to(device)
```

- [ ] **Step 5: Keep optimizer and loss code unchanged**

No trainer loss/replay changes should be made in this task because the hierarchical network still returns external `(batch, 48)` Q values.

- [ ] **Step 6: Verify GREEN**

Run:

```powershell
python -m unittest tests.learn.test_train_joint_curriculum -v
```

Expected: all tests pass.

### Task 5: Documentation And Final Verification

**Files:**

- Modify: `docs/00_PROJECT_DASHBOARD.md`
- Create: `docs/runs/20260611_hierarchical_dqn_action_architecture_patch_report.md`

- [ ] **Step 1: Update dashboard**

Record:

- teacher-retention failed at 250k,
- hierarchical architecture patch status,
- no training/eval/config was created,
- flat-to-hierarchical conversion remains blocked until a separate config/initialization decision.

- [ ] **Step 2: Write patch report**

Include:

- chosen architecture,
- files changed,
- checkpoint compatibility decision,
- runtime compatibility proof,
- tests run and results,
- protected no-mutation proof,
- whether config/training is ready or blocked.

- [ ] **Step 3: Run focused tests**

Run:

```powershell
python -m unittest tests.act.test_discrete_action_mapper_contract tests.think.test_hierarchical_dqn_policy tests.orchestration.test_torch_joint_runtime tests.eval.test_real_world_scenario_evaluator tests.learn.test_train_joint_curriculum -v
```

- [ ] **Step 4: Run compile check**

Run:

```powershell
python -m py_compile src\think\joint_policies.py src\orchestration\torch_joint_runtime.py src\eval\real_world_scenario_arena.py src\learn\train_joint_torch.py tests\act\test_discrete_action_mapper_contract.py tests\think\test_hierarchical_dqn_policy.py tests\orchestration\test_torch_joint_runtime.py tests\eval\test_real_world_scenario_evaluator.py tests\learn\test_train_joint_curriculum.py
```

- [ ] **Step 5: Verify protected paths**

Run:

```powershell
Get-FileHash models\registry\active_models.json -Algorithm SHA256
Get-FileHash models\registry\models.jsonl -Algorithm SHA256
Get-FileHash models\production\joint_torch_v5_balanced_retention_ft_200k_20260608\joint_torch_latest.pt -Algorithm SHA256
```

Expected hashes:

- `active_models.json`: `E34DCCA1EA897CBAC4FA8DFD01066D11A0B2604E12FB5FC2CF37B2142975F6CD`
- `models.jsonl`: `AFB686085AC284748693232E189F472A0F0AD272508E45999C7ACB0C120D62EE`
- production checkpoint: `7A6E4EAAC7CB9CDBF5926A544CF1CAB7812940586C26E4C01CC757F3FD81B52E`

## Stop/Proceed Criteria

Proceed to code only if the independent design reviewer agrees that `hierarchical_v1` is a safe minimal patch and that implicit flat-to-hierarchical checkpoint conversion must be rejected.

If the reviewer requests a broader conversion/distillation design before code, stop with classification `HIERARCHICAL_DQN_ARCHITECTURE_PLAN_READY`.

If the patch is implemented and independently approved, do not create a config in this task. The next task must decide how a hierarchical candidate is initialized from the production parent without treating flat DQN weights as an exact-resume-compatible parent.
