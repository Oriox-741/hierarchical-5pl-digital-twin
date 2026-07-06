# Torch Joint Runtime Production Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe production runtime path for registered `torch_joint` candidates without making the current candidate active, promoting production, overwriting baselines, mutating DB rows, or changing checkpoints.

**Architecture:** Keep registry candidate review separate from runtime activation. Add an explicit Torch joint runtime loader that reuses the proven offline evaluation checkpoint-loading path, then teach `PolicyService.predict_joint()` to choose the Torch joint loader only when both active PPO and DQN records form one validated pair. Preserve the existing SB3 path for `.zip` active models and keep heuristic fallback only for missing active models.

**Tech Stack:** Python `unittest`, PyTorch, `src.think.joint_policies.JointPolicyBundle`, `src.learn.model_registry.ModelRegistry`, `src.orchestration.policy_service.PolicyService`, `src.act.observation_builder.OBSERVATION_DIM`, `src.act.discrete_action_mapper.DISCRETE_ACTION_COUNT`.

---

## Current Candidate Metadata

Logical model id: `joint_torch_v5_balanced_retention_ft_200k_20260608`

Candidate registry ids:
- PPO candidate id: `556b047a-7178-43bc-832c-0f9b49e6ef3c`
- DQN candidate id: `9fe92e2b-3ac2-46c3-9ade-3b6e3d8d5686`

Candidate env id: `joint_curriculum_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k`

Candidate joint checkpoint:
`models/checkpoints/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608/joint_torch_latest.pt`

Candidate split artifact paths currently recorded in `models/registry/models.jsonl`:
- PPO: `models/checkpoints/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608/ppo_torch_joint_final_balanced_retention_ft_200k.pt`
- DQN: `models/checkpoints/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608/dqn_torch_joint_final_balanced_retention_ft_200k.pt`

Metadata common to both candidate records:
- `status`: `candidate`
- `framework`: `torch_joint`
- `runtime_loader`: `joint_torch`
- `contract`: `physical_reality_v5_route_candidate_visibility`
- `obs_dim`: `73`
- `action_count`: `48`
- `training_step`: `200000`
- `parent_checkpoint`: `models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608/joint_torch_latest.pt`
- `parent_env_id`: `joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean`
- `parent_step`: `1000000`
- `eval_output_dir`: `models/eval/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios`
- `eval_verdict`: `PASS`
- `hard_blocker_status`: `zero`
- `registry_scope`: `candidate_only`
- `production_ready`: `false`
- `baseline_update`: `false`

Checkpoint metadata read from the candidate joint checkpoint:
- `checkpoint_version`: `torch_joint_policy_v1`
- `artifact_kind`: `joint_final`
- `global_step`: `200000`
- `observation_dim`: `73`
- `continuous_action_dim`: `5`
- `discrete_action_count`: `48`
- `model_state_dicts`: `ppo_feature_extractor`, `ppo_actor`, `ppo_critic`, `dqn_q_network`, `dqn_target_q_network`
- initialized from parent global step `1000000` under the same v5 contract

## Why This Candidate Is Not Active Or Production Yet

The candidate has passed offline evaluation, but runtime production readiness is incomplete:

1. `models/registry/active_models.json` is still `{}`.
2. `PolicyService` currently loads active PPO/DQN entries with SB3 `PPO.load()` and `DQN.load()` for `.zip` artifacts.
3. `PolicyService` now explicitly rejects `framework=torch_joint` / `runtime_loader=joint_torch` `.pt` artifacts from the SB3 path.
4. Orchestration runtime uses `PolicyService.predict_joint()` for joint control, but that method currently calls the single-policy SB3 path twice.
5. The working Torch joint inference path exists in `src/eval/real_world_scenario_arena.py`, not in production orchestration.

Therefore, making this candidate active before a Torch joint runtime loader exists would either fail safely or fall back to heuristics, depending on caller settings. It would not run the intended trained Torch joint model.

## Offline Evaluation Evidence Summary

Evaluation output:
`models/eval/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608_offline_scenarios`

All eight scenarios passed scenario thresholds and global hard blockers:
- `baseline_normal`: PASS, service `0.9382522683564719`, delivered_delta `0.5393518518518519`
- `demand_spike_volatility`: PASS, service `0.8431220059969657`, delivered_delta `1.2893518518518519`
- `high_holding_cost`: PASS, service `0.9426584871321113`, delivered_delta `0.4618055555555556`
- `lead_time_volatility`: PASS, service `0.9356873967985079`, delivered_delta `0.5092592592592592`
- `mixed_stress`: PASS, service `0.9511247910016719`, delivered_delta `1.4421296296296298`
- `premium_sla_pressure`: PASS, service `0.9802233129538735`, delivered_delta `0.5219907407407408`
- `route_disruption_congestion`: PASS, service `0.9291304983425458`, delivered_delta `0.5543981481481481`
- `vehicle_scarcity_capacity_shock`: PASS, service `0.9140800624047087`, delivered_delta `0.861111111111111`

Hard blockers were zero in the evaluation summary, including:
- fake dispatch credit
- customer revisit
- route-failure zero-success positive dispatch/progress credit
- NaN/Inf
- DQN negative/positive train row mismatch
- no-current-work delivery credit
- hold delivery leak
- action8 route/delivery credit leak
- unsafe 24/25 candidate credit
- no-work positive DQN local
- emergency zero-useful positive credit

Residual watches to keep as non-blocking production review notes:
- `route_disruption_congestion` has `high_resilience_selected_when_not_best_count=411`.
- `demand_spike_volatility` has `true_lateness_pressure=0.17686320800109925`.
- `mixed_success_route_failure_steps` appears in `baseline_normal`, `lead_time_volatility`, and `mixed_stress` with count `1` each.
- Baseline updates remain separate from runtime activation.

## Non-Goals

This plan must not:
- write `models/registry/active_models.json`
- append another registry row
- promote to production
- overwrite or update baselines
- mutate DB rows
- mutate checkpoints
- train
- run offline evaluation
- alter existing candidate artifacts

## Required Runtime Design

The safe runtime path needs:

## Production Requirement Decisions

| Requirement | Decision | Reason |
|---|---|---|
| Joint checkpoint loader | Required | The trained candidate is a synchronized PPO+DQN Torch joint checkpoint, and the offline evaluator already proves this path can run deterministic inference. |
| PPO/DQN split loader | Not required for production joint mode | The split PPO/DQN `.pt` registry paths identify the two roles, but their metadata points to one shared `joint_checkpoint`; loading the split files independently would duplicate the same payload and make pair validation harder. |
| Action mapper | No new mapper required | Runtime should return a continuous vector plus discrete integer; `FivePLDigitalTwinEnv.step()` already applies the action projector and discrete mapper. |
| Observation builder contract validation | Required | The model was trained under v5 route candidate visibility with observation dim 73; runtime must reject old observations or stale checkpoints. |
| Device handling | Required | Runtime must support CPU by default and explicit device selection without moving tensors across inconsistent devices. |
| Deterministic inference mode | Required | Production joint inference should use deterministic PPO action selection and DQN epsilon `0.0` unless a separate exploration mode is explicitly approved. |
| Fallback behavior | Required | Missing active mappings may fall back to heuristics, but incompatible active mappings must fail loudly rather than silently serving heuristics. |

1. **Joint checkpoint loader**
   - Load `joint_torch_latest.pt` with `torch.load(..., map_location=device, weights_only=False)`.
   - Require checkpoint mapping type.
   - Require `checkpoint_version == "torch_joint_policy_v1"`.
   - Require `artifact_kind in {"joint", "joint_final"}` for the production joint loader.
   - Require contract `physical_reality_v5_route_candidate_visibility`.
   - Require `observation_dim == 73`.
   - Require `continuous_action_dim == 5`.
   - Require `discrete_action_count == 48`.
   - Load weights into `JointPolicyBundle`.
   - Move to requested device and set `.eval()`.

2. **Registry active pair resolver**
   - Read explicit active PPO path for `continuous_control`.
   - Read explicit active DQN path for `tactical_dispatch`.
   - Find matching active registry records for both paths.
   - Require both records have `status="active"`.
   - Require both records have the same `logical_model_id`.
   - Require both records have the same `joint_checkpoint`.
   - Require `framework="torch_joint"` and `runtime_loader="joint_torch"` on both records.
   - Require artifact kinds `ppo_final` and `dqn_final`.
   - Require metadata contract, obs dim, and action count match runtime constants.
   - Return the shared `joint_checkpoint` as the actual runtime artifact.

3. **PolicyService changes**
   - Keep existing SB3 behavior for `.zip` active records.
   - Add a Torch joint cache keyed by `(joint_checkpoint, device)`.
   - Add `load_active_joint()` for paired active Torch joint records.
   - Update `predict_joint()` to:
     - try Torch joint paired-active loading first,
     - return trained Torch continuous and discrete actions in one `PolicyInference`,
     - preserve heuristic fallback when active records are missing and `fallback_to_heuristic=True`,
     - raise clear errors when active records exist but are incompatible.
   - Keep `predict("ppo")` and `predict("dqn")` SB3-only until standalone Torch single-agent runtime is explicitly needed.

4. **Action mapper / projector requirements**
   - No production runtime should manually decode actions before `env.step()`.
   - `PolicyService.predict_joint()` should return:
     - `"continuous"`: 1-D `np.float32` vector of length `5`, bounded by tanh in `[-1, 1]`.
     - `"discrete"`: integer in `[0, 47]`.
   - `FivePLDigitalTwinEnv.step()` remains responsible for projection and discrete mapping through existing action plumbing.

5. **Observation builder contract validation**
   - Before inference, reject observations whose final dimension is not `OBSERVATION_DIM == 73`.
   - The loader must reject checkpoints whose metadata or config declares a different observation dimension or contract.

6. **Device handling**
   - Default production device should be CPU unless the runtime config explicitly chooses otherwise.
   - The loader must pass `map_location=device`, call `.to(device)`, and create observation tensors on the same device as the policy.
   - Device should be included in the cache key.

7. **Deterministic inference**
   - PPO: call `select_ppo_action(observation_tensor, deterministic=True)` by default.
   - DQN: call `select_dqn_action(observation_tensor, epsilon=0.0)`.
   - Honor `deterministic=False` only if orchestration explicitly asks for stochastic PPO behavior; DQN should still use `epsilon=0.0` in production unless a separate exploration mode is approved.

8. **Fallback behavior**
   - Missing active records may fall back to heuristics if `fallback_to_heuristic=True`.
   - Incompatible active records must raise a clear error; do not silently fall back if the registry says a production model exists but cannot be loaded.

## Implementation Tasks

### Task 1: Add Torch Joint Runtime Loader

**Files:**
- Create: `src/orchestration/torch_joint_runtime.py`
- Create: `tests/orchestration/__init__.py`
- Create: `tests/orchestration/test_torch_joint_runtime.py`

- [ ] **Step 1: Write loader tests**

Create `tests/orchestration/test_torch_joint_runtime.py` with tests covering metadata validation, deterministic inference, device handling, and observation-shape rejection.

```python
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from src.act.action_projector import CONTINUOUS_ACTION_DIM
from src.act.discrete_action_mapper import DISCRETE_ACTION_COUNT
from src.act.observation_builder import OBSERVATION_DIM
from src.learn.train_joint_torch import MDP_CONTRACT_VERSION
from src.orchestration.torch_joint_runtime import TorchJointRuntimePolicy, load_torch_joint_policy
from src.think.joint_policies import JointPolicyBundle


def _minimal_config() -> dict:
    return {
        "mdp_contract_version": MDP_CONTRACT_VERSION,
        "shared_global_parameters": {
            "environment_id": "unit_test_joint_runtime",
            "observation_dim": OBSERVATION_DIM,
        },
    }


def _write_checkpoint(path: Path, *, overrides: dict | None = None) -> None:
    policies = JointPolicyBundle()
    checkpoint = policies.build_checkpoint(global_step=123, config=_minimal_config())
    checkpoint["artifact_kind"] = "joint_final"
    if overrides:
        checkpoint.update(overrides)
    torch.save(checkpoint, path)


class TorchJointRuntimePolicyTests(unittest.TestCase):
    def test_load_valid_joint_checkpoint_and_predict_deterministically(self) -> None:
        with TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "joint_torch_latest.pt"
            _write_checkpoint(checkpoint_path)

            policy = load_torch_joint_policy(checkpoint_path, device=torch.device("cpu"))
            observation = np.zeros(OBSERVATION_DIM, dtype=np.float32)
            first = policy.predict_joint(observation, deterministic=True)
            second = policy.predict_joint(observation, deterministic=True)

            self.assertEqual((CONTINUOUS_ACTION_DIM,), first.continuous.shape)
            self.assertTrue(np.all(first.continuous >= -1.0))
            self.assertTrue(np.all(first.continuous <= 1.0))
            self.assertGreaterEqual(first.discrete, 0)
            self.assertLess(first.discrete, DISCRETE_ACTION_COUNT)
            np.testing.assert_allclose(first.continuous, second.continuous)
            self.assertEqual(first.discrete, second.discrete)

    def test_rejects_wrong_contract(self) -> None:
        with TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "joint_torch_latest.pt"
            bad_config = _minimal_config()
            bad_config["mdp_contract_version"] = "old_contract"
            _write_checkpoint(checkpoint_path, overrides={"config": bad_config})

            with self.assertRaisesRegex(ValueError, "MDP contract"):
                load_torch_joint_policy(checkpoint_path, device=torch.device("cpu"))

    def test_rejects_wrong_observation_shape(self) -> None:
        with TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "joint_torch_latest.pt"
            _write_checkpoint(checkpoint_path)
            policy = load_torch_joint_policy(checkpoint_path, device=torch.device("cpu"))

            with self.assertRaisesRegex(ValueError, "observation"):
                policy.predict_joint(np.zeros(OBSERVATION_DIM - 1, dtype=np.float32))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```powershell
python -m unittest tests.orchestration.test_torch_joint_runtime -v
```

Expected: FAIL because `src.orchestration.torch_joint_runtime` does not exist.

- [ ] **Step 3: Implement the loader**

Create `src/orchestration/torch_joint_runtime.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.act.action_projector import CONTINUOUS_ACTION_DIM
from src.act.discrete_action_mapper import DISCRETE_ACTION_COUNT
from src.act.observation_builder import OBSERVATION_DIM
from src.learn.train_joint_torch import MDP_CONTRACT_VERSION
from src.think.joint_policies import CHECKPOINT_VERSION, JointPolicyBundle


@dataclass(frozen=True, slots=True)
class TorchJointAction:
    continuous: np.ndarray
    discrete: int


@dataclass(slots=True)
class TorchJointRuntimePolicy:
    checkpoint_path: Path
    checkpoint: Mapping[str, Any]
    policies: JointPolicyBundle
    device: torch.device

    def predict_joint(self, observation: np.ndarray, *, deterministic: bool = True) -> TorchJointAction:
        observation_array = np.asarray(observation, dtype=np.float32).reshape(-1)
        if observation_array.shape != (OBSERVATION_DIM,):
            raise ValueError(f"expected observation dimension {OBSERVATION_DIM}, got {observation_array.shape}")
        observation_tensor = torch.as_tensor(observation_array, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            ppo = self.policies.select_ppo_action(observation_tensor, deterministic=deterministic)
            dqn = self.policies.select_dqn_action(observation_tensor, epsilon=0.0)
        continuous = ppo.action.squeeze(0).detach().cpu().numpy().astype(np.float32)
        discrete = int(dqn.action.squeeze(0).detach().cpu().item())
        if continuous.shape != (CONTINUOUS_ACTION_DIM,):
            raise ValueError(f"expected continuous action dim {CONTINUOUS_ACTION_DIM}, got {continuous.shape}")
        if not 0 <= discrete < DISCRETE_ACTION_COUNT:
            raise ValueError(f"discrete action must be in [0, {DISCRETE_ACTION_COUNT - 1}], got {discrete}")
        return TorchJointAction(continuous=continuous, discrete=discrete)


def load_torch_joint_policy(path: Path, *, device: torch.device) -> TorchJointRuntimePolicy:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping.")
    _validate_torch_joint_checkpoint(checkpoint)
    policies = JointPolicyBundle(
        observation_dim=int(checkpoint["observation_dim"]),
        continuous_action_dim=int(checkpoint["continuous_action_dim"]),
        discrete_action_count=int(checkpoint["discrete_action_count"]),
    ).to(device)
    policies.load_checkpoint_state(dict(checkpoint))
    policies.eval()
    return TorchJointRuntimePolicy(
        checkpoint_path=path,
        checkpoint=checkpoint,
        policies=policies,
        device=device,
    )


def _validate_torch_joint_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported checkpoint_version: expected {CHECKPOINT_VERSION!r}, got {checkpoint.get('checkpoint_version')!r}"
        )
    artifact_kind = checkpoint.get("artifact_kind")
    if artifact_kind not in {"joint", "joint_final"}:
        raise ValueError(f"expected joint artifact_kind, got {artifact_kind!r}")
    config = checkpoint.get("config")
    version = config.get("mdp_contract_version") if isinstance(config, Mapping) else None
    if version != MDP_CONTRACT_VERSION:
        raise ValueError(f"MDP contract mismatch: expected {MDP_CONTRACT_VERSION!r}, got {version!r}")
    if int(checkpoint.get("observation_dim", -1)) != OBSERVATION_DIM:
        raise ValueError(f"checkpoint observation_dim mismatch: expected {OBSERVATION_DIM}")
    if int(checkpoint.get("continuous_action_dim", -1)) != CONTINUOUS_ACTION_DIM:
        raise ValueError(f"checkpoint continuous_action_dim mismatch: expected {CONTINUOUS_ACTION_DIM}")
    if int(checkpoint.get("discrete_action_count", -1)) != DISCRETE_ACTION_COUNT:
        raise ValueError(f"checkpoint discrete_action_count mismatch: expected {DISCRETE_ACTION_COUNT}")
```

- [ ] **Step 4: Run loader tests**

Run:
```powershell
python -m unittest tests.orchestration.test_torch_joint_runtime -v
```

Expected: PASS.

### Task 2: Add Registry Pair Resolution For Active Torch Joint Records

**Files:**
- Modify: `src/orchestration/policy_service.py`
- Test: `tests/orchestration/test_policy_service_torch_joint.py`

- [ ] **Step 1: Write active-pair tests**

Create `tests/orchestration/test_policy_service_torch_joint.py` with tests proving:
- missing active entries still use heuristic fallback when allowed
- candidate records alone are not active
- active PPO and DQN records with matching `logical_model_id` and `joint_checkpoint` resolve to one Torch joint policy
- mismatched PPO/DQN active metadata raises an error
- active Torch `.pt` records are not passed to SB3 loaders

```python
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.act.observation_builder import OBSERVATION_DIM
from src.learn.model_registry import ModelRegistry
from src.orchestration.policy_service import PolicyService


class PolicyServiceTorchJointTests(unittest.TestCase):
    def test_candidate_records_do_not_activate_joint_runtime(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            joint = root / "joint_torch_latest.pt"
            joint.write_text("placeholder", encoding="utf-8")
            registry = ModelRegistry(root / "registry")
            metadata = {
                "logical_model_id": "candidate_only",
                "joint_checkpoint": str(joint),
                "framework": "torch_joint",
                "runtime_loader": "joint_torch",
                "contract": "physical_reality_v5_route_candidate_visibility",
                "obs_dim": 73,
                "action_count": 48,
            }
            registry.register(
                algorithm="ppo",
                path=root / "ppo.pt",
                agent_role="continuous_control",
                status="candidate",
                metadata={**metadata, "artifact_kind": "ppo_final"},
            )
            registry.register(
                algorithm="dqn",
                path=root / "dqn.pt",
                agent_role="tactical_dispatch",
                status="candidate",
                metadata={**metadata, "artifact_kind": "dqn_final"},
            )
            service = PolicyService(registry=registry)

            inference = service.predict_joint(np.zeros(OBSERVATION_DIM, dtype=np.float32), fallback_to_heuristic=True)

            self.assertEqual("joint", inference.algorithm)
            self.assertEqual({"ppo": None, "dqn": None}, inference.model_path)

    def test_active_torch_joint_pair_uses_joint_loader_not_sb3(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry = ModelRegistry(root / "registry")
            joint = root / "joint_torch_latest.pt"
            ppo = root / "ppo.pt"
            dqn = root / "dqn.pt"
            for path in (joint, ppo, dqn):
                path.write_text("placeholder", encoding="utf-8")
            metadata = {
                "logical_model_id": "paired_active",
                "joint_checkpoint": str(joint),
                "framework": "torch_joint",
                "runtime_loader": "joint_torch",
                "contract": "physical_reality_v5_route_candidate_visibility",
                "obs_dim": 73,
                "action_count": 48,
            }
            registry.register(
                algorithm="ppo",
                path=ppo,
                agent_role="continuous_control",
                status="active",
                metadata={**metadata, "artifact_kind": "ppo_final"},
            )
            registry.register(
                algorithm="dqn",
                path=dqn,
                agent_role="tactical_dispatch",
                status="active",
                metadata={**metadata, "artifact_kind": "dqn_final"},
            )
            service = PolicyService(registry=registry)

            fake_action = type("FakeAction", (), {"continuous": np.zeros(5, dtype=np.float32), "discrete": 7})()
            fake_policy = type("FakeTorchPolicy", (), {"predict_joint": lambda self, observation, deterministic=True: fake_action})()

            with patch("src.orchestration.policy_service.load_torch_joint_policy", return_value=fake_policy) as loader:
                with patch("src.orchestration.policy_service.PPO.load") as ppo_load:
                    with patch("src.orchestration.policy_service.DQN.load") as dqn_load:
                        inference = service.predict_joint(
                            np.zeros(OBSERVATION_DIM, dtype=np.float32),
                            deterministic=True,
                            fallback_to_heuristic=False,
                        )

            loader.assert_called_once_with(joint, device=torch.device("cpu"))
            ppo_load.assert_not_called()
            dqn_load.assert_not_called()
            self.assertEqual("torch_joint", inference.algorithm)
            self.assertEqual(7, inference.action["discrete"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```powershell
python -m unittest tests.orchestration.test_policy_service_torch_joint -v
```

Expected: FAIL because paired active Torch joint resolution is not implemented.

- [ ] **Step 3: Implement pair resolution**

Modify `src/orchestration/policy_service.py`:
- add a `device: torch.device | str = "cpu"` constructor argument
- import `load_torch_joint_policy`
- add `load_active_joint()`
- add `_active_record_for_path()`
- add `_resolve_active_torch_joint_pair()`
- update `predict_joint()` to use the Torch joint pair before falling back to SB3/heuristic behavior

The implementation must validate both active records before loading the joint checkpoint and must not call `PPO.load()` or `DQN.load()` for `torch_joint` records.

- [ ] **Step 4: Run focused tests**

Run:
```powershell
python -m unittest tests.orchestration.test_policy_service_torch_joint tests.learn.test_model_registry -v
python -m py_compile src\orchestration\policy_service.py src\orchestration\torch_joint_runtime.py tests\orchestration\test_torch_joint_runtime.py tests\orchestration\test_policy_service_torch_joint.py
```

Expected: PASS.

### Task 3: Add Runtime Contract Tests Against The Real Candidate Checkpoint

**Files:**
- Modify: `tests/orchestration/test_torch_joint_runtime.py`

- [ ] **Step 1: Add read-only candidate smoke tests**

Add tests that load the real candidate checkpoint read-only and verify metadata plus deterministic action shape. Keep the test CPU-only and do not step the environment.

```python
def test_real_balanced_retention_candidate_loads_read_only(self) -> None:
    checkpoint = Path(
        "models/checkpoints/joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608/"
        "joint_torch_latest.pt"
    )
    self.assertTrue(checkpoint.exists())

    policy = load_torch_joint_policy(checkpoint, device=torch.device("cpu"))
    self.assertEqual("joint_final", policy.checkpoint["artifact_kind"])
    self.assertEqual(200000, int(policy.checkpoint["global_step"]))
    action = policy.predict_joint(np.zeros(OBSERVATION_DIM, dtype=np.float32), deterministic=True)

    self.assertEqual((CONTINUOUS_ACTION_DIM,), action.continuous.shape)
    self.assertGreaterEqual(action.discrete, 0)
    self.assertLess(action.discrete, DISCRETE_ACTION_COUNT)
```

- [ ] **Step 2: Run focused tests**

Run:
```powershell
python -m unittest tests.orchestration.test_torch_joint_runtime -v
```

Expected: PASS.

### Task 4: Add Safe Registry Candidate-To-Active Promotion Command

**Files:**
- Create: `src/learn/register_torch_joint_candidate.py`
- Test: `tests/learn/test_register_torch_joint_candidate.py`

- [ ] **Step 1: Write CLI tests**

Write tests proving the command can:
- locate exactly the two expected candidate ids
- validate candidate metadata
- perform `--dry-run` without writing registry files
- write active records only when `--activate` is supplied
- refuse activation if hard blocker status is not `zero`
- refuse activation if eval verdict is not `PASS`
- refuse activation if PPO/DQN candidates point to different joint checkpoints

The CLI should not copy checkpoints, touch production, touch baselines, or mutate DB rows.

- [ ] **Step 2: Implement CLI with dry-run default**

The command should require candidate ids and default to dry-run:

```powershell
python -m src.learn.register_torch_joint_candidate `
  --logical-model-id joint_torch_v5_balanced_retention_ft_200k_20260608 `
  --ppo-candidate-id 556b047a-7178-43bc-832c-0f9b49e6ef3c `
  --dqn-candidate-id 9fe92e2b-3ac2-46c3-9ade-3b6e3d8d5686 `
  --dry-run
```

Activation must require an explicit flag:

```powershell
python -m src.learn.register_torch_joint_candidate `
  --logical-model-id joint_torch_v5_balanced_retention_ft_200k_20260608 `
  --ppo-candidate-id 556b047a-7178-43bc-832c-0f9b49e6ef3c `
  --dqn-candidate-id 9fe92e2b-3ac2-46c3-9ade-3b6e3d8d5686 `
  --activate
```

Activation should append two new active registry rows and write `active_models.json` with:
- `ppo:continuous_control` pointing to the active PPO final `.pt`
- `dqn:tactical_dispatch` pointing to the active DQN final `.pt`

Do not run the activation command until Tasks 1-3 are complete and verified.

- [ ] **Step 3: Run focused registry CLI tests**

Run:
```powershell
python -m unittest tests.learn.test_register_torch_joint_candidate tests.learn.test_model_registry -v
python -m py_compile src\learn\register_torch_joint_candidate.py tests\learn\test_register_torch_joint_candidate.py
```

Expected: PASS.

### Task 5: Production Promotion Gate Checklist

**Files:**
- Create: `docs/runbooks/torch_joint_runtime_promotion_checklist.md`

- [ ] **Step 1: Write the checklist**

The checklist must require:
- candidate ids match the expected PPO/DQN ids
- candidate checkpoint exists and has `checkpoint_version=torch_joint_policy_v1`
- candidate checkpoint has `artifact_kind=joint_final`
- contract is `physical_reality_v5_route_candidate_visibility`
- observation/action metadata is `73/48`
- offline scenario evaluation is PASS for all eight scenarios
- global hard blockers are zero
- residual watches are accepted by a human reviewer
- `PolicyService` Torch joint tests pass
- real candidate read-only runtime load test passes
- dry-run activation command output is reviewed
- registry active write is run only once, manually
- production promotion remains a later separate step
- baseline update remains a later separate step

- [ ] **Step 2: Verify checklist content**

Run:
```powershell
Select-String -Path docs\runbooks\torch_joint_runtime_promotion_checklist.md -Pattern "active_models.json","baseline","production","hard blockers","dry-run","rollback"
```

Expected: every required gate appears.

## Production Promotion Gates

Do not promote to production until all gates pass:

1. Torch joint runtime loader tests pass.
2. PolicyService paired-active tests pass.
3. Real candidate checkpoint read-only runtime load test passes.
4. Registry activation CLI dry-run passes and shows only the intended registry writes.
5. Manual reviewer accepts residual watches.
6. Active registry write is performed manually and once.
7. Post-active audit confirms:
   - `active_models.json` contains only intended active PPO/DQN role keys.
   - active records reference the expected logical model id.
   - `PolicyService.predict_joint(..., fallback_to_heuristic=False)` returns `algorithm="torch_joint"`.
   - no SB3 loader is called for Torch joint artifacts.
8. A separate production promotion plan is created and reviewed.

## Rollback Plan

If active activation causes a runtime problem:

1. Save a copy of current `models/registry/active_models.json` for audit.
2. Restore `models/registry/active_models.json` to `{}` or to the previous known-good active mapping.
3. Do not delete candidate rows from `models/registry/models.jsonl`; append archival/rollback notes only if a registry note mechanism is added later.
4. Do not delete or mutate checkpoints.
5. Do not overwrite baselines.
6. Rerun the read-only registry/runtime audit:
   ```powershell
   python -m unittest tests.learn.test_model_registry tests.orchestration.test_policy_service_torch_joint -v
   ```
7. Confirm `PolicyService.predict_joint(..., fallback_to_heuristic=True)` returns heuristic fallback when no active mapping exists.

## Baseline Update Policy

Baseline update must remain separate from registry activation and production promotion.

Reasons:
- Baselines are comparison artifacts, not runtime active pointers.
- Updating baselines can hide regressions in later evaluations.
- Candidate activation can be rolled back by editing registry active state; baseline overwrite cannot be treated as the same reversible operation.

Baseline update requires a separate plan, separate approval, and a new audit of baseline directory mutations.

## Recommended Next Implementation Step

Start with Task 1 only: implement `src/orchestration/torch_joint_runtime.py` and `tests/orchestration/test_torch_joint_runtime.py`. This creates a production-safe Torch joint loader without touching registry active state, production directories, baselines, DB rows, checkpoints, training, or evaluation outputs.

After Task 1 passes, proceed to Task 2 to integrate the loader into `PolicyService.predict_joint()` behind explicit paired-active registry validation.
