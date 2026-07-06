from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from src.act.action_projector import CONTINUOUS_ACTION_DIM
from src.act.discrete_action_mapper import DISCRETE_ACTION_COUNT
from src.act.observation_builder import OBSERVATION_DIM
from src.learn.train_joint_torch import MDP_CONTRACT_VERSION
from src.orchestration.torch_joint_runtime import EXPECTED_MDP_CONTRACT, load_torch_joint_policy
from src.think.joint_policies import CHECKPOINT_VERSION, HIERARCHICAL_DQN_ARCHITECTURE, JointPolicyBundle


BALANCED_RETENTION_CANDIDATE = Path(
    "models/checkpoints/"
    "joint_torch_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k_20260608/"
    "joint_torch_latest.pt"
)


def _valid_training_config() -> dict:
    return {
        "mdp_contract_version": MDP_CONTRACT_VERSION,
        "shared_global_parameters": {
            "environment_id": "unit_test_torch_joint_runtime",
            "observation_dim": OBSERVATION_DIM,
        },
    }


def _write_checkpoint(path: Path, *, overrides: dict | None = None, dqn_architecture: str | None = None) -> dict:
    checkpoint = JointPolicyBundle(
        **({"dqn_architecture": dqn_architecture} if dqn_architecture is not None else {})
    ).build_checkpoint(
        global_step=123,
        config=_valid_training_config(),
    )
    checkpoint["artifact_kind"] = "joint_final"
    if overrides:
        checkpoint.update(overrides)
    torch.save(checkpoint, path)
    return checkpoint


class TorchJointRuntimeTests(unittest.TestCase):
    def test_runtime_contract_constant_matches_training_contract(self) -> None:
        self.assertEqual(MDP_CONTRACT_VERSION, EXPECTED_MDP_CONTRACT)

    def test_load_valid_joint_checkpoint_and_predict_deterministically(self) -> None:
        with TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "joint_torch_latest.pt"
            _write_checkpoint(checkpoint_path)

            policy = load_torch_joint_policy(checkpoint_path, device=torch.device("cpu"))
            observation = np.zeros(OBSERVATION_DIM, dtype=np.float32)
            first = policy.predict_joint(observation, deterministic=True)
            second = policy.predict_joint(observation, deterministic=True)

            self.assertEqual((CONTINUOUS_ACTION_DIM,), first.continuous.shape)
            self.assertTrue(np.all(np.isfinite(first.continuous)))
            self.assertTrue(np.all(first.continuous >= -1.0))
            self.assertTrue(np.all(first.continuous <= 1.0))
            self.assertIsInstance(first.discrete, int)
            self.assertGreaterEqual(first.discrete, 0)
            self.assertLess(first.discrete, DISCRETE_ACTION_COUNT)
            np.testing.assert_allclose(first.continuous, second.continuous)
            self.assertEqual(first.discrete, second.discrete)

    def test_load_hierarchical_joint_checkpoint_and_predicts_external_action(self) -> None:
        with TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "joint_torch_latest.pt"
            _write_checkpoint(checkpoint_path, dqn_architecture=HIERARCHICAL_DQN_ARCHITECTURE)

            policy = load_torch_joint_policy(checkpoint_path, device=torch.device("cpu"))
            action = policy.predict_joint(np.zeros(OBSERVATION_DIM, dtype=np.float32), deterministic=True)

            self.assertEqual(HIERARCHICAL_DQN_ARCHITECTURE, policy.policies.dqn_architecture)
            self.assertIsInstance(action.discrete, int)
            self.assertGreaterEqual(action.discrete, 0)
            self.assertLess(action.discrete, DISCRETE_ACTION_COUNT)

    def test_rejects_wrong_contract(self) -> None:
        with TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "joint_torch_latest.pt"
            config = _valid_training_config()
            config["mdp_contract_version"] = "old_contract"
            _write_checkpoint(checkpoint_path, overrides={"config": config})

            with self.assertRaisesRegex(ValueError, "contract"):
                load_torch_joint_policy(checkpoint_path, device=torch.device("cpu"))

    def test_rejects_wrong_observation_dim_metadata(self) -> None:
        with TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "joint_torch_latest.pt"
            _write_checkpoint(checkpoint_path, overrides={"observation_dim": OBSERVATION_DIM - 1})

            with self.assertRaisesRegex(ValueError, "observation_dim"):
                load_torch_joint_policy(checkpoint_path, device=torch.device("cpu"))

    def test_rejects_wrong_action_count_metadata(self) -> None:
        with TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "joint_torch_latest.pt"
            _write_checkpoint(checkpoint_path, overrides={"discrete_action_count": DISCRETE_ACTION_COUNT + 1})

            with self.assertRaisesRegex(ValueError, "discrete_action_count"):
                load_torch_joint_policy(checkpoint_path, device=torch.device("cpu"))

    def test_rejects_wrong_artifact_kind(self) -> None:
        for artifact_kind in ("ppo_final", "dqn_final"):
            with self.subTest(artifact_kind=artifact_kind):
                with TemporaryDirectory() as temp_dir:
                    checkpoint_path = Path(temp_dir) / f"{artifact_kind}.pt"
                    _write_checkpoint(checkpoint_path, overrides={"artifact_kind": artifact_kind})

                    with self.assertRaisesRegex(ValueError, "artifact_kind"):
                        load_torch_joint_policy(checkpoint_path, device=torch.device("cpu"))

    def test_rejects_wrong_runtime_observation_shape(self) -> None:
        with TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "joint_torch_latest.pt"
            _write_checkpoint(checkpoint_path)
            policy = load_torch_joint_policy(checkpoint_path, device=torch.device("cpu"))

            with self.assertRaisesRegex(ValueError, "observation"):
                policy.predict_joint(np.zeros(OBSERVATION_DIM - 1, dtype=np.float32))

    def test_real_balanced_retention_candidate_loads_read_only(self) -> None:
        self.assertTrue(
            BALANCED_RETENTION_CANDIDATE.exists(),
            f"candidate checkpoint not present: {BALANCED_RETENTION_CANDIDATE}",
        )

        policy = load_torch_joint_policy(BALANCED_RETENTION_CANDIDATE, device=torch.device("cpu"))
        checkpoint = policy.checkpoint
        config = checkpoint["config"]
        shared = config["shared_global_parameters"]

        self.assertEqual("torch_joint_policy_v1", checkpoint["checkpoint_version"])
        self.assertEqual(CHECKPOINT_VERSION, checkpoint["checkpoint_version"])
        self.assertEqual("joint_final", checkpoint["artifact_kind"])
        self.assertEqual(200000, int(checkpoint["global_step"]))
        self.assertEqual(MDP_CONTRACT_VERSION, config["mdp_contract_version"])
        self.assertEqual(EXPECTED_MDP_CONTRACT, config["mdp_contract_version"])
        self.assertEqual(OBSERVATION_DIM, int(checkpoint["observation_dim"]))
        self.assertEqual(OBSERVATION_DIM, int(shared["observation_dim"]))
        self.assertEqual(CONTINUOUS_ACTION_DIM, int(checkpoint["continuous_action_dim"]))
        self.assertEqual(DISCRETE_ACTION_COUNT, int(checkpoint["discrete_action_count"]))

        observation = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        first = policy.predict_joint(observation, deterministic=True)
        second = policy.predict_joint(observation, deterministic=True)
        self.assertEqual((CONTINUOUS_ACTION_DIM,), first.continuous.shape)
        self.assertEqual(np.float32, first.continuous.dtype)
        self.assertTrue(np.all(np.isfinite(first.continuous)))
        self.assertTrue(np.all(first.continuous >= -1.0))
        self.assertTrue(np.all(first.continuous <= 1.0))
        self.assertIsInstance(first.discrete, int)
        self.assertGreaterEqual(first.discrete, 0)
        self.assertLess(first.discrete, DISCRETE_ACTION_COUNT)
        np.testing.assert_allclose(first.continuous, second.continuous)
        self.assertEqual(first.discrete, second.discrete)
        with self.assertRaisesRegex(ValueError, "observation"):
            policy.predict_joint(np.zeros(OBSERVATION_DIM - 1, dtype=np.float32), deterministic=True)


if __name__ == "__main__":
    unittest.main()
