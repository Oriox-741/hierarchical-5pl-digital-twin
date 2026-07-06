from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

import src.learn.train_joint_torch as train_joint_torch
from src.act.discrete_action_mapper import (
    DISCRETE_ACTION_COUNT,
    DiscreteActionMapper,
    DiscreteLogisticsAction,
    DispatchDecision,
    ModeDecision,
    ReorderDecision,
    RouteDecision,
)
from src.act.observation_builder import OBSERVATION_DIM
from src.learn.hierarchical_dqn_initialization import (
    HIERARCHICAL_INIT_METHOD_FLAT_TEACHER_DISTILLATION,
    HIERARCHICAL_INIT_METHOD_KEY,
    HierarchicalDQNInitializationConfig,
    compute_flat_to_hierarchical_distillation_loss,
    distill_hierarchical_dqn_from_flat_teacher,
    load_flat_dqn_teacher_checkpoint,
    load_hierarchical_distillation_initial_checkpoint,
    sample_hierarchical_init_observation_bank,
)
from src.learn.joint_buffers import DQNReplayBuffer, JointRewardBlendConfig
from src.learn.train_joint_torch import TrainingState, build_checkpoint_payload
from src.think.joint_policies import FLAT_DQN_ARCHITECTURE, HIERARCHICAL_DQN_ARCHITECTURE, JointPolicyBundle


def _valid_config() -> dict:
    return {
        "mdp_contract_version": train_joint_torch.MDP_CONTRACT_VERSION,
        "shared_global_parameters": {"observation_dim": OBSERVATION_DIM},
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_policy_checkpoint(path: Path, policy: JointPolicyBundle) -> dict:
    checkpoint = policy.build_checkpoint(global_step=200_000, config=_valid_config())
    torch.save(checkpoint, path)
    return checkpoint


def _force_flat_teacher_action(policy: JointPolicyBundle, action_id: int) -> None:
    for parameter in policy.dqn_q_network.parameters():
        torch.nn.init.zeros_(parameter)
    final_layer = policy.dqn_q_network.net[-1]
    final_layer.bias.data[action_id] = 12.0
    policy.hard_update_dqn_target()


def _zero_hierarchical_student_dqn(policy: JointPolicyBundle) -> None:
    for parameter in policy.dqn_q_network.parameters():
        torch.nn.init.zeros_(parameter)
    policy.hard_update_dqn_target()


class HierarchicalDQNInitializationTests(unittest.TestCase):
    def test_flat_checkpoint_still_rejects_hierarchical_exact_resume(self) -> None:
        checkpoint = JointPolicyBundle(dqn_architecture=FLAT_DQN_ARCHITECTURE).build_checkpoint(
            global_step=200_000,
            config=_valid_config(),
        )

        with self.assertRaisesRegex(ValueError, "dqn_architecture mismatch"):
            JointPolicyBundle(dqn_architecture=HIERARCHICAL_DQN_ARCHITECTURE).load_checkpoint_state(checkpoint)

    def test_flat_teacher_loads_read_only_for_distillation(self) -> None:
        source = JointPolicyBundle(dqn_architecture=FLAT_DQN_ARCHITECTURE)

        with TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "flat_teacher.pt"
            _write_policy_checkpoint(checkpoint_path, source)
            before_hash = _sha256(checkpoint_path)

            teacher = load_flat_dqn_teacher_checkpoint(
                checkpoint_path,
                device=torch.device("cpu"),
                teacher_name="production",
            )

            self.assertEqual(before_hash, _sha256(checkpoint_path))
            self.assertEqual("production", teacher.name)
            self.assertEqual(FLAT_DQN_ARCHITECTURE, teacher.policy.dqn_architecture)
            self.assertFalse(teacher.policy.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in teacher.policy.parameters()))

    def test_sampled_observation_bank_is_finite_normalized_and_contract_sized(self) -> None:
        observations = sample_hierarchical_init_observation_bank(
            sample_count=17,
            seed=123,
            device=torch.device("cpu"),
        )

        self.assertEqual((17, OBSERVATION_DIM), tuple(observations.shape))
        self.assertTrue(torch.isfinite(observations).all())
        self.assertGreaterEqual(float(observations.min().item()), 0.0)
        self.assertLessEqual(float(observations.max().item()), 1.0)

    def test_flat_to_hierarchical_distillation_loss_is_finite_and_bounded(self) -> None:
        teacher = JointPolicyBundle(dqn_architecture=FLAT_DQN_ARCHITECTURE)
        student = JointPolicyBundle(dqn_architecture=HIERARCHICAL_DQN_ARCHITECTURE)
        observations = torch.rand((8, OBSERVATION_DIM), dtype=torch.float32)
        config = HierarchicalDQNInitializationConfig(
            enabled=True,
            method=HIERARCHICAL_INIT_METHOD_FLAT_TEACHER_DISTILLATION,
            teacher_checkpoint=Path("teacher.pt"),
            max_loss=0.05,
        )

        result = compute_flat_to_hierarchical_distillation_loss(
            student_policy=student,
            teacher_policy=teacher,
            observations=observations,
            config=config,
        )

        self.assertTrue(torch.isfinite(result.loss).all())
        self.assertLessEqual(float(result.loss.item()), 0.05)
        self.assertGreaterEqual(result.metrics["hierarchical_distillation_action_match_rate"], 0.0)
        self.assertLessEqual(result.metrics["hierarchical_distillation_action_match_rate"], 1.0)

    def test_tiny_supervised_distillation_step_can_match_flat_teacher_action(self) -> None:
        mapper = DiscreteActionMapper()
        target_action = mapper.encode(
            DiscreteLogisticsAction(
                dispatch=DispatchDecision.DISPATCH,
                route=RouteDecision.SHORTEST,
                mode=ModeDecision.PRIMARY_FLEET,
                reorder=ReorderDecision.CONSERVATIVE,
            )
        )
        teacher = JointPolicyBundle(dqn_architecture=FLAT_DQN_ARCHITECTURE)
        _force_flat_teacher_action(teacher, target_action)
        student = JointPolicyBundle(dqn_architecture=HIERARCHICAL_DQN_ARCHITECTURE)
        _zero_hierarchical_student_dqn(student)
        observations = torch.zeros((16, OBSERVATION_DIM), dtype=torch.float32)
        config = HierarchicalDQNInitializationConfig(
            enabled=True,
            method=HIERARCHICAL_INIT_METHOD_FLAT_TEACHER_DISTILLATION,
            teacher_checkpoint=Path("teacher.pt"),
            distillation_steps=20,
            batch_size=8,
            learning_rate=0.20,
            max_loss=2.0,
            seed=7,
        )

        metrics = distill_hierarchical_dqn_from_flat_teacher(
            student_policy=student,
            teacher_policy=teacher,
            observations=observations,
            config=config,
        )

        with torch.no_grad():
            student_actions = student.dqn_q_network(observations).argmax(dim=1)
        self.assertTrue(torch.all(student_actions == target_action))
        self.assertGreaterEqual(metrics["hierarchical_distillation_action_match_rate"], 0.99)

    def test_hierarchical_distillation_initial_checkpoint_records_warm_start_metadata(self) -> None:
        flat_parent = JointPolicyBundle(dqn_architecture=FLAT_DQN_ARCHITECTURE)
        flat_parent.ppo_actor.log_std.data.fill_(-0.125)
        student = JointPolicyBundle(dqn_architecture=HIERARCHICAL_DQN_ARCHITECTURE)
        state = TrainingState(global_step=99, episode_count=3, dqn_update_count=4)
        config = HierarchicalDQNInitializationConfig(
            enabled=True,
            method=HIERARCHICAL_INIT_METHOD_FLAT_TEACHER_DISTILLATION,
            teacher_checkpoint=Path("parent.pt"),
            observation_sample_count=12,
            distillation_steps=1,
            batch_size=6,
            learning_rate=0.01,
            max_loss=1.0,
            seed=5,
        )

        with TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "parent.pt"
            _write_policy_checkpoint(checkpoint_path, flat_parent)
            before_hash = _sha256(checkpoint_path)
            config = config.with_teacher_checkpoint(checkpoint_path)

            metadata = load_hierarchical_distillation_initial_checkpoint(
                checkpoint_path,
                policies=student,
                state=state,
                device=torch.device("cpu"),
                init_config=config,
            )

            self.assertEqual(before_hash, _sha256(checkpoint_path))
            self.assertEqual(0, state.global_step)
            self.assertEqual(0, state.episode_count)
            self.assertEqual(0, state.dqn_update_count)
            self.assertEqual(HIERARCHICAL_INIT_METHOD_FLAT_TEACHER_DISTILLATION, metadata[HIERARCHICAL_INIT_METHOD_KEY])
            self.assertEqual(FLAT_DQN_ARCHITECTURE, metadata["hierarchical_init_source_dqn_architecture"])
            self.assertEqual(HIERARCHICAL_DQN_ARCHITECTURE, metadata["hierarchical_init_target_dqn_architecture"])
            self.assertEqual("fresh_replay_by_design", metadata["replay_restore_status"])
            self.assertEqual("init_from_joint_checkpoint", metadata["resume_mode"])
            self.assertTrue(torch.equal(student.ppo_actor.log_std.detach(), flat_parent.ppo_actor.log_std.detach()))

    def test_checkpoint_payload_records_hierarchical_initialization_metadata(self) -> None:
        policies = JointPolicyBundle(dqn_architecture=HIERARCHICAL_DQN_ARCHITECTURE)
        ppo_optimizer = torch.optim.Adam(
            [
                *policies.ppo_feature_extractor.parameters(),
                *policies.ppo_actor.parameters(),
                *policies.ppo_critic.parameters(),
            ],
            lr=0.001,
        )
        dqn_optimizer = torch.optim.Adam(policies.dqn_q_network.parameters(), lr=0.001)
        replay = DQNReplayBuffer(
            capacity=4,
            learning_starts=1,
            reward_blend=JointRewardBlendConfig(),
            device=torch.device("cpu"),
        )
        metadata = {
            HIERARCHICAL_INIT_METHOD_KEY: HIERARCHICAL_INIT_METHOD_FLAT_TEACHER_DISTILLATION,
            "hierarchical_distillation_steps": 4,
            "hierarchical_distillation_action_match_rate": 1.0,
        }
        config = {**_valid_config(), train_joint_torch.FINE_TUNE_INITIALIZATION_KEY: metadata}

        payload = build_checkpoint_payload(
            policies=policies,
            ppo_optimizer=ppo_optimizer,
            dqn_optimizer=dqn_optimizer,
            state=TrainingState(global_step=1),
            config=config,
            dqn_replay=replay,
            rng_state={"version": train_joint_torch.RNG_STATE_VERSION},
        )

        self.assertEqual(HIERARCHICAL_DQN_ARCHITECTURE, payload["dqn_architecture"])
        self.assertEqual(HIERARCHICAL_INIT_METHOD_FLAT_TEACHER_DISTILLATION, payload[HIERARCHICAL_INIT_METHOD_KEY])
        self.assertEqual(4, payload["hierarchical_distillation_steps"])
        self.assertEqual(1.0, payload["hierarchical_distillation_action_match_rate"])
        self.assertEqual(
            HIERARCHICAL_INIT_METHOD_FLAT_TEACHER_DISTILLATION,
            payload[train_joint_torch.FINE_TUNE_INITIALIZATION_KEY][HIERARCHICAL_INIT_METHOD_KEY],
        )


if __name__ == "__main__":
    unittest.main()
