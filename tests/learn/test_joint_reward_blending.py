from __future__ import annotations

import unittest

import torch

from src.act.action_projector import CONTINUOUS_ACTION_DIM
from src.act.observation_builder import OBSERVATION_DIM
from src.learn.joint_buffers import (
    DQNReplayBuffer,
    JointRewardBlendConfig,
    JointTransition,
    dqn_training_reward,
    ppo_training_reward,
    smoothstep,
)


class JointRewardBlendingTests(unittest.TestCase):
    def test_smoothstep_clamps_and_smooths(self) -> None:
        self.assertEqual(smoothstep(-1.0), 0.0)
        self.assertEqual(smoothstep(0.0), 0.0)
        self.assertAlmostEqual(smoothstep(0.5), 0.5)
        self.assertEqual(smoothstep(1.0), 1.0)
        self.assertEqual(smoothstep(2.0), 1.0)

    def test_ppo_uses_role_specific_global_and_local_blend(self) -> None:
        reward = ppo_training_reward(reward_global=0.8, reward_ppo_local=0.4)
        self.assertAlmostEqual(reward, (0.35 * 0.8) + (0.65 * 0.4))

    def test_dqn_global_reward_is_gated_when_local_reward_is_badly_negative(self) -> None:
        reward = dqn_training_reward(reward_global=0.8, reward_dqn_local=-0.5)
        self.assertAlmostEqual(reward, 0.90 * -0.5)

    def test_dqn_gets_partial_global_reward_when_local_reward_is_near_neutral(self) -> None:
        reward = dqn_training_reward(reward_global=0.8, reward_dqn_local=0.0)
        local_health = smoothstep((0.0 + 0.10) / 0.25)
        expected = (0.90 * 0.0) + (0.10 * local_health * 0.8)
        self.assertGreater(local_health, 0.0)
        self.assertLess(local_health, 1.0)
        self.assertAlmostEqual(reward, expected)

    def test_dqn_gets_full_global_share_when_local_reward_is_positive_enough(self) -> None:
        reward = dqn_training_reward(reward_global=0.8, reward_dqn_local=0.2)
        expected = (0.90 * 0.2) + (0.10 * 1.0 * 0.8)
        self.assertAlmostEqual(reward, expected)

    def test_joint_transition_and_replay_use_dqn_gated_reward(self) -> None:
        transition = _transition(reward_global=0.8, reward_ppo_local=0.4, reward_dqn_local=-0.5)
        self.assertAlmostEqual(transition.reward_ppo_train, (0.35 * 0.8) + (0.65 * 0.4))
        self.assertAlmostEqual(transition.reward_dqn_train, 0.90 * -0.5)

        replay = DQNReplayBuffer(
            capacity=4,
            learning_starts=0,
            device="cpu",
            reward_blend=JointRewardBlendConfig(),
        )
        replay.add(transition)
        batch = replay.sample(1)
        self.assertAlmostEqual(float(batch.rewards.item()), transition.reward_dqn_train, places=6)


class DQNReplayBufferStorageTests(unittest.TestCase):
    def test_replay_samples_tensor_backed_ring_slots_without_list_stack_collation(self) -> None:
        replay = DQNReplayBuffer(
            capacity=3,
            learning_starts=0,
            device="cpu",
            reward_blend=JointRewardBlendConfig(),
            observation_dim=4,
        )
        for index in range(4):
            replay.add(
                _transition(
                    reward_global=0.1 * index,
                    reward_ppo_local=0.0,
                    reward_dqn_local=0.1,
                    observation=torch.full((4,), float(index), dtype=torch.float32),
                    next_observation=torch.full((4,), float(index + 1), dtype=torch.float32),
                    dqn_action=index,
                )
            )

        self.assertIsInstance(replay._observations, torch.Tensor)
        self.assertEqual(replay._observations.shape, (3, 4))

        timing: dict[str, float] = {}
        batch = replay._sample_at_indices(torch.tensor([0, 1, 2], dtype=torch.long), timing=timing)

        self.assertEqual(batch.observations.shape, (3, 4))
        self.assertEqual(batch.actions.tolist(), [3, 1, 2])
        self.assertEqual(batch.observations[:, 0].tolist(), [3.0, 1.0, 2.0])
        self.assertEqual(batch.next_observations[:, 0].tolist(), [4.0, 2.0, 3.0])
        for key in {
            "dqn_replay_sample_seconds_window",
            "dqn_tensor_build_seconds_window",
        }:
            self.assertIn(key, timing)
            self.assertGreaterEqual(timing[key], 0.0)
        self.assertNotIn("dqn_validation_seconds_window", timing)

        validation_timing: dict[str, float] = {}
        replay._sample_at_indices(torch.tensor([0], dtype=torch.long), timing=validation_timing, validate_tensors=True)
        self.assertIn("dqn_validation_seconds_window", validation_timing)
        self.assertGreaterEqual(validation_timing["dqn_validation_seconds_window"], 0.0)

    def test_replay_rejects_non_finite_observations_before_storage(self) -> None:
        replay = DQNReplayBuffer(
            capacity=2,
            learning_starts=0,
            device="cpu",
            reward_blend=JointRewardBlendConfig(),
            observation_dim=2,
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            replay.add(
                _transition(
                    reward_global=0.0,
                    reward_ppo_local=0.0,
                    reward_dqn_local=0.0,
                    observation=torch.tensor([1.0, float("nan")], dtype=torch.float32),
                    next_observation=torch.zeros(2, dtype=torch.float32),
                )
            )

    def test_replay_uses_bootstrap_done_so_truncation_remains_bootstrappable(self) -> None:
        replay = DQNReplayBuffer(
            capacity=2,
            learning_starts=0,
            device="cpu",
            reward_blend=JointRewardBlendConfig(),
            observation_dim=2,
        )
        replay.add(
            _transition(
                reward_global=0.0,
                reward_ppo_local=0.0,
                reward_dqn_local=0.0,
                observation=torch.zeros(2, dtype=torch.float32),
                next_observation=torch.ones(2, dtype=torch.float32),
                terminated=False,
                truncated=True,
            )
        )
        replay.add(
            _transition(
                reward_global=0.0,
                reward_ppo_local=0.0,
                reward_dqn_local=0.0,
                observation=torch.ones(2, dtype=torch.float32),
                next_observation=torch.full((2,), 2.0, dtype=torch.float32),
                terminated=True,
                truncated=False,
            )
        )

        batch = replay._sample_at_indices(torch.tensor([0, 1], dtype=torch.long))

        self.assertEqual(batch.dones.tolist(), [False, True])

    def test_replay_state_dict_round_trips_ring_storage_and_cursor(self) -> None:
        source = DQNReplayBuffer(
            capacity=3,
            learning_starts=2,
            device="cpu",
            reward_blend=JointRewardBlendConfig(),
            observation_dim=4,
        )
        for index in range(5):
            source.add(
                _transition(
                    reward_global=0.1,
                    reward_ppo_local=0.0,
                    reward_dqn_local=0.1 * index,
                    observation=torch.full((4,), float(index), dtype=torch.float32),
                    next_observation=torch.full((4,), float(index + 1), dtype=torch.float32),
                    dqn_action=index,
                    terminated=index == 4,
                )
            )

        state = source.state_dict()
        target = DQNReplayBuffer(
            capacity=3,
            learning_starts=2,
            device="cpu",
            reward_blend=JointRewardBlendConfig(),
            observation_dim=4,
        )

        target.load_state_dict(state)

        self.assertEqual(len(target), 3)
        self.assertEqual(target.total_added, 5)
        self.assertTrue(target.can_sample)
        self.assertEqual(state["position"], 2)
        indices = torch.tensor([0, 1, 2], dtype=torch.long)
        source_batch = source._sample_at_indices(indices)
        target_batch = target._sample_at_indices(indices)
        self.assertTrue(torch.equal(target_batch.observations, source_batch.observations))
        self.assertTrue(torch.equal(target_batch.actions, source_batch.actions))
        self.assertTrue(torch.equal(target_batch.rewards, source_batch.rewards))
        self.assertTrue(torch.equal(target_batch.next_observations, source_batch.next_observations))
        self.assertTrue(torch.equal(target_batch.dones, source_batch.dones))


def _transition(
    *,
    reward_global: float,
    reward_ppo_local: float,
    reward_dqn_local: float,
    observation: torch.Tensor | None = None,
    next_observation: torch.Tensor | None = None,
    dqn_action: int = 0,
    terminated: bool = False,
    truncated: bool = False,
) -> JointTransition:
    observation_tensor = torch.zeros(OBSERVATION_DIM, dtype=torch.float32) if observation is None else observation
    next_observation_tensor = (
        torch.zeros_like(observation_tensor, dtype=torch.float32) if next_observation is None else next_observation
    )
    return JointTransition.from_reward_components(
        observation=observation_tensor,
        ppo_action=torch.zeros(CONTINUOUS_ACTION_DIM, dtype=torch.float32),
        ppo_log_prob=torch.tensor(0.0, dtype=torch.float32),
        ppo_value=torch.tensor(0.0, dtype=torch.float32),
        dqn_action=torch.tensor(dqn_action, dtype=torch.long),
        reward_total=reward_global + reward_ppo_local + reward_dqn_local,
        reward_global=reward_global,
        reward_ppo_local=reward_ppo_local,
        reward_dqn_local=reward_dqn_local,
        reward_blend=JointRewardBlendConfig(),
        next_observation=next_observation_tensor,
        terminated=terminated,
        truncated=truncated,
        projected=False,
        blocked=False,
    )


if __name__ == "__main__":
    unittest.main()
