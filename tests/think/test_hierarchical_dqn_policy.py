from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

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
from src.learn.train_joint_torch import MDP_CONTRACT_VERSION
from src.think.joint_policies import (
    FLAT_DQN_ARCHITECTURE,
    HIERARCHICAL_DQN_ARCHITECTURE,
    HierarchicalDQNQNetwork,
    JointPolicyBundle,
)


def _valid_config() -> dict:
    return {
        "mdp_contract_version": MDP_CONTRACT_VERSION,
        "shared_global_parameters": {"observation_dim": OBSERVATION_DIM},
    }


def _zero_head_weights(policy: JointPolicyBundle) -> None:
    for module in (
        policy.dqn_q_network.dispatch_head,
        policy.dqn_q_network.route_head,
        policy.dqn_q_network.mode_head,
        policy.dqn_q_network.reorder_head,
    ):
        torch.nn.init.zeros_(module.weight)
        torch.nn.init.zeros_(module.bias)


class HierarchicalDQNPolicyTests(unittest.TestCase):
    def test_hierarchical_q_network_returns_external_action_q_values(self) -> None:
        network = HierarchicalDQNQNetwork()
        observation = torch.zeros((3, OBSERVATION_DIM), dtype=torch.float32)

        q_values = network(observation)

        self.assertEqual((3, DISCRETE_ACTION_COUNT), tuple(q_values.shape))
        self.assertTrue(torch.isfinite(q_values).all())

    def test_composed_head_choices_map_to_exact_external_action_id(self) -> None:
        mapper = DiscreteActionMapper()
        policy = JointPolicyBundle(dqn_architecture=HIERARCHICAL_DQN_ARCHITECTURE)
        _zero_head_weights(policy)
        policy.dqn_q_network.dispatch_head.bias.data.copy_(torch.tensor([0.0, 10.0]))
        policy.dqn_q_network.route_head.bias.data.copy_(torch.tensor([0.0, 9.0, 0.0]))
        policy.dqn_q_network.mode_head.bias.data.copy_(torch.tensor([0.0, 8.0]))
        policy.dqn_q_network.reorder_head.bias.data.copy_(torch.tensor([0.0, 7.0, 0.0, 0.0]))
        expected_action = mapper.encode(
            DiscreteLogisticsAction(
                dispatch=DispatchDecision.DISPATCH,
                route=RouteDecision.LOW_CONGESTION,
                mode=ModeDecision.PRIMARY_FLEET,
                reorder=ReorderDecision.CONSERVATIVE,
            )
        )

        output = policy.select_dqn_action(torch.zeros((1, OBSERVATION_DIM)), epsilon=0.0)

        self.assertEqual(expected_action, int(output.action.item()))
        self.assertEqual(expected_action, int(output.greedy_action.item()))
        self.assertGreaterEqual(int(output.action.item()), 0)
        self.assertLess(int(output.action.item()), DISCRETE_ACTION_COUNT)

    def test_hold_actions_ignore_route_and_mode_head_labels(self) -> None:
        mapper = DiscreteActionMapper()
        network = HierarchicalDQNQNetwork()
        for module in (network.dispatch_head, network.route_head, network.mode_head, network.reorder_head):
            torch.nn.init.zeros_(module.weight)
            torch.nn.init.zeros_(module.bias)
        network.dispatch_head.bias.data.copy_(torch.tensor([5.0, 0.0]))
        network.route_head.bias.data.copy_(torch.tensor([0.0, 10.0, -7.0]))
        network.mode_head.bias.data.copy_(torch.tensor([-3.0, 12.0]))
        network.reorder_head.bias.data.copy_(torch.tensor([0.0, 1.5, 0.0, 0.0]))
        observation = torch.zeros((1, OBSERVATION_DIM), dtype=torch.float32)

        q_values = network(observation).squeeze(0)
        hold_action_ids = [
            mapper.encode(
                DiscreteLogisticsAction(
                    dispatch=DispatchDecision.HOLD,
                    route=route,
                    mode=mode,
                    reorder=ReorderDecision.CONSERVATIVE,
                )
            )
            for route in RouteDecision
            for mode in ModeDecision
        ]
        hold_q_values = q_values[torch.tensor(hold_action_ids, dtype=torch.long)]

        self.assertTrue(torch.allclose(hold_q_values, hold_q_values[0].expand_as(hold_q_values)))

    def test_dispatch_only_loss_updates_dispatch_head_without_route_mode_reorder_gradients(self) -> None:
        network = HierarchicalDQNQNetwork()
        observation = torch.randn((4, OBSERVATION_DIM), dtype=torch.float32)

        loss = F.cross_entropy(network.dispatch_q_values(observation), torch.ones(4, dtype=torch.long))
        loss.backward()

        self.assertIsNotNone(network.dispatch_head.weight.grad)
        self.assertGreater(float(network.dispatch_head.weight.grad.abs().sum().item()), 0.0)
        self.assertIsNone(network.route_head.weight.grad)
        self.assertIsNone(network.mode_head.weight.grad)
        self.assertIsNone(network.reorder_head.weight.grad)

    def test_hierarchical_checkpoint_records_architecture_metadata(self) -> None:
        policy = JointPolicyBundle(dqn_architecture=HIERARCHICAL_DQN_ARCHITECTURE)

        checkpoint = policy.build_checkpoint(global_step=5, config=_valid_config())

        self.assertEqual(HIERARCHICAL_DQN_ARCHITECTURE, checkpoint["dqn_architecture"])
        self.assertEqual(DISCRETE_ACTION_COUNT, checkpoint["external_discrete_action_count"])
        self.assertEqual(
            {
                "dispatch": 2,
                "route": 3,
                "mode": 2,
                "reorder": 4,
                "composition": "dispatch + reorder + dispatch_mask(route + mode)",
            },
            checkpoint["internal_heads"],
        )

    def test_legacy_missing_architecture_metadata_is_flat_v1_only(self) -> None:
        flat_policy = JointPolicyBundle(dqn_architecture=FLAT_DQN_ARCHITECTURE)
        checkpoint = flat_policy.build_checkpoint(global_step=5, config=_valid_config())
        checkpoint.pop("dqn_architecture")
        checkpoint.pop("external_discrete_action_count")
        checkpoint.pop("internal_heads")

        JointPolicyBundle(dqn_architecture=FLAT_DQN_ARCHITECTURE).load_checkpoint_state(checkpoint)
        with self.assertRaisesRegex(ValueError, "dqn_architecture mismatch"):
            JointPolicyBundle(dqn_architecture=HIERARCHICAL_DQN_ARCHITECTURE).load_checkpoint_state(checkpoint)

    def test_rejects_loading_hierarchical_checkpoint_into_flat_policy(self) -> None:
        checkpoint = JointPolicyBundle(dqn_architecture=HIERARCHICAL_DQN_ARCHITECTURE).build_checkpoint(
            global_step=5,
            config=_valid_config(),
        )

        with self.assertRaisesRegex(ValueError, "dqn_architecture mismatch"):
            JointPolicyBundle(dqn_architecture=FLAT_DQN_ARCHITECTURE).load_checkpoint_state(checkpoint)


if __name__ == "__main__":
    unittest.main()
