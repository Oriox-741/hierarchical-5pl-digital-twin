from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.act.observation_builder import OBSERVATION_DIM
from src.learn.train_joint_torch import (
    MDP_CONTRACT_MISMATCH_MESSAGE,
    MDP_CONTRACT_VERSION,
    load_config,
    validate_active_mdp_contract,
    validate_checkpoint_mdp_contract,
)


class ColdStartContractTest(unittest.TestCase):
    def test_old_checkpoint_without_mdp_contract_version_is_rejected(self) -> None:
        checkpoint = {"config": {"schema_version": "joint_training_v1"}}

        with self.assertRaisesRegex(ValueError, MDP_CONTRACT_MISMATCH_MESSAGE):
            validate_checkpoint_mdp_contract(checkpoint)

    def test_checkpoint_with_previous_contract_version_is_rejected(self) -> None:
        checkpoint = {"config": {"mdp_contract_version": "joint_training_v1"}}

        with self.assertRaisesRegex(ValueError, MDP_CONTRACT_MISMATCH_MESSAGE):
            validate_checkpoint_mdp_contract(checkpoint)

    def test_v3_dispatch_feasibility_checkpoint_is_rejected_after_v5_contract_change(self) -> None:
        checkpoint = {
            "config": {
                "mdp_contract_version": "physical_reality_v3_dispatch_feasibility",
                "shared_global_parameters": {"observation_dim": 49},
            },
            "observation_dim": 49,
        }

        with self.assertRaisesRegex(ValueError, MDP_CONTRACT_MISMATCH_MESSAGE):
            validate_checkpoint_mdp_contract(checkpoint)

    def test_v4_stress_visibility_checkpoint_is_rejected_after_v5_contract_change(self) -> None:
        checkpoint = {
            "config": {
                "mdp_contract_version": "physical_reality_v4_real_world_stress_visibility",
                "shared_global_parameters": {"observation_dim": 57},
            },
            "observation_dim": 57,
        }

        with self.assertRaisesRegex(ValueError, MDP_CONTRACT_MISMATCH_MESSAGE):
            validate_checkpoint_mdp_contract(checkpoint)

    def test_checkpoint_with_current_route_candidate_visibility_contract_is_accepted(self) -> None:
        checkpoint = {"config": {"mdp_contract_version": MDP_CONTRACT_VERSION}}

        validate_checkpoint_mdp_contract(checkpoint)

    def test_active_config_must_declare_current_route_candidate_visibility_contract(self) -> None:
        validate_active_mdp_contract({"mdp_contract_version": MDP_CONTRACT_VERSION})

        with self.assertRaisesRegex(ValueError, "Config MDP contract mismatch"):
            validate_active_mdp_contract({"mdp_contract_version": "physical_reality_v4_real_world_stress_visibility"})

    def test_active_config_observation_dim_must_match_runtime_contract(self) -> None:
        validate_active_mdp_contract(
            {
                "mdp_contract_version": MDP_CONTRACT_VERSION,
                "shared_global_parameters": {"observation_dim": OBSERVATION_DIM},
            }
        )

        with self.assertRaisesRegex(ValueError, "Config observation dimension mismatch"):
            validate_active_mdp_contract(
                {
                    "mdp_contract_version": MDP_CONTRACT_VERSION,
                    "shared_global_parameters": {"observation_dim": 44},
                }
            )

    def test_config_loader_accepts_windows_utf8_bom_files(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "training_joint_smoke.json"
            path.write_text(
                f'{{"mdp_contract_version": "{MDP_CONTRACT_VERSION}"}}',
                encoding="utf-8-sig",
            )

            config = load_config(path)

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)


if __name__ == "__main__":
    unittest.main()
