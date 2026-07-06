from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.learn.model_registry import ModelRegistry
from src.orchestration.policy_service import PolicyService


class ModelRegistryCandidateSafetyTests(unittest.TestCase):
    def test_candidate_is_not_runtime_active_fallback_when_no_active_entry(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = self._checkpoint(root / "candidate.pt")
            registry = ModelRegistry(root / "registry")

            registry.register(
                algorithm="ppo",
                path=checkpoint,
                agent_role="continuous_control",
                status="candidate",
                metrics={"service_level": 0.99},
            )

            with self.assertRaisesRegex(FileNotFoundError, "No explicit active ppo model configured"):
                registry.active_path_for("ppo", agent_role="continuous_control")

    def test_archived_is_not_runtime_active_fallback_when_no_active_entry(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = self._checkpoint(root / "archived.zip")
            registry = ModelRegistry(root / "registry")

            registry.register(
                algorithm="dqn",
                path=checkpoint,
                agent_role="tactical_dispatch",
                status="archived",
                metrics={"service_level": 0.99},
            )

            with self.assertRaisesRegex(FileNotFoundError, "No explicit active dqn model configured"):
                registry.active_path_for("dqn", agent_role="tactical_dispatch")

    def test_explicit_active_entry_returns_active_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = self._checkpoint(root / "active.zip")
            registry = ModelRegistry(root / "registry")

            registry.register(
                algorithm="ppo",
                path=checkpoint,
                agent_role="continuous_control",
                status="active",
            )

            self.assertEqual(
                checkpoint,
                registry.active_path_for("ppo", agent_role="continuous_control"),
            )

    def test_candidate_registration_appends_without_altering_active_models(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = self._checkpoint(root / "candidate.pt")
            registry = ModelRegistry(root / "registry")
            registry.active_path.write_text("{}\n", encoding="utf-8")
            active_before = registry.active_path.read_text(encoding="utf-8")

            record = registry.register(
                algorithm="ppo",
                path=checkpoint,
                agent_role="continuous_control",
                status="candidate",
            )

            self.assertEqual(active_before, registry.active_path.read_text(encoding="utf-8"))
            lines = registry.index_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(1, len(lines))
            self.assertEqual(record.model_id, json.loads(lines[0])["model_id"])
            self.assertEqual("candidate", json.loads(lines[0])["status"])

    def test_candidate_listing_and_best_path_are_explicit_review_operations(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidate = self._checkpoint(root / "candidate.pt")
            active = self._checkpoint(root / "active.zip")
            registry = ModelRegistry(root / "registry")

            candidate_record = registry.register(
                algorithm="ppo",
                path=candidate,
                agent_role="continuous_control",
                status="candidate",
                metrics={"service_level": 0.98},
            )
            registry.register(
                algorithm="ppo",
                path=active,
                agent_role="continuous_control",
                status="active",
                metrics={"service_level": 0.50},
            )

            candidates = registry.candidate_records(
                algorithm="ppo",
                agent_role="continuous_control",
            )
            self.assertEqual([candidate_record.model_id], [record.model_id for record in candidates])
            self.assertEqual(
                candidate,
                registry.best_path("ppo", agent_role="continuous_control", status="candidate"),
            )
            self.assertEqual(
                active,
                registry.best_path("ppo", agent_role="continuous_control", status="active"),
            )

    def test_torch_joint_metadata_is_preserved_for_candidate_review(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = self._checkpoint(root / "joint_torch_latest.pt")
            registry = ModelRegistry(root / "registry")

            registry.register(
                algorithm="ppo",
                path=checkpoint,
                agent_role="continuous_control",
                status="candidate",
                metadata={
                    "framework": "torch_joint",
                    "artifact_kind": "ppo_final",
                    "runtime_loader": "joint_torch",
                },
            )

            [record] = registry.candidate_records(
                algorithm="ppo",
                agent_role="continuous_control",
            )
            self.assertEqual("torch_joint", record.metadata["framework"])
            self.assertEqual("ppo_final", record.metadata["artifact_kind"])
            self.assertEqual("joint_torch", record.metadata["runtime_loader"])

    @staticmethod
    def _checkpoint(path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("dummy checkpoint", encoding="utf-8")
        return path


class PolicyServiceArtifactCompatibilityTests(unittest.TestCase):
    def test_policy_service_rejects_torch_joint_pt_before_sb3_loader(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = ModelRegistryCandidateSafetyTests._checkpoint(root / "joint_torch_latest.pt")
            registry = ModelRegistry(root / "registry")
            registry.register(
                algorithm="ppo",
                path=checkpoint,
                agent_role="continuous_control",
                status="active",
                metadata={
                    "framework": "torch_joint",
                    "artifact_kind": "ppo_final",
                    "runtime_loader": "joint_torch",
                },
            )
            service = PolicyService(registry=registry)

            with patch("src.orchestration.policy_service.PPO.load") as ppo_load:
                with self.assertRaisesRegex(ValueError, "joint_torch"):
                    service.load_active("ppo", agent_role="continuous_control")
                ppo_load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
