from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "ablation"
EXPECTED_CONFIGS = {
    "neutral_ppo_eval.json",
    "mask_route_candidates_eval.json",
    "safety_projection_diagnostic.json",
    "flat_vs_hierarchical_comparison.json",
    "no_teacher_distillation_plan.json",
    "reward_blend_sensitivity_plan.json",
}
REQUIRED_FIELDS = {
    "name",
    "type",
    "purpose",
    "required_artifacts",
    "expected_outputs",
    "claim_status",
    "safety_notes",
    "scenario_config_reference",
    "output_dir",
}


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


class AblationConfigTests(unittest.TestCase):
    def test_all_ablation_configs_exist_and_are_valid_json(self) -> None:
        missing = sorted(name for name in EXPECTED_CONFIGS if not (CONFIG_DIR / name).exists())
        self.assertEqual([], missing, f"Missing ablation config files: {missing}")

        for config_name in EXPECTED_CONFIGS:
            with self.subTest(config=config_name):
                data = _load_config(CONFIG_DIR / config_name)
                missing_fields = sorted(REQUIRED_FIELDS - data.keys())
                self.assertEqual([], missing_fields, f"{config_name} missing required fields: {missing_fields}")
                self.assertIn(data["type"], {"evaluation_time", "training_time_plan"})
                self.assertIn(data["claim_status"], {"planned", "completed"})
                self.assertTrue(str(data["output_dir"]).startswith("reports/ablation/"))

    def test_configs_do_not_claim_completed_results_without_report_files(self) -> None:
        for config_path in CONFIG_DIR.glob("*.json"):
            with self.subTest(config=config_path.name):
                data = _load_config(config_path)
                if data["claim_status"] == "completed":
                    output_dir = REPO_ROOT / data["output_dir"]
                    self.assertTrue(
                        (output_dir / "run_metadata.json").exists(),
                        f"{config_path.name} claims completed results but no run metadata exists",
                    )

    def test_training_time_ablations_are_future_plans(self) -> None:
        for name in {"no_teacher_distillation_plan.json", "reward_blend_sensitivity_plan.json"}:
            with self.subTest(config=name):
                data = _load_config(CONFIG_DIR / name)
                self.assertEqual("training_time_plan", data["type"])
                self.assertEqual("planned", data["claim_status"])
                self.assertIn("future", " ".join(data["safety_notes"]).lower())

    def test_safety_projection_diagnostic_keeps_safety_enabled_by_default(self) -> None:
        data = _load_config(CONFIG_DIR / "safety_projection_diagnostic.json")
        notes = " ".join(data["safety_notes"]).lower()
        self.assertIn("safety remains enabled by default", notes)
        self.assertIs(data.get("default_disables_safety"), False)


if __name__ == "__main__":
    unittest.main()
