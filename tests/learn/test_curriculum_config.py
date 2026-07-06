from __future__ import annotations

import unittest
from pathlib import Path

from src.act.discrete_action_mapper import DISCRETE_ACTION_COUNT
from src.act.observation_builder import OBSERVATION_DIM
from src.eval.scenario_overrides import SUPPORTED_OVERRIDE_KEYS
from src.learn.curriculum import (
    CurriculumConfig,
    CurriculumRandomizationConfig,
    CurriculumSampler,
    REAL_WORLD_CURRICULUM_STAGES,
    curriculum_config_from_training_config,
)
from src.learn.hierarchical_dqn_initialization import (
    HIERARCHICAL_INIT_METHOD_FLAT_TEACHER_DISTILLATION,
    hierarchical_dqn_initialization_config_from_training_config,
)
from src.learn.train_joint_torch import (
    MDP_CONTRACT_MISMATCH_MESSAGE,
    MDP_CONTRACT_VERSION,
    load_config,
    teacher_retention_config_from_training_config,
    validate_active_mdp_contract,
    validate_checkpoint_mdp_contract,
)


def _disallowed_resume_path_keys(value: object, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        disallowed: list[str] = []
        for key, nested in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if "resume" in key_text.lower() and key_text != "allow_resume_contract":
                disallowed.append(path)
            disallowed.extend(_disallowed_resume_path_keys(nested, path))
        return disallowed
    if isinstance(value, list):
        disallowed = []
        for index, nested in enumerate(value):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            disallowed.extend(_disallowed_resume_path_keys(nested, path))
        return disallowed
    return []


class CurriculumConfigTests(unittest.TestCase):
    def test_curriculum_config_file_loads_and_validates(self) -> None:
        config = load_config(Path("configs/training_joint_curriculum.json"))

        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)

        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "normal_v4")
        self.assertEqual(curriculum.seed, 42)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual([entry.name for entry in curriculum.stage_schedule], list(REAL_WORLD_CURRICULUM_STAGES))

    def test_curriculum_smoke_config_file_loads_and_is_smoke_safe(self) -> None:
        config = load_config(Path("configs/training_joint_curriculum_smoke.json"))

        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        torch_cfg = config["torch_joint_training"]
        cold_start = config["cold_start"]

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["total_timesteps"], 10_000)
        self.assertLess(shared["total_timesteps"], 10_000_000)
        self.assertEqual(shared["trace_sample_interval"], 50)
        self.assertEqual(torch_cfg["checkpoint_interval_steps"], 10_000)
        self.assertEqual(torch_cfg["metrics_interval_steps"], 128)
        self.assertEqual(cold_start["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "normal_v4")
        self.assertEqual(curriculum.seed, 42)
        self.assertEqual(curriculum.baseline_anchor_probability, 0.05)
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")

    def test_high_holding_cost_curriculum_smoke_config_file_loads_and_is_smoke_safe(self) -> None:
        normal_config = load_config(Path("configs/training_joint_curriculum_smoke.json"))
        config = load_config(Path("configs/training_joint_curriculum_smoke_high_holding_cost.json"))

        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        torch_cfg = config["torch_joint_training"]
        cold_start = config["cold_start"]

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["total_timesteps"], 10_000)
        self.assertLess(shared["total_timesteps"], 10_000_000)
        self.assertEqual(shared["trace_sample_interval"], 50)
        self.assertEqual(torch_cfg["checkpoint_interval_steps"], 10_000)
        self.assertEqual(torch_cfg["metrics_interval_steps"], 128)
        self.assertEqual(cold_start["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "high_holding_cost")
        self.assertEqual(curriculum.seed, 42)
        self.assertEqual(curriculum.baseline_anchor_probability, 0.05)
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertNotEqual(
            shared["environment_id"],
            normal_config["shared_global_parameters"]["environment_id"],
        )
        self.assertEqual(
            shared["environment_id"],
            "joint_curriculum_v4_high_holding_cost_5pl_288_smoke",
        )
        self.assertEqual(normal_config["curriculum"]["stage"], "normal_v4")

    def test_vehicle_scarcity_curriculum_smoke_config_file_loads_and_is_smoke_safe(self) -> None:
        normal_config = load_config(Path("configs/training_joint_curriculum_smoke.json"))
        holding_config = load_config(Path("configs/training_joint_curriculum_smoke_high_holding_cost.json"))
        config = load_config(Path("configs/training_joint_curriculum_smoke_vehicle_scarcity.json"))

        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        torch_cfg = config["torch_joint_training"]
        cold_start = config["cold_start"]

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["total_timesteps"], 10_000)
        self.assertLess(shared["total_timesteps"], 10_000_000)
        self.assertEqual(shared["trace_sample_interval"], 50)
        self.assertEqual(torch_cfg["checkpoint_interval_steps"], 10_000)
        self.assertEqual(torch_cfg["metrics_interval_steps"], 128)
        self.assertEqual(cold_start["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "vehicle_scarcity")
        self.assertEqual(curriculum.seed, 42)
        self.assertEqual(curriculum.baseline_anchor_probability, 0.05)
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(
            shared["environment_id"],
            "joint_curriculum_v4_vehicle_scarcity_5pl_288_smoke",
        )
        self.assertNotEqual(
            shared["environment_id"],
            normal_config["shared_global_parameters"]["environment_id"],
        )
        self.assertNotEqual(
            shared["environment_id"],
            holding_config["shared_global_parameters"]["environment_id"],
        )
        self.assertEqual(normal_config["curriculum"]["stage"], "normal_v4")
        self.assertEqual(holding_config["curriculum"]["stage"], "high_holding_cost")

    def test_route_disruption_curriculum_smoke_config_file_loads_and_is_smoke_safe(self) -> None:
        normal_config = load_config(Path("configs/training_joint_curriculum_smoke.json"))
        holding_config = load_config(Path("configs/training_joint_curriculum_smoke_high_holding_cost.json"))
        scarcity_config = load_config(Path("configs/training_joint_curriculum_smoke_vehicle_scarcity.json"))
        config = load_config(Path("configs/training_joint_curriculum_smoke_route_disruption.json"))

        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        torch_cfg = config["torch_joint_training"]
        cold_start = config["cold_start"]
        previous_environment_ids = {
            normal_config["shared_global_parameters"]["environment_id"],
            holding_config["shared_global_parameters"]["environment_id"],
            scarcity_config["shared_global_parameters"]["environment_id"],
        }

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["total_timesteps"], 10_000)
        self.assertLess(shared["total_timesteps"], 10_000_000)
        self.assertEqual(shared["trace_sample_interval"], 50)
        self.assertEqual(torch_cfg["checkpoint_interval_steps"], 10_000)
        self.assertEqual(torch_cfg["metrics_interval_steps"], 128)
        self.assertEqual(cold_start["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "route_disruption")
        self.assertEqual(curriculum.seed, 42)
        self.assertEqual(curriculum.baseline_anchor_probability, 0.05)
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(
            shared["environment_id"],
            "joint_curriculum_v4_route_disruption_5pl_288_smoke",
        )
        self.assertNotIn(shared["environment_id"], previous_environment_ids)
        self.assertEqual(normal_config["curriculum"]["stage"], "normal_v4")
        self.assertEqual(holding_config["curriculum"]["stage"], "high_holding_cost")
        self.assertEqual(scarcity_config["curriculum"]["stage"], "vehicle_scarcity")

    def test_premium_sla_curriculum_smoke_config_file_loads_and_is_smoke_safe(self) -> None:
        normal_config = load_config(Path("configs/training_joint_curriculum_smoke.json"))
        holding_config = load_config(Path("configs/training_joint_curriculum_smoke_high_holding_cost.json"))
        scarcity_config = load_config(Path("configs/training_joint_curriculum_smoke_vehicle_scarcity.json"))
        route_config = load_config(Path("configs/training_joint_curriculum_smoke_route_disruption.json"))
        config = load_config(Path("configs/training_joint_curriculum_smoke_premium_sla.json"))

        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        torch_cfg = config["torch_joint_training"]
        cold_start = config["cold_start"]
        previous_environment_ids = {
            normal_config["shared_global_parameters"]["environment_id"],
            holding_config["shared_global_parameters"]["environment_id"],
            scarcity_config["shared_global_parameters"]["environment_id"],
            route_config["shared_global_parameters"]["environment_id"],
        }

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["total_timesteps"], 10_000)
        self.assertLess(shared["total_timesteps"], 10_000_000)
        self.assertEqual(shared["trace_sample_interval"], 50)
        self.assertEqual(torch_cfg["checkpoint_interval_steps"], 10_000)
        self.assertEqual(torch_cfg["metrics_interval_steps"], 128)
        self.assertEqual(cold_start["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "premium_sla")
        self.assertEqual(curriculum.seed, 42)
        self.assertEqual(curriculum.baseline_anchor_probability, 0.05)
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(
            shared["environment_id"],
            "joint_curriculum_v4_premium_sla_5pl_288_smoke",
        )
        self.assertNotIn(shared["environment_id"], previous_environment_ids)
        self.assertEqual(normal_config["curriculum"]["stage"], "normal_v4")
        self.assertEqual(holding_config["curriculum"]["stage"], "high_holding_cost")
        self.assertEqual(scarcity_config["curriculum"]["stage"], "vehicle_scarcity")
        self.assertEqual(route_config["curriculum"]["stage"], "route_disruption")

    def test_demand_spike_curriculum_smoke_config_file_loads_and_is_smoke_safe(self) -> None:
        normal_config = load_config(Path("configs/training_joint_curriculum_smoke.json"))
        holding_config = load_config(Path("configs/training_joint_curriculum_smoke_high_holding_cost.json"))
        scarcity_config = load_config(Path("configs/training_joint_curriculum_smoke_vehicle_scarcity.json"))
        route_config = load_config(Path("configs/training_joint_curriculum_smoke_route_disruption.json"))
        premium_config = load_config(Path("configs/training_joint_curriculum_smoke_premium_sla.json"))
        config = load_config(Path("configs/training_joint_curriculum_smoke_demand_spike.json"))

        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        torch_cfg = config["torch_joint_training"]
        cold_start = config["cold_start"]
        previous_environment_ids = {
            normal_config["shared_global_parameters"]["environment_id"],
            holding_config["shared_global_parameters"]["environment_id"],
            scarcity_config["shared_global_parameters"]["environment_id"],
            route_config["shared_global_parameters"]["environment_id"],
            premium_config["shared_global_parameters"]["environment_id"],
        }

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["total_timesteps"], 10_000)
        self.assertLess(shared["total_timesteps"], 10_000_000)
        self.assertEqual(shared["trace_sample_interval"], 50)
        self.assertEqual(torch_cfg["checkpoint_interval_steps"], 10_000)
        self.assertEqual(torch_cfg["metrics_interval_steps"], 128)
        self.assertEqual(cold_start["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "demand_spike")
        self.assertEqual(curriculum.seed, 42)
        self.assertEqual(curriculum.baseline_anchor_probability, 0.05)
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(
            shared["environment_id"],
            "joint_curriculum_v4_demand_spike_5pl_288_smoke",
        )
        self.assertNotIn(shared["environment_id"], previous_environment_ids)
        self.assertEqual(normal_config["curriculum"]["stage"], "normal_v4")
        self.assertEqual(holding_config["curriculum"]["stage"], "high_holding_cost")
        self.assertEqual(scarcity_config["curriculum"]["stage"], "vehicle_scarcity")
        self.assertEqual(route_config["curriculum"]["stage"], "route_disruption")
        self.assertEqual(premium_config["curriculum"]["stage"], "premium_sla")

    def test_lead_time_delay_curriculum_smoke_config_file_loads_and_is_smoke_safe(self) -> None:
        normal_config = load_config(Path("configs/training_joint_curriculum_smoke.json"))
        holding_config = load_config(Path("configs/training_joint_curriculum_smoke_high_holding_cost.json"))
        scarcity_config = load_config(Path("configs/training_joint_curriculum_smoke_vehicle_scarcity.json"))
        route_config = load_config(Path("configs/training_joint_curriculum_smoke_route_disruption.json"))
        premium_config = load_config(Path("configs/training_joint_curriculum_smoke_premium_sla.json"))
        demand_config = load_config(Path("configs/training_joint_curriculum_smoke_demand_spike.json"))
        config = load_config(Path("configs/training_joint_curriculum_smoke_lead_time_delay.json"))

        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        torch_cfg = config["torch_joint_training"]
        cold_start = config["cold_start"]
        previous_environment_ids = {
            normal_config["shared_global_parameters"]["environment_id"],
            holding_config["shared_global_parameters"]["environment_id"],
            scarcity_config["shared_global_parameters"]["environment_id"],
            route_config["shared_global_parameters"]["environment_id"],
            premium_config["shared_global_parameters"]["environment_id"],
            demand_config["shared_global_parameters"]["environment_id"],
        }

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["total_timesteps"], 10_000)
        self.assertLess(shared["total_timesteps"], 10_000_000)
        self.assertEqual(shared["trace_sample_interval"], 50)
        self.assertEqual(torch_cfg["checkpoint_interval_steps"], 10_000)
        self.assertEqual(torch_cfg["metrics_interval_steps"], 128)
        self.assertEqual(cold_start["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "lead_time_delay")
        self.assertEqual(curriculum.seed, 42)
        self.assertEqual(curriculum.baseline_anchor_probability, 0.05)
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(
            shared["environment_id"],
            "joint_curriculum_v4_lead_time_delay_5pl_288_smoke",
        )
        self.assertNotIn(shared["environment_id"], previous_environment_ids)
        self.assertEqual(normal_config["curriculum"]["stage"], "normal_v4")
        self.assertEqual(holding_config["curriculum"]["stage"], "high_holding_cost")
        self.assertEqual(scarcity_config["curriculum"]["stage"], "vehicle_scarcity")
        self.assertEqual(route_config["curriculum"]["stage"], "route_disruption")
        self.assertEqual(premium_config["curriculum"]["stage"], "premium_sla")
        self.assertEqual(demand_config["curriculum"]["stage"], "demand_spike")

    def test_mixed_stress_curriculum_smoke_config_file_loads_and_is_smoke_safe(self) -> None:
        normal_config = load_config(Path("configs/training_joint_curriculum_smoke.json"))
        holding_config = load_config(Path("configs/training_joint_curriculum_smoke_high_holding_cost.json"))
        scarcity_config = load_config(Path("configs/training_joint_curriculum_smoke_vehicle_scarcity.json"))
        route_config = load_config(Path("configs/training_joint_curriculum_smoke_route_disruption.json"))
        premium_config = load_config(Path("configs/training_joint_curriculum_smoke_premium_sla.json"))
        demand_config = load_config(Path("configs/training_joint_curriculum_smoke_demand_spike.json"))
        lead_time_config = load_config(Path("configs/training_joint_curriculum_smoke_lead_time_delay.json"))
        config = load_config(Path("configs/training_joint_curriculum_smoke_mixed_stress.json"))

        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        torch_cfg = config["torch_joint_training"]
        cold_start = config["cold_start"]
        previous_environment_ids = {
            normal_config["shared_global_parameters"]["environment_id"],
            holding_config["shared_global_parameters"]["environment_id"],
            scarcity_config["shared_global_parameters"]["environment_id"],
            route_config["shared_global_parameters"]["environment_id"],
            premium_config["shared_global_parameters"]["environment_id"],
            demand_config["shared_global_parameters"]["environment_id"],
            lead_time_config["shared_global_parameters"]["environment_id"],
        }

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["total_timesteps"], 10_000)
        self.assertLess(shared["total_timesteps"], 10_000_000)
        self.assertEqual(shared["trace_sample_interval"], 50)
        self.assertEqual(torch_cfg["checkpoint_interval_steps"], 10_000)
        self.assertEqual(torch_cfg["metrics_interval_steps"], 128)
        self.assertEqual(cold_start["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "mixed_stress")
        self.assertEqual(curriculum.seed, 42)
        self.assertEqual(curriculum.baseline_anchor_probability, 0.05)
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(
            shared["environment_id"],
            "joint_curriculum_v4_mixed_stress_5pl_288_smoke",
        )
        self.assertNotIn(shared["environment_id"], previous_environment_ids)
        self.assertEqual(normal_config["curriculum"]["stage"], "normal_v4")
        self.assertEqual(holding_config["curriculum"]["stage"], "high_holding_cost")
        self.assertEqual(scarcity_config["curriculum"]["stage"], "vehicle_scarcity")
        self.assertEqual(route_config["curriculum"]["stage"], "route_disruption")
        self.assertEqual(premium_config["curriculum"]["stage"], "premium_sla")
        self.assertEqual(demand_config["curriculum"]["stage"], "demand_spike")
        self.assertEqual(lead_time_config["curriculum"]["stage"], "lead_time_delay")

    def test_strong_premium_and_demand_smoke_configs_load_and_are_stronger(self) -> None:
        previous_configs = [
            load_config(Path("configs/training_joint_curriculum_smoke.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_high_holding_cost.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_vehicle_scarcity.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_route_disruption.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_premium_sla.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_demand_spike.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_lead_time_delay.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_mixed_stress.json")),
        ]
        premium_config = load_config(Path("configs/training_joint_curriculum_smoke_premium_sla_strong.json"))
        demand_config = load_config(Path("configs/training_joint_curriculum_smoke_demand_spike_strong.json"))
        previous_environment_ids = {
            config["shared_global_parameters"]["environment_id"]
            for config in previous_configs
        }

        for config, stage, environment_id in (
            (
                premium_config,
                "premium_sla",
                "joint_curriculum_v4_premium_sla_strong_5pl_288_smoke",
            ),
            (
                demand_config,
                "demand_spike",
                "joint_curriculum_v4_demand_spike_strong_5pl_288_smoke",
            ),
        ):
            validate_active_mdp_contract(config)
            curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
            shared = config["shared_global_parameters"]

            self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
            self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
            self.assertEqual(shared["total_timesteps"], 10_000)
            self.assertLess(shared["total_timesteps"], 10_000_000)
            self.assertEqual(config["cold_start"]["allow_resume_contract"], MDP_CONTRACT_VERSION)
            self.assertTrue(curriculum.enabled)
            self.assertEqual(curriculum.stage, stage)
            self.assertEqual(curriculum.seed, 42)
            self.assertFalse(curriculum.randomization.enabled)
            self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
            self.assertEqual(shared["environment_id"], environment_id)
            self.assertNotIn(shared["environment_id"], previous_environment_ids)

        base_premium = CurriculumSampler(CurriculumConfig(enabled=True, stage="premium_sla")).sample(0).environment_overrides
        strong_premium = _first_non_anchor_sample(
            CurriculumSampler(curriculum_config_from_training_config(premium_config, fallback_seed=99))
        ).environment_overrides
        self.assertGreater(strong_premium["premium_sla_ratio"], base_premium["premium_sla_ratio"])
        self.assertLess(strong_premium["urgent_due_window_multiplier"], base_premium["urgent_due_window_multiplier"])
        self.assertGreater(
            strong_premium["premium_lateness_penalty_multiplier"],
            base_premium["premium_lateness_penalty_multiplier"],
        )

        base_demand = CurriculumSampler(CurriculumConfig(enabled=True, stage="demand_spike")).sample(0).environment_overrides
        strong_demand = _first_non_anchor_sample(
            CurriculumSampler(curriculum_config_from_training_config(demand_config, fallback_seed=99))
        ).environment_overrides
        self.assertGreater(
            strong_demand["rolling_demand_probability_multiplier"],
            base_demand["rolling_demand_probability_multiplier"],
        )
        self.assertGreater(
            strong_demand["rolling_demand_volume_multiplier"],
            base_demand["rolling_demand_volume_multiplier"],
        )
        self.assertGreater(strong_demand["order_units_multiplier"], base_demand["order_units_multiplier"])
        self.assertGreater(
            strong_demand["urgent_order_probability_multiplier"],
            base_demand["urgent_order_probability_multiplier"],
        )

        self.assertEqual(previous_configs[0]["curriculum"]["stage"], "normal_v4")
        self.assertEqual(previous_configs[4]["curriculum"]["stage"], "premium_sla")
        self.assertEqual(previous_configs[5]["curriculum"]["stage"], "demand_spike")

    def test_strong_demand_repair_v2_smoke_config_loads_and_is_isolated(self) -> None:
        previous_configs = [
            load_config(Path("configs/training_joint_curriculum_smoke.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_high_holding_cost.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_vehicle_scarcity.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_route_disruption.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_premium_sla.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_demand_spike.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_lead_time_delay.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_mixed_stress.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_premium_sla_strong.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_demand_spike_strong.json")),
        ]
        old_strong_demand = previous_configs[-1]
        config = load_config(Path("configs/training_joint_curriculum_smoke_demand_spike_strong_repair_v2.json"))
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        previous_environment_ids = {
            previous["shared_global_parameters"]["environment_id"]
            for previous in previous_configs
        }

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["total_timesteps"], 10_000)
        self.assertLess(shared["total_timesteps"], 10_000_000)
        self.assertEqual(config["cold_start"]["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "demand_spike")
        self.assertEqual(curriculum.seed, 42)
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(
            shared["environment_id"],
            "joint_curriculum_v4_demand_spike_strong_repair_v2_5pl_288_smoke",
        )
        self.assertNotIn(shared["environment_id"], previous_environment_ids)
        self.assertEqual(
            config["curriculum"]["explicit_overrides"],
            old_strong_demand["curriculum"]["explicit_overrides"],
        )
        self.assertEqual(
            old_strong_demand["shared_global_parameters"]["environment_id"],
            "joint_curriculum_v4_demand_spike_strong_5pl_288_smoke",
        )
        self.assertEqual(old_strong_demand["curriculum"]["stage"], "demand_spike")

    def test_strong_demand_repair_v3_smoke_config_loads_and_is_isolated(self) -> None:
        previous_configs = [
            load_config(Path("configs/training_joint_curriculum_smoke.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_high_holding_cost.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_vehicle_scarcity.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_route_disruption.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_premium_sla.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_demand_spike.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_lead_time_delay.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_mixed_stress.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_premium_sla_strong.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_demand_spike_strong.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_demand_spike_strong_repair_v2.json")),
        ]
        old_strong_demand = previous_configs[-2]
        v2_strong_demand = previous_configs[-1]
        config = load_config(Path("configs/training_joint_curriculum_smoke_demand_spike_strong_repair_v3.json"))
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        previous_environment_ids = {
            previous["shared_global_parameters"]["environment_id"]
            for previous in previous_configs
        }

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["total_timesteps"], 10_000)
        self.assertLess(shared["total_timesteps"], 10_000_000)
        self.assertEqual(config["cold_start"]["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "demand_spike")
        self.assertEqual(curriculum.seed, 42)
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(
            shared["environment_id"],
            "joint_curriculum_v4_demand_spike_strong_repair_v3_5pl_288_smoke",
        )
        self.assertNotIn(shared["environment_id"], previous_environment_ids)
        self.assertEqual(
            config["curriculum"]["explicit_overrides"],
            old_strong_demand["curriculum"]["explicit_overrides"],
        )
        self.assertEqual(
            config["curriculum"]["explicit_overrides"],
            v2_strong_demand["curriculum"]["explicit_overrides"],
        )
        self.assertEqual(
            old_strong_demand["shared_global_parameters"]["environment_id"],
            "joint_curriculum_v4_demand_spike_strong_5pl_288_smoke",
        )
        self.assertEqual(
            v2_strong_demand["shared_global_parameters"]["environment_id"],
            "joint_curriculum_v4_demand_spike_strong_repair_v2_5pl_288_smoke",
        )
        self.assertEqual(old_strong_demand["curriculum"]["stage"], "demand_spike")
        self.assertEqual(v2_strong_demand["curriculum"]["stage"], "demand_spike")

    def test_strong_demand_surge_repair_v4_smoke_config_loads_and_is_isolated(self) -> None:
        previous_configs = [
            load_config(Path("configs/training_joint_curriculum_smoke.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_high_holding_cost.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_vehicle_scarcity.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_route_disruption.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_premium_sla.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_demand_spike.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_lead_time_delay.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_mixed_stress.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_premium_sla_strong.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_demand_spike_strong.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_demand_spike_strong_repair_v2.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_demand_spike_strong_repair_v3.json")),
        ]
        old_strong_demand = previous_configs[-3]
        v2_strong_demand = previous_configs[-2]
        v3_strong_demand = previous_configs[-1]
        config = load_config(Path("configs/training_joint_curriculum_smoke_demand_spike_strong_surge_repair_v4.json"))
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        previous_environment_ids = {
            previous["shared_global_parameters"]["environment_id"]
            for previous in previous_configs
        }

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["total_timesteps"], 10_000)
        self.assertLess(shared["total_timesteps"], 10_000_000)
        self.assertEqual(config["cold_start"]["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "demand_spike")
        self.assertEqual(curriculum.seed, 42)
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(
            shared["environment_id"],
            "joint_curriculum_v4_demand_spike_strong_surge_repair_v4_5pl_288_smoke",
        )
        self.assertNotIn(shared["environment_id"], previous_environment_ids)
        self.assertEqual(
            config["curriculum"]["explicit_overrides"],
            old_strong_demand["curriculum"]["explicit_overrides"],
        )
        self.assertEqual(
            config["curriculum"]["explicit_overrides"],
            v2_strong_demand["curriculum"]["explicit_overrides"],
        )
        self.assertEqual(
            config["curriculum"]["explicit_overrides"],
            v3_strong_demand["curriculum"]["explicit_overrides"],
        )
        self.assertEqual(
            old_strong_demand["shared_global_parameters"]["environment_id"],
            "joint_curriculum_v4_demand_spike_strong_5pl_288_smoke",
        )
        self.assertEqual(
            v2_strong_demand["shared_global_parameters"]["environment_id"],
            "joint_curriculum_v4_demand_spike_strong_repair_v2_5pl_288_smoke",
        )
        self.assertEqual(
            v3_strong_demand["shared_global_parameters"]["environment_id"],
            "joint_curriculum_v4_demand_spike_strong_repair_v3_5pl_288_smoke",
        )
        self.assertEqual(old_strong_demand["curriculum"]["stage"], "demand_spike")
        self.assertEqual(v2_strong_demand["curriculum"]["stage"], "demand_spike")
        self.assertEqual(v3_strong_demand["curriculum"]["stage"], "demand_spike")

    def test_strong_demand_surge_repair_v4_50k_smoke_config_loads_and_is_isolated(self) -> None:
        previous_configs = [
            load_config(Path("configs/training_joint_curriculum_smoke.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_high_holding_cost.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_vehicle_scarcity.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_route_disruption.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_premium_sla.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_demand_spike.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_lead_time_delay.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_mixed_stress.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_premium_sla_strong.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_demand_spike_strong.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_demand_spike_strong_repair_v2.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_demand_spike_strong_repair_v3.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_demand_spike_strong_surge_repair_v4.json")),
        ]
        old_strong_demand = previous_configs[-4]
        v2_strong_demand = previous_configs[-3]
        v3_strong_demand = previous_configs[-2]
        v4_10k_strong_demand = previous_configs[-1]
        config = load_config(Path("configs/training_joint_curriculum_smoke_demand_spike_strong_surge_repair_v4_50k.json"))
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        previous_environment_ids = {
            previous["shared_global_parameters"]["environment_id"]
            for previous in previous_configs
        }

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["total_timesteps"], 50_000)
        self.assertNotEqual(shared["total_timesteps"], 10_000)
        self.assertLess(shared["total_timesteps"], 10_000_000)
        self.assertEqual(config["cold_start"]["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "demand_spike")
        self.assertEqual(curriculum.seed, 42)
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(
            shared["environment_id"],
            "joint_curriculum_v4_demand_spike_strong_surge_repair_v4_50k_smoke",
        )
        self.assertNotIn(shared["environment_id"], previous_environment_ids)
        self.assertEqual(
            config["curriculum"]["explicit_overrides"],
            old_strong_demand["curriculum"]["explicit_overrides"],
        )
        self.assertEqual(
            config["curriculum"]["explicit_overrides"],
            v2_strong_demand["curriculum"]["explicit_overrides"],
        )
        self.assertEqual(
            config["curriculum"]["explicit_overrides"],
            v3_strong_demand["curriculum"]["explicit_overrides"],
        )
        self.assertEqual(
            config["curriculum"]["explicit_overrides"],
            v4_10k_strong_demand["curriculum"]["explicit_overrides"],
        )
        self.assertEqual(
            old_strong_demand["shared_global_parameters"]["environment_id"],
            "joint_curriculum_v4_demand_spike_strong_5pl_288_smoke",
        )
        self.assertEqual(
            v2_strong_demand["shared_global_parameters"]["environment_id"],
            "joint_curriculum_v4_demand_spike_strong_repair_v2_5pl_288_smoke",
        )
        self.assertEqual(
            v3_strong_demand["shared_global_parameters"]["environment_id"],
            "joint_curriculum_v4_demand_spike_strong_repair_v3_5pl_288_smoke",
        )
        self.assertEqual(
            v4_10k_strong_demand["shared_global_parameters"]["environment_id"],
            "joint_curriculum_v4_demand_spike_strong_surge_repair_v4_5pl_288_smoke",
        )
        self.assertEqual(v4_10k_strong_demand["shared_global_parameters"]["total_timesteps"], 10_000)
        self.assertEqual(old_strong_demand["curriculum"]["stage"], "demand_spike")
        self.assertEqual(v2_strong_demand["curriculum"]["stage"], "demand_spike")
        self.assertEqual(v3_strong_demand["curriculum"]["stage"], "demand_spike")
        self.assertEqual(v4_10k_strong_demand["curriculum"]["stage"], "demand_spike")

    def test_production_curriculum_config_may_remain_production_length(self) -> None:
        config = load_config(Path("configs/training_joint_curriculum.json"))

        validate_active_mdp_contract(config)
        self.assertEqual(config["shared_global_parameters"]["total_timesteps"], 10_000_000)
        self.assertTrue(config["curriculum"]["enabled"])

    def test_clean_cold_start_1m_pilot_config_loads_and_is_isolated(self) -> None:
        production = load_config(Path("configs/training_joint_curriculum.json"))
        smoke_configs = [
            load_config(Path("configs/training_joint_curriculum_smoke.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_high_holding_cost.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_vehicle_scarcity.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_route_disruption.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_premium_sla.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_demand_spike.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_lead_time_delay.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_mixed_stress.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_premium_sla_strong.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_demand_spike_strong.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_demand_spike_strong_repair_v2.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_demand_spike_strong_repair_v3.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_demand_spike_strong_surge_repair_v4.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_demand_spike_strong_surge_repair_v4_50k.json")),
        ]
        config = load_config(Path("configs/training_joint_curriculum_clean_cold_start_1m.json"))

        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        smoke_environment_ids = {
            item["shared_global_parameters"]["environment_id"]
            for item in smoke_configs
        }

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(config["cold_start"]["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["total_timesteps"], 1_000_000)
        self.assertNotEqual(shared["total_timesteps"], 10_000)
        self.assertNotEqual(shared["total_timesteps"], 10_000_000)
        self.assertEqual(
            shared["environment_id"],
            "joint_curriculum_v4_clean_cold_start_1m_5pl_288",
        )
        self.assertNotEqual(
            shared["environment_id"],
            production["shared_global_parameters"]["environment_id"],
        )
        self.assertNotIn(shared["environment_id"], smoke_environment_ids)
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "normal_v4")
        self.assertEqual(curriculum.seed, 42)
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertNotIn("explicit_overrides", config["curriculum"])
        self.assertEqual(
            [entry.name for entry in curriculum.stage_schedule],
            list(REAL_WORLD_CURRICULUM_STAGES),
        )
        self.assertEqual(
            production["shared_global_parameters"]["total_timesteps"],
            10_000_000,
        )
        self.assertEqual(
            production["shared_global_parameters"]["environment_id"],
            "joint_curriculum_v4_rolling_5pl_288",
        )
        self.assertEqual(
            smoke_configs[0]["shared_global_parameters"]["environment_id"],
            "joint_curriculum_v4_rolling_5pl_288_smoke",
        )

        stale_v3_config = {
            "mdp_contract_version": "physical_reality_v3_dispatch_feasibility",
            "shared_global_parameters": {"observation_dim": 49},
        }
        with self.assertRaisesRegex(ValueError, "Config MDP contract mismatch"):
            validate_active_mdp_contract(stale_v3_config)

    def test_v5_clean_cold_start_1m_config_loads_and_isolated_from_prior_runs(self) -> None:
        config_path = Path("configs/training_joint_curriculum_v5_clean_cold_start_1m.json")
        self.assertTrue(config_path.exists())

        config = load_config(config_path)
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        cold_start = config["cold_start"]
        config_text = config_path.read_text(encoding="utf-8")
        prior_environment_ids = set()
        for path in Path("configs").glob("training_joint*.json"):
            if path == config_path:
                continue
            existing = load_config(path)
            existing_shared = existing.get("shared_global_parameters", {})
            if "environment_id" in existing_shared:
                prior_environment_ids.add(existing_shared["environment_id"])

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(cold_start["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertTrue(cold_start["required"])
        self.assertTrue(cold_start["reject_stale_contracts"])
        self.assertNotIn("active_checkpoint_dir", cold_start)
        self.assertNotIn("archive_dir", cold_start)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], 73)
        self.assertEqual(shared["total_timesteps"], 1_000_000)
        self.assertEqual(shared["environment_id"], "joint_curriculum_v5_clean_cold_start_1m")
        self.assertEqual(shared["experiment_name"], "joint_curriculum_v5_clean_cold_start_1m")
        self.assertNotIn(shared["environment_id"], prior_environment_ids)
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "normal_v4")
        self.assertEqual(curriculum.seed, 42)
        self.assertTrue(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "medium_single_stress")
        self.assertNotIn("explicit_overrides", config["curriculum"])
        self.assertEqual(
            [entry.name for entry in curriculum.stage_schedule],
            list(REAL_WORLD_CURRICULUM_STAGES),
        )
        self.assertEqual(_disallowed_resume_path_keys(config), [])
        self.assertNotIn("models/checkpoints", config_text)
        self.assertNotIn("models/baselines", config_text)
        self.assertNotIn("joint_torch_fresh_route_disruption_v5_candidate_visibility_50k_diagnostic", config_text)
        self.assertNotIn("joint_torch_fresh_route_disruption_v5_candidate_visibility_100k_diagnostic", config_text)
        self.assertNotIn("joint_curriculum_v5_fresh_route_disruption_candidate_visibility_50k_diagnostic", config_text)
        self.assertNotIn("joint_curriculum_v5_fresh_route_disruption_candidate_visibility_100k_diagnostic", config_text)

        stale_v4_config = {
            "mdp_contract_version": "physical_reality_v4_real_world_stress_visibility",
            "shared_global_parameters": {"observation_dim": 57},
        }
        stale_v4_checkpoint = {
            "config": stale_v4_config,
            "observation_dim": 57,
        }
        with self.assertRaisesRegex(ValueError, "Config MDP contract mismatch"):
            validate_active_mdp_contract(stale_v4_config)
        with self.assertRaisesRegex(ValueError, MDP_CONTRACT_MISMATCH_MESSAGE):
            validate_checkpoint_mdp_contract(stale_v4_checkpoint)

    def test_v5_clean_cold_start_1m_schedule_reaches_configured_stages(self) -> None:
        config = load_config(Path("configs/training_joint_curriculum_v5_clean_cold_start_1m.json"))
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        total_timesteps = int(config["shared_global_parameters"]["total_timesteps"])
        sampler = CurriculumSampler(curriculum, total_timesteps=total_timesteps)
        schedule_weight = sum(entry.steps for entry in curriculum.stage_schedule)

        sampled_by_stage = {}
        cursor = 0
        for entry in curriculum.stage_schedule:
            global_step = int(((cursor + (entry.steps / 2.0)) / schedule_weight) * total_timesteps)
            sample = _first_non_anchor_sample(sampler, global_step=global_step)
            sampled_by_stage[entry.name] = sample
            cursor += entry.steps

        self.assertEqual(list(sampled_by_stage), list(REAL_WORLD_CURRICULUM_STAGES))
        self.assertGreater(len({sample.stage for sample in sampled_by_stage.values()}), 1)
        self.assertEqual(sampled_by_stage["normal_v4"].environment_overrides, {})
        self.assertIn("route_disruption_probability", sampled_by_stage["route_disruption"].environment_overrides)
        self.assertIn("premium_sla_ratio", sampled_by_stage["premium_sla"].environment_overrides)
        self.assertIn("rolling_demand_volume_multiplier", sampled_by_stage["demand_spike"].environment_overrides)
        self.assertTrue(sampled_by_stage["mixed_stress"].environment_overrides)
        for stage, sample in sampled_by_stage.items():
            self.assertEqual(sample.stage, stage)
            self.assertEqual(sample.effective_stage, stage)

    def test_v5_route_diagnostic_remains_fixed_route_stage_with_schedule_present(self) -> None:
        config = load_config(Path("configs/training_joint_curriculum_fresh_route_disruption_v5_candidate_visibility_50k.json"))
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        total_timesteps = int(config["shared_global_parameters"]["total_timesteps"])
        sampler = CurriculumSampler(curriculum, total_timesteps=total_timesteps)

        for global_step in (0, total_timesteps // 2, total_timesteps - 1):
            with self.subTest(global_step=global_step):
                sample = _first_non_anchor_sample(sampler, global_step=global_step)
                self.assertEqual(sample.stage, "route_disruption")
                self.assertEqual(sample.effective_stage, "route_disruption")
                self.assertIn("route_disruption_probability", sample.environment_overrides)

    def test_v5_clean_cold_start_1m_after_blockers_config_is_fresh_and_scheduled(self) -> None:
        config_path = Path("configs/training_joint_curriculum_v5_clean_cold_start_1m_after_blockers.json")
        self.assertTrue(config_path.exists())

        config = load_config(config_path)
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        cold_start = config["cold_start"]
        config_text = config_path.read_text(encoding="utf-8")
        prior_environment_ids = set()
        for path in Path("configs").glob("training_joint*.json"):
            if path == config_path:
                continue
            existing = load_config(path)
            existing_shared = existing.get("shared_global_parameters", {})
            if "environment_id" in existing_shared:
                prior_environment_ids.add(existing_shared["environment_id"])

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], 73)
        self.assertEqual(DISCRETE_ACTION_COUNT, 48)
        self.assertEqual(shared["total_timesteps"], 1_000_000)
        self.assertEqual(shared["environment_id"], "joint_curriculum_v5_clean_cold_start_1m_after_blockers")
        self.assertEqual(shared["experiment_name"], "joint_curriculum_v5_clean_cold_start_1m_after_blockers")
        self.assertEqual(shared["team_id"], "joint_curriculum_v5_clean_cold_start_1m_after_blockers")
        self.assertNotIn(shared["environment_id"], prior_environment_ids)
        self.assertNotEqual(shared["environment_id"], "joint_curriculum_v5_clean_cold_start_1m")
        self.assertNotEqual(shared["environment_id"], "joint_curriculum_v4_clean_cold_start_1m_5pl_288")
        self.assertTrue(cold_start["required"])
        self.assertEqual(cold_start["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertTrue(cold_start["reject_stale_contracts"])
        self.assertNotIn("active_checkpoint_dir", cold_start)
        self.assertNotIn("archive_dir", cold_start)
        self.assertEqual(_disallowed_resume_path_keys(config), [])
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "normal_v4")
        self.assertTrue(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "medium_single_stress")
        self.assertNotIn("explicit_overrides", config["curriculum"])
        self.assertEqual([entry.name for entry in curriculum.stage_schedule], list(REAL_WORLD_CURRICULUM_STAGES))

        self.assertNotIn("models/checkpoints", config_text)
        self.assertNotIn("models/baselines", config_text)
        self.assertNotIn("joint_torch_v5_clean_cold_start_1m", config_text)
        self.assertNotIn("joint_torch_fresh_route_disruption", config_text)
        self.assertNotIn("joint_curriculum_v5_fresh_route_disruption_candidate_visibility_50k_diagnostic", config_text)
        self.assertNotIn("joint_curriculum_v5_fresh_route_disruption_candidate_visibility_100k_diagnostic", config_text)
        self.assertNotIn("joint_curriculum_v4_fresh_route_disruption", config_text)
        self.assertNotIn("700k", config_text)
        self.assertNotIn("quarantined", config_text)
        self.assertNotIn("physical_reality_v3", config_text)
        self.assertNotIn("physical_reality_v4", config_text)
        self.assertNotIn('"observation_dim": 57', config_text)

        sampler = CurriculumSampler(curriculum, total_timesteps=int(shared["total_timesteps"]))
        self.assertEqual(sampler.stage_progression, "scheduled")
        schedule_weight = sum(entry.steps for entry in curriculum.stage_schedule)
        sampled_by_stage = {}
        cursor = 0
        for entry in curriculum.stage_schedule:
            global_step = int(((cursor + (entry.steps / 2.0)) / schedule_weight) * shared["total_timesteps"])
            sampled_by_stage[entry.name] = _first_non_anchor_sample(sampler, global_step=global_step)
            cursor += entry.steps
        self.assertEqual(list(sampled_by_stage), list(REAL_WORLD_CURRICULUM_STAGES))
        self.assertGreater(len({sample.stage for sample in sampled_by_stage.values()}), 1)
        self.assertIn("holding_cost_multiplier", sampled_by_stage["high_holding_cost"].environment_overrides)
        self.assertIn("vehicle_availability_multiplier", sampled_by_stage["vehicle_scarcity"].environment_overrides)
        self.assertIn("route_disruption_probability", sampled_by_stage["route_disruption"].environment_overrides)
        self.assertIn("premium_sla_ratio", sampled_by_stage["premium_sla"].environment_overrides)
        self.assertIn("rolling_demand_volume_multiplier", sampled_by_stage["demand_spike"].environment_overrides)
        self.assertIn("supplier_delay_probability", sampled_by_stage["lead_time_delay"].environment_overrides)
        self.assertTrue(sampled_by_stage["mixed_stress"].environment_overrides)

        stale_v4_config = {
            "mdp_contract_version": "physical_reality_v4_real_world_stress_visibility",
            "shared_global_parameters": {"observation_dim": 57},
        }
        with self.assertRaisesRegex(ValueError, "Config MDP contract mismatch"):
            validate_active_mdp_contract(stale_v4_config)

    def test_v5_clean_cold_start_1m_after_rewardfix_config_is_fresh_and_scheduled(self) -> None:
        source_path = Path("configs/training_joint_curriculum_v5_clean_cold_start_1m_after_blockers.json")
        config_path = Path("configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix.json")
        source = load_config(source_path)
        expected = {
            **source,
            "shared_global_parameters": {
                **source["shared_global_parameters"],
                "experiment_name": "joint_curriculum_v5_clean_cold_start_1m_after_rewardfix",
                "environment_id": "joint_curriculum_v5_clean_cold_start_1m_after_rewardfix",
                "team_id": "joint_curriculum_v5_clean_cold_start_1m_after_rewardfix",
            },
        }

        self.assertTrue(config_path.exists())

        config = load_config(config_path)
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        cold_start = config["cold_start"]
        config_text = config_path.read_text(encoding="utf-8")
        prior_environment_ids = set()
        for path in Path("configs").glob("training_joint*.json"):
            if path == config_path:
                continue
            existing = load_config(path)
            existing_shared = existing.get("shared_global_parameters", {})
            if "environment_id" in existing_shared:
                prior_environment_ids.add(existing_shared["environment_id"])

        self.assertEqual(config, expected)
        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], 73)
        self.assertEqual(DISCRETE_ACTION_COUNT, 48)
        self.assertEqual(shared["total_timesteps"], 1_000_000)
        self.assertEqual(shared["environment_id"], "joint_curriculum_v5_clean_cold_start_1m_after_rewardfix")
        self.assertEqual(shared["experiment_name"], "joint_curriculum_v5_clean_cold_start_1m_after_rewardfix")
        self.assertEqual(shared["team_id"], "joint_curriculum_v5_clean_cold_start_1m_after_rewardfix")
        self.assertNotIn(shared["environment_id"], prior_environment_ids)
        self.assertNotEqual(shared["environment_id"], "joint_curriculum_v5_clean_cold_start_1m")
        self.assertNotEqual(shared["environment_id"], "joint_curriculum_v4_clean_cold_start_1m_5pl_288")
        self.assertTrue(cold_start["required"])
        self.assertEqual(cold_start["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertTrue(cold_start["reject_stale_contracts"])
        self.assertNotIn("active_checkpoint_dir", cold_start)
        self.assertNotIn("archive_dir", cold_start)
        self.assertEqual(_disallowed_resume_path_keys(config), [])
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "normal_v4")
        self.assertTrue(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "medium_single_stress")
        self.assertNotIn("explicit_overrides", config["curriculum"])
        self.assertEqual([entry.name for entry in curriculum.stage_schedule], list(REAL_WORLD_CURRICULUM_STAGES))

        self.assertNotIn("models/checkpoints", config_text)
        self.assertNotIn("models/baselines", config_text)
        self.assertNotIn("joint_torch_v5_clean_cold_start_1m", config_text)
        self.assertNotIn("joint_torch_fresh_route_disruption", config_text)
        self.assertNotIn("joint_curriculum_v5_fresh_route_disruption_candidate_visibility_50k_diagnostic", config_text)
        self.assertNotIn("joint_curriculum_v5_fresh_route_disruption_candidate_visibility_100k_diagnostic", config_text)
        self.assertNotIn("joint_curriculum_v4_fresh_route_disruption", config_text)
        self.assertNotIn("physical_reality_v4", config_text)
        self.assertNotIn('"observation_dim": 57', config_text)

        sampler = CurriculumSampler(curriculum, total_timesteps=int(shared["total_timesteps"]))
        self.assertEqual(sampler.stage_progression, "scheduled")
        schedule_weight = sum(entry.steps for entry in curriculum.stage_schedule)
        sampled_by_stage = {}
        cursor = 0
        for entry in curriculum.stage_schedule:
            global_step = int(((cursor + (entry.steps / 2.0)) / schedule_weight) * shared["total_timesteps"])
            sampled_by_stage[entry.name] = _first_non_anchor_sample(sampler, global_step=global_step)
            cursor += entry.steps
        self.assertEqual(list(sampled_by_stage), list(REAL_WORLD_CURRICULUM_STAGES))
        self.assertGreater(len({sample.stage for sample in sampled_by_stage.values()}), 1)
        self.assertIn("holding_cost_multiplier", sampled_by_stage["high_holding_cost"].environment_overrides)
        self.assertIn("vehicle_availability_multiplier", sampled_by_stage["vehicle_scarcity"].environment_overrides)
        self.assertIn("route_disruption_probability", sampled_by_stage["route_disruption"].environment_overrides)
        self.assertIn("premium_sla_ratio", sampled_by_stage["premium_sla"].environment_overrides)
        self.assertIn("rolling_demand_volume_multiplier", sampled_by_stage["demand_spike"].environment_overrides)
        self.assertIn("supplier_delay_probability", sampled_by_stage["lead_time_delay"].environment_overrides)
        self.assertTrue(sampled_by_stage["mixed_stress"].environment_overrides)

    def test_v5_clean_cold_start_1m_after_rewardfix_perfclean_config_is_fresh_and_scheduled(self) -> None:
        source_path = Path("configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix.json")
        config_path = Path("configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean.json")
        source = load_config(source_path)
        stale_env_id = "joint_curriculum_v5_clean_cold_start_1m_after_rewardfix"
        perfclean_env_id = "joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean"
        expected = {
            **source,
            "shared_global_parameters": {
                **source["shared_global_parameters"],
                "experiment_name": perfclean_env_id,
                "environment_id": perfclean_env_id,
                "team_id": perfclean_env_id,
            },
        }

        self.assertTrue(config_path.exists())

        config = load_config(config_path)
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        cold_start = config["cold_start"]
        config_text = config_path.read_text(encoding="utf-8")
        prior_environment_ids = set()
        for path in Path("configs").glob("training_joint*.json"):
            if path == config_path:
                continue
            existing = load_config(path)
            existing_shared = existing.get("shared_global_parameters", {})
            if "environment_id" in existing_shared:
                prior_environment_ids.add(existing_shared["environment_id"])

        self.assertEqual(config, expected)
        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], 73)
        self.assertEqual(DISCRETE_ACTION_COUNT, 48)
        self.assertEqual(shared["total_timesteps"], 1_000_000)
        self.assertEqual(shared["environment_id"], perfclean_env_id)
        self.assertEqual(shared["experiment_name"], perfclean_env_id)
        self.assertEqual(shared["team_id"], perfclean_env_id)
        self.assertNotEqual(shared["environment_id"], stale_env_id)
        self.assertNotIn(shared["environment_id"], prior_environment_ids)
        self.assertNotEqual(shared["environment_id"], "joint_curriculum_v5_clean_cold_start_1m")
        self.assertNotEqual(shared["environment_id"], "joint_curriculum_v4_clean_cold_start_1m_5pl_288")
        self.assertTrue(cold_start["required"])
        self.assertEqual(cold_start["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertTrue(cold_start["reject_stale_contracts"])
        self.assertNotIn("active_checkpoint_dir", cold_start)
        self.assertNotIn("archive_dir", cold_start)
        self.assertEqual(_disallowed_resume_path_keys(config), [])
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "normal_v4")
        self.assertTrue(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "medium_single_stress")
        self.assertNotIn("explicit_overrides", config["curriculum"])
        self.assertEqual([entry.name for entry in curriculum.stage_schedule], list(REAL_WORLD_CURRICULUM_STAGES))

        self.assertNotIn("models/checkpoints", config_text)
        self.assertNotIn("models/baselines", config_text)
        self.assertNotIn("joint_torch_v5_clean_cold_start_1m", config_text)
        self.assertNotIn("joint_torch_fresh_route_disruption", config_text)
        self.assertNotIn("joint_curriculum_v5_fresh_route_disruption_candidate_visibility_50k_diagnostic", config_text)
        self.assertNotIn("joint_curriculum_v5_fresh_route_disruption_candidate_visibility_100k_diagnostic", config_text)
        self.assertNotIn("joint_curriculum_v4_fresh_route_disruption", config_text)
        self.assertNotIn("physical_reality_v4", config_text)
        self.assertNotIn('"observation_dim": 57', config_text)

        sampler = CurriculumSampler(curriculum, total_timesteps=int(shared["total_timesteps"]))
        self.assertEqual(sampler.stage_progression, "scheduled")
        schedule_weight = sum(entry.steps for entry in curriculum.stage_schedule)
        sampled_by_stage = {}
        cursor = 0
        for entry in curriculum.stage_schedule:
            global_step = int(((cursor + (entry.steps / 2.0)) / schedule_weight) * shared["total_timesteps"])
            sampled_by_stage[entry.name] = _first_non_anchor_sample(sampler, global_step=global_step)
            cursor += entry.steps
        self.assertEqual(list(sampled_by_stage), list(REAL_WORLD_CURRICULUM_STAGES))
        self.assertGreater(len({sample.stage for sample in sampled_by_stage.values()}), 1)
        self.assertIn("holding_cost_multiplier", sampled_by_stage["high_holding_cost"].environment_overrides)
        self.assertIn("vehicle_availability_multiplier", sampled_by_stage["vehicle_scarcity"].environment_overrides)
        self.assertIn("route_disruption_probability", sampled_by_stage["route_disruption"].environment_overrides)
        self.assertIn("premium_sla_ratio", sampled_by_stage["premium_sla"].environment_overrides)
        self.assertIn("rolling_demand_volume_multiplier", sampled_by_stage["demand_spike"].environment_overrides)
        self.assertIn("supplier_delay_probability", sampled_by_stage["lead_time_delay"].environment_overrides)
        self.assertTrue(sampled_by_stage["mixed_stress"].environment_overrides)

    def test_v5_clean_cold_start_10k_after_rewardfix_perfclean_probe_config_is_fresh(self) -> None:
        source_path = Path("configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean.json")
        config_path = Path(
            "configs/training_joint_curriculum_v5_clean_cold_start_10k_after_rewardfix_perfclean_probe.json"
        )
        source = load_config(source_path)
        source_env_id = "joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean"
        probe_env_id = "joint_curriculum_v5_clean_cold_start_10k_after_rewardfix_perfclean_probe"
        expected = {
            **source,
            "shared_global_parameters": {
                **source["shared_global_parameters"],
                "experiment_name": probe_env_id,
                "total_timesteps": 10_000,
                "environment_id": probe_env_id,
                "team_id": probe_env_id,
            },
        }

        self.assertTrue(config_path.exists())

        config = load_config(config_path)
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        cold_start = config["cold_start"]
        config_text = config_path.read_text(encoding="utf-8")
        prior_environment_ids = set()
        for path in Path("configs").glob("training_joint*.json"):
            if path == config_path:
                continue
            existing = load_config(path)
            existing_shared = existing.get("shared_global_parameters", {})
            if "environment_id" in existing_shared:
                prior_environment_ids.add(existing_shared["environment_id"])

        self.assertEqual(config, expected)
        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], 73)
        self.assertEqual(DISCRETE_ACTION_COUNT, 48)
        self.assertEqual(shared["total_timesteps"], 10_000)
        self.assertEqual(shared["environment_id"], probe_env_id)
        self.assertEqual(shared["experiment_name"], probe_env_id)
        self.assertEqual(shared["team_id"], probe_env_id)
        self.assertNotEqual(shared["environment_id"], source_env_id)
        self.assertNotIn(shared["environment_id"], prior_environment_ids)
        self.assertNotEqual(shared["environment_id"], "joint_curriculum_v5_clean_cold_start_1m")
        self.assertNotEqual(shared["environment_id"], "joint_curriculum_v4_clean_cold_start_1m_5pl_288")
        self.assertTrue(cold_start["required"])
        self.assertEqual(cold_start["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertTrue(cold_start["reject_stale_contracts"])
        self.assertNotIn("active_checkpoint_dir", cold_start)
        self.assertNotIn("archive_dir", cold_start)
        self.assertEqual(_disallowed_resume_path_keys(config), [])
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "normal_v4")
        self.assertTrue(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "medium_single_stress")
        self.assertNotIn("explicit_overrides", config["curriculum"])
        self.assertEqual([entry.name for entry in curriculum.stage_schedule], list(REAL_WORLD_CURRICULUM_STAGES))

        self.assertNotIn("models/checkpoints", config_text)
        self.assertNotIn("models/baselines", config_text)
        self.assertNotIn("joint_torch_v5_clean_cold_start_1m", config_text)
        self.assertNotIn("joint_torch_fresh_route_disruption", config_text)
        self.assertNotIn("joint_curriculum_v5_fresh_route_disruption_candidate_visibility_50k_diagnostic", config_text)
        self.assertNotIn("joint_curriculum_v5_fresh_route_disruption_candidate_visibility_100k_diagnostic", config_text)
        self.assertNotIn("joint_curriculum_v4_fresh_route_disruption", config_text)
        self.assertNotIn("physical_reality_v4", config_text)
        self.assertNotIn('"observation_dim": 57', config_text)

    def test_v5_clean_cold_start_30k_after_rewardfix_perfclean_probe_config_is_fresh(self) -> None:
        source_path = Path("configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean.json")
        config_path = Path(
            "configs/training_joint_curriculum_v5_clean_cold_start_30k_after_rewardfix_perfclean_probe.json"
        )
        source = load_config(source_path)
        source_env_id = "joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean"
        ten_k_probe_env_id = "joint_curriculum_v5_clean_cold_start_10k_after_rewardfix_perfclean_probe"
        probe_env_id = "joint_curriculum_v5_clean_cold_start_30k_after_rewardfix_perfclean_probe"
        expected = {
            **source,
            "shared_global_parameters": {
                **source["shared_global_parameters"],
                "experiment_name": probe_env_id,
                "total_timesteps": 30_000,
                "environment_id": probe_env_id,
                "team_id": probe_env_id,
            },
        }

        self.assertTrue(config_path.exists())

        config = load_config(config_path)
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        cold_start = config["cold_start"]
        config_text = config_path.read_text(encoding="utf-8")
        prior_environment_ids = set()
        for path in Path("configs").glob("training_joint*.json"):
            if path == config_path:
                continue
            existing = load_config(path)
            existing_shared = existing.get("shared_global_parameters", {})
            if "environment_id" in existing_shared:
                prior_environment_ids.add(existing_shared["environment_id"])

        self.assertEqual(config, expected)
        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], 73)
        self.assertEqual(DISCRETE_ACTION_COUNT, 48)
        self.assertEqual(shared["total_timesteps"], 30_000)
        self.assertEqual(shared["environment_id"], probe_env_id)
        self.assertEqual(shared["experiment_name"], probe_env_id)
        self.assertEqual(shared["team_id"], probe_env_id)
        self.assertNotEqual(shared["environment_id"], source_env_id)
        self.assertNotEqual(shared["environment_id"], ten_k_probe_env_id)
        self.assertNotIn(shared["environment_id"], prior_environment_ids)
        self.assertNotEqual(shared["environment_id"], "joint_curriculum_v5_clean_cold_start_1m")
        self.assertNotEqual(shared["environment_id"], "joint_curriculum_v4_clean_cold_start_1m_5pl_288")
        self.assertTrue(cold_start["required"])
        self.assertEqual(cold_start["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertTrue(cold_start["reject_stale_contracts"])
        self.assertNotIn("active_checkpoint_dir", cold_start)
        self.assertNotIn("archive_dir", cold_start)
        self.assertEqual(_disallowed_resume_path_keys(config), [])
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "normal_v4")
        self.assertTrue(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "medium_single_stress")
        self.assertNotIn("explicit_overrides", config["curriculum"])
        self.assertEqual([entry.name for entry in curriculum.stage_schedule], list(REAL_WORLD_CURRICULUM_STAGES))

        self.assertNotIn("models/checkpoints", config_text)
        self.assertNotIn("models/baselines", config_text)
        self.assertNotIn("joint_torch_v5_clean_cold_start_1m", config_text)
        self.assertNotIn("joint_torch_fresh_route_disruption", config_text)
        self.assertNotIn("joint_curriculum_v5_fresh_route_disruption_candidate_visibility_50k_diagnostic", config_text)
        self.assertNotIn("joint_curriculum_v5_fresh_route_disruption_candidate_visibility_100k_diagnostic", config_text)
        self.assertNotIn("joint_curriculum_v4_fresh_route_disruption", config_text)
        self.assertNotIn("700k", config_text)
        self.assertNotIn("quarantined", config_text)
        self.assertNotIn("joint_torch_v5_clean_cold_start_1m_after_rewardfix", config_text)
        self.assertNotIn("joint_curriculum_v5_clean_cold_start_700k", config_text)
        self.assertNotIn("physical_reality_v3", config_text)
        self.assertNotIn("physical_reality_v4", config_text)
        self.assertNotIn('"observation_dim": 57', config_text)

    def test_v5_clean_cold_start_30k_after_rewardfix_perfclean_dqn_cadence_probe_configs_are_isolated(self) -> None:
        base_path = Path(
            "configs/training_joint_curriculum_v5_clean_cold_start_30k_after_rewardfix_perfclean_probe.json"
        )
        base = load_config(base_path)
        canonical = load_config(Path("configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean.json"))
        self.assertEqual(canonical["torch_joint_training"]["dqn_gradient_steps_per_rollout"], 2048)

        cases = {
            "dqn1024": 1024,
            "dqn512": 512,
        }
        seen_ids = {base["shared_global_parameters"]["environment_id"]}
        for suffix, expected_steps in cases.items():
            config_path = Path(
                f"configs/training_joint_curriculum_v5_clean_cold_start_30k_after_rewardfix_perfclean_probe_{suffix}.json"
            )
            self.assertTrue(config_path.exists())
            config = load_config(config_path)
            validate_active_mdp_contract(config)
            shared = config["shared_global_parameters"]
            torch_cfg = config["torch_joint_training"]
            config_text = config_path.read_text(encoding="utf-8")
            probe_env_id = f"joint_curriculum_v5_clean_cold_start_30k_after_rewardfix_perfclean_probe_{suffix}"
            expected = {
                **base,
                "shared_global_parameters": {
                    **base["shared_global_parameters"],
                    "experiment_name": probe_env_id,
                    "environment_id": probe_env_id,
                    "team_id": probe_env_id,
                },
                "torch_joint_training": {
                    **base["torch_joint_training"],
                    "dqn_gradient_steps_per_rollout": expected_steps,
                },
            }

            self.assertEqual(config, expected)
            self.assertEqual(shared["total_timesteps"], 30_000)
            self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
            self.assertEqual(shared["observation_dim"], 73)
            self.assertEqual(DISCRETE_ACTION_COUNT, 48)
            self.assertEqual(shared["experiment_name"], probe_env_id)
            self.assertEqual(shared["environment_id"], probe_env_id)
            self.assertEqual(shared["team_id"], probe_env_id)
            self.assertNotIn(shared["environment_id"], seen_ids)
            seen_ids.add(shared["environment_id"])
            self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
            self.assertTrue(config["cold_start"]["required"])
            self.assertEqual(config["cold_start"]["allow_resume_contract"], MDP_CONTRACT_VERSION)
            self.assertTrue(config["cold_start"]["reject_stale_contracts"])
            self.assertEqual(_disallowed_resume_path_keys(config), [])
            self.assertEqual(torch_cfg["dqn_gradient_steps_per_rollout"], expected_steps)
            self.assertEqual(torch_cfg["dqn_batch_size"], base["torch_joint_training"]["dqn_batch_size"])
            self.assertEqual(torch_cfg["rollout_steps"], base["torch_joint_training"]["rollout_steps"])
            self.assertEqual(torch_cfg["dqn_learning_starts"], base["torch_joint_training"]["dqn_learning_starts"])
            self.assertEqual(torch_cfg["dqn_target_update_interval"], base["torch_joint_training"]["dqn_target_update_interval"])
            self.assertNotIn("models/checkpoints", config_text)
            self.assertNotIn("models/baselines", config_text)
            self.assertNotIn("joint_torch_v5_clean_cold_start_1m", config_text)
            self.assertNotIn("700k", config_text)
            self.assertNotIn("quarantined", config_text)
            self.assertNotIn("physical_reality_v3", config_text)
            self.assertNotIn("physical_reality_v4", config_text)
            self.assertNotIn('"observation_dim": 57', config_text)

    def test_1m_targeted_response_check_configs_resume_for_10k_more_steps(self) -> None:
        production = load_config(Path("configs/training_joint_curriculum.json"))
        one_m = load_config(Path("configs/training_joint_curriculum_clean_cold_start_1m.json"))
        smoke_configs = [
            load_config(Path("configs/training_joint_curriculum_smoke.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_high_holding_cost.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_vehicle_scarcity.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_route_disruption.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_premium_sla.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_demand_spike.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_lead_time_delay.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_mixed_stress.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_premium_sla_strong.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_demand_spike_strong.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_demand_spike_strong_repair_v2.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_demand_spike_strong_repair_v3.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_demand_spike_strong_surge_repair_v4.json")),
            load_config(Path("configs/training_joint_curriculum_smoke_demand_spike_strong_surge_repair_v4_50k.json")),
        ]
        smoke_environment_ids = {
            item["shared_global_parameters"]["environment_id"]
            for item in smoke_configs
        }
        pre_repair_targeted_configs = [
            load_config(Path("configs/training_joint_curriculum_1m_targeted_premium_sla_10k.json")),
            load_config(Path("configs/training_joint_curriculum_1m_targeted_demand_spike_10k.json")),
            load_config(Path("configs/training_joint_curriculum_1m_targeted_route_disruption_10k.json")),
        ]
        pre_repair_targeted_environment_ids = {
            item["shared_global_parameters"]["environment_id"]
            for item in pre_repair_targeted_configs
        }
        expected = {
            "configs/training_joint_curriculum_1m_targeted_premium_sla_10k.json": (
                "premium_sla",
                "joint_curriculum_v4_1m_targeted_premium_sla_10k_check",
            ),
            "configs/training_joint_curriculum_1m_targeted_demand_spike_10k.json": (
                "demand_spike",
                "joint_curriculum_v4_1m_targeted_demand_spike_10k_check",
            ),
            "configs/training_joint_curriculum_1m_targeted_route_disruption_10k.json": (
                "route_disruption",
                "joint_curriculum_v4_1m_targeted_route_disruption_10k_check",
            ),
            "configs/training_joint_curriculum_1m_targeted_premium_sla_10k_post_repair.json": (
                "premium_sla",
                "joint_curriculum_v4_1m_targeted_premium_sla_10k_post_repair_check",
            ),
            "configs/training_joint_curriculum_1m_targeted_demand_spike_10k_post_repair.json": (
                "demand_spike",
                "joint_curriculum_v4_1m_targeted_demand_spike_10k_post_repair_check",
            ),
            "configs/training_joint_curriculum_1m_targeted_route_disruption_10k_post_repair.json": (
                "route_disruption",
                "joint_curriculum_v4_1m_targeted_route_disruption_10k_post_repair_check",
            ),
        }
        post_repair_paths = {
            "configs/training_joint_curriculum_1m_targeted_premium_sla_10k_post_repair.json",
            "configs/training_joint_curriculum_1m_targeted_demand_spike_10k_post_repair.json",
            "configs/training_joint_curriculum_1m_targeted_route_disruption_10k_post_repair.json",
        }
        seen_environment_ids: set[str] = set()

        for path, (stage, environment_id) in expected.items():
            config = load_config(Path(path))
            validate_active_mdp_contract(config)
            curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
            shared = config["shared_global_parameters"]

            self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
            self.assertEqual(config["cold_start"]["allow_resume_contract"], MDP_CONTRACT_VERSION)
            self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
            self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
            self.assertEqual(shared["total_timesteps"], 1_010_000)
            self.assertNotEqual(shared["total_timesteps"], 10_000)
            self.assertNotEqual(shared["total_timesteps"], 1_000_000)
            self.assertNotEqual(shared["total_timesteps"], 10_000_000)
            self.assertEqual(shared["environment_id"], environment_id)
            self.assertNotIn(shared["environment_id"], smoke_environment_ids)
            self.assertNotEqual(
                shared["environment_id"],
                production["shared_global_parameters"]["environment_id"],
            )
            self.assertNotEqual(
                shared["environment_id"],
                one_m["shared_global_parameters"]["environment_id"],
            )
            self.assertNotIn(shared["environment_id"], seen_environment_ids)
            seen_environment_ids.add(shared["environment_id"])
            if path in post_repair_paths:
                self.assertNotIn(shared["environment_id"], pre_repair_targeted_environment_ids)
            self.assertTrue(curriculum.enabled)
            self.assertEqual(curriculum.stage, stage)
            self.assertEqual(curriculum.seed, 42)
            self.assertFalse(curriculum.randomization.enabled)
            self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
            self.assertNotIn("explicit_overrides", config["curriculum"])

        self.assertEqual(
            production["shared_global_parameters"]["total_timesteps"],
            10_000_000,
        )
        self.assertEqual(one_m["shared_global_parameters"]["total_timesteps"], 1_000_000)
        self.assertEqual(one_m["curriculum"]["stage"], "normal_v4")

    def test_fresh_route_disruption_post_repair_50k_diagnostic_config_loads(self) -> None:
        production = load_config(Path("configs/training_joint_curriculum.json"))
        route_smoke = load_config(Path("configs/training_joint_curriculum_smoke_route_disruption.json"))
        route_targeted = load_config(Path("configs/training_joint_curriculum_1m_targeted_route_disruption_10k.json"))
        route_targeted_post_repair = load_config(
            Path("configs/training_joint_curriculum_1m_targeted_route_disruption_10k_post_repair.json")
        )
        config = load_config(Path("configs/training_joint_curriculum_fresh_route_disruption_post_repair_50k.json"))

        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        previous_environment_ids = {
            production["shared_global_parameters"]["environment_id"],
            route_smoke["shared_global_parameters"]["environment_id"],
            route_targeted["shared_global_parameters"]["environment_id"],
            route_targeted_post_repair["shared_global_parameters"]["environment_id"],
        }

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(config["cold_start"]["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["total_timesteps"], 50_000)
        self.assertNotEqual(shared["total_timesteps"], 10_000)
        self.assertNotEqual(shared["total_timesteps"], 1_010_000)
        self.assertNotEqual(shared["total_timesteps"], 10_000_000)
        self.assertEqual(
            shared["environment_id"],
            "joint_curriculum_v4_fresh_route_disruption_post_repair_50k_diagnostic",
        )
        self.assertNotIn(shared["environment_id"], previous_environment_ids)
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "route_disruption")
        self.assertEqual(curriculum.seed, 42)
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertNotIn("explicit_overrides", config["curriculum"])
        self.assertFalse(any("resume" in key.lower() for key in config))
        self.assertEqual(production["shared_global_parameters"]["total_timesteps"], 10_000_000)

    def test_fresh_route_disruption_post_balance_50k_diagnostic_config_loads(self) -> None:
        production = load_config(Path("configs/training_joint_curriculum.json"))
        one_m = load_config(Path("configs/training_joint_curriculum_clean_cold_start_1m.json"))
        route_smoke = load_config(Path("configs/training_joint_curriculum_smoke_route_disruption.json"))
        route_targeted = load_config(Path("configs/training_joint_curriculum_1m_targeted_route_disruption_10k.json"))
        route_targeted_post_repair = load_config(
            Path("configs/training_joint_curriculum_1m_targeted_route_disruption_10k_post_repair.json")
        )
        fresh_post_repair = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_repair_50k.json")
        )
        config = load_config(Path("configs/training_joint_curriculum_fresh_route_disruption_post_balance_50k.json"))

        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        previous_environment_ids = {
            production["shared_global_parameters"]["environment_id"],
            one_m["shared_global_parameters"]["environment_id"],
            route_smoke["shared_global_parameters"]["environment_id"],
            route_targeted["shared_global_parameters"]["environment_id"],
            route_targeted_post_repair["shared_global_parameters"]["environment_id"],
            fresh_post_repair["shared_global_parameters"]["environment_id"],
        }

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(config["cold_start"]["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["total_timesteps"], 50_000)
        self.assertEqual(curriculum.stage, "route_disruption")
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.seed, 42)
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(
            shared["environment_id"],
            "joint_curriculum_v4_fresh_route_disruption_post_balance_50k_diagnostic",
        )
        self.assertNotIn(shared["environment_id"], previous_environment_ids)
        self.assertNotIn("explicit_overrides", config["curriculum"])
        self.assertFalse(any("resume" in key.lower() for key in config))
        self.assertEqual(production["shared_global_parameters"]["total_timesteps"], 10_000_000)
        self.assertEqual(one_m["shared_global_parameters"]["total_timesteps"], 1_000_000)
        self.assertEqual(
            fresh_post_repair["shared_global_parameters"]["environment_id"],
            "joint_curriculum_v4_fresh_route_disruption_post_repair_50k_diagnostic",
        )

    def test_fresh_route_disruption_post_third_pass_50k_diagnostic_config_loads(self) -> None:
        production = load_config(Path("configs/training_joint_curriculum.json"))
        one_m = load_config(Path("configs/training_joint_curriculum_clean_cold_start_1m.json"))
        route_smoke = load_config(Path("configs/training_joint_curriculum_smoke_route_disruption.json"))
        route_targeted = load_config(Path("configs/training_joint_curriculum_1m_targeted_route_disruption_10k.json"))
        route_targeted_post_repair = load_config(
            Path("configs/training_joint_curriculum_1m_targeted_route_disruption_10k_post_repair.json")
        )
        fresh_post_repair = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_repair_50k.json")
        )
        fresh_post_balance = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_balance_50k.json")
        )
        config = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_third_pass_50k.json")
        )

        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        previous_environment_ids = {
            production["shared_global_parameters"]["environment_id"],
            one_m["shared_global_parameters"]["environment_id"],
            route_smoke["shared_global_parameters"]["environment_id"],
            route_targeted["shared_global_parameters"]["environment_id"],
            route_targeted_post_repair["shared_global_parameters"]["environment_id"],
            fresh_post_repair["shared_global_parameters"]["environment_id"],
            fresh_post_balance["shared_global_parameters"]["environment_id"],
        }

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(config["cold_start"]["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["total_timesteps"], 50_000)
        self.assertEqual(curriculum.stage, "route_disruption")
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.seed, 42)
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(
            shared["environment_id"],
            "joint_curriculum_v4_fresh_route_disruption_post_third_pass_50k_diagnostic",
        )
        self.assertNotIn(shared["environment_id"], previous_environment_ids)
        self.assertNotIn("explicit_overrides", config["curriculum"])
        self.assertFalse(any("resume" in key.lower() for key in config))
        self.assertEqual(production["shared_global_parameters"]["total_timesteps"], 10_000_000)
        self.assertEqual(one_m["shared_global_parameters"]["total_timesteps"], 1_000_000)
        self.assertEqual(
            fresh_post_balance["shared_global_parameters"]["environment_id"],
            "joint_curriculum_v4_fresh_route_disruption_post_balance_50k_diagnostic",
        )

    def test_fresh_route_disruption_post_fourth_pass_50k_diagnostic_config_loads(self) -> None:
        production = load_config(Path("configs/training_joint_curriculum.json"))
        one_m = load_config(Path("configs/training_joint_curriculum_clean_cold_start_1m.json"))
        route_smoke = load_config(Path("configs/training_joint_curriculum_smoke_route_disruption.json"))
        route_targeted = load_config(Path("configs/training_joint_curriculum_1m_targeted_route_disruption_10k.json"))
        route_targeted_post_repair = load_config(
            Path("configs/training_joint_curriculum_1m_targeted_route_disruption_10k_post_repair.json")
        )
        fresh_post_repair = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_repair_50k.json")
        )
        fresh_post_balance = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_balance_50k.json")
        )
        fresh_post_third_pass = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_third_pass_50k.json")
        )
        config = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_fourth_pass_50k.json")
        )

        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        previous_environment_ids = {
            production["shared_global_parameters"]["environment_id"],
            one_m["shared_global_parameters"]["environment_id"],
            route_smoke["shared_global_parameters"]["environment_id"],
            route_targeted["shared_global_parameters"]["environment_id"],
            route_targeted_post_repair["shared_global_parameters"]["environment_id"],
            fresh_post_repair["shared_global_parameters"]["environment_id"],
            fresh_post_balance["shared_global_parameters"]["environment_id"],
            fresh_post_third_pass["shared_global_parameters"]["environment_id"],
        }

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(config["cold_start"]["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["total_timesteps"], 50_000)
        self.assertEqual(curriculum.stage, "route_disruption")
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.seed, 42)
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(
            shared["environment_id"],
            "joint_curriculum_v4_fresh_route_disruption_post_fourth_pass_50k_diagnostic",
        )
        self.assertNotIn(shared["environment_id"], previous_environment_ids)
        self.assertNotIn("explicit_overrides", config["curriculum"])
        self.assertFalse(any("resume" in key.lower() for key in config))
        self.assertEqual(production["shared_global_parameters"]["total_timesteps"], 10_000_000)
        self.assertEqual(one_m["shared_global_parameters"]["total_timesteps"], 1_000_000)
        self.assertEqual(
            fresh_post_third_pass["shared_global_parameters"]["environment_id"],
            "joint_curriculum_v4_fresh_route_disruption_post_third_pass_50k_diagnostic",
        )

    def test_fresh_route_disruption_post_fifth_pass_50k_diagnostic_config_loads(self) -> None:
        production = load_config(Path("configs/training_joint_curriculum.json"))
        one_m = load_config(Path("configs/training_joint_curriculum_clean_cold_start_1m.json"))
        route_smoke = load_config(Path("configs/training_joint_curriculum_smoke_route_disruption.json"))
        route_targeted = load_config(Path("configs/training_joint_curriculum_1m_targeted_route_disruption_10k.json"))
        route_targeted_post_repair = load_config(
            Path("configs/training_joint_curriculum_1m_targeted_route_disruption_10k_post_repair.json")
        )
        fresh_post_repair = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_repair_50k.json")
        )
        fresh_post_balance = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_balance_50k.json")
        )
        fresh_post_third_pass = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_third_pass_50k.json")
        )
        fresh_post_fourth_pass = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_fourth_pass_50k.json")
        )
        config = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_fifth_pass_50k.json")
        )

        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        previous_environment_ids = {
            production["shared_global_parameters"]["environment_id"],
            one_m["shared_global_parameters"]["environment_id"],
            route_smoke["shared_global_parameters"]["environment_id"],
            route_targeted["shared_global_parameters"]["environment_id"],
            route_targeted_post_repair["shared_global_parameters"]["environment_id"],
            fresh_post_repair["shared_global_parameters"]["environment_id"],
            fresh_post_balance["shared_global_parameters"]["environment_id"],
            fresh_post_third_pass["shared_global_parameters"]["environment_id"],
            fresh_post_fourth_pass["shared_global_parameters"]["environment_id"],
        }

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(config["cold_start"]["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["total_timesteps"], 50_000)
        self.assertEqual(curriculum.stage, "route_disruption")
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.seed, 42)
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(
            shared["environment_id"],
            "joint_curriculum_v4_fresh_route_disruption_post_fifth_pass_50k_diagnostic",
        )
        self.assertNotIn(shared["environment_id"], previous_environment_ids)
        self.assertNotIn("explicit_overrides", config["curriculum"])
        self.assertFalse(any("resume" in key.lower() for key in config))
        self.assertEqual(production["shared_global_parameters"]["total_timesteps"], 10_000_000)
        self.assertEqual(one_m["shared_global_parameters"]["total_timesteps"], 1_000_000)
        self.assertEqual(
            fresh_post_fourth_pass["shared_global_parameters"]["environment_id"],
            "joint_curriculum_v4_fresh_route_disruption_post_fourth_pass_50k_diagnostic",
        )

    def test_fresh_route_disruption_post_sixth_pass_50k_diagnostic_config_loads(self) -> None:
        production = load_config(Path("configs/training_joint_curriculum.json"))
        one_m = load_config(Path("configs/training_joint_curriculum_clean_cold_start_1m.json"))
        route_smoke = load_config(Path("configs/training_joint_curriculum_smoke_route_disruption.json"))
        route_targeted = load_config(Path("configs/training_joint_curriculum_1m_targeted_route_disruption_10k.json"))
        route_targeted_post_repair = load_config(
            Path("configs/training_joint_curriculum_1m_targeted_route_disruption_10k_post_repair.json")
        )
        fresh_post_repair = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_repair_50k.json")
        )
        fresh_post_balance = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_balance_50k.json")
        )
        fresh_post_third_pass = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_third_pass_50k.json")
        )
        fresh_post_fourth_pass = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_fourth_pass_50k.json")
        )
        fresh_post_fifth_pass = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_fifth_pass_50k.json")
        )
        config_path = Path("configs/training_joint_curriculum_fresh_route_disruption_post_sixth_pass_50k.json")

        self.assertTrue(config_path.exists())
        config = load_config(config_path)
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        previous_environment_ids = {
            production["shared_global_parameters"]["environment_id"],
            one_m["shared_global_parameters"]["environment_id"],
            route_smoke["shared_global_parameters"]["environment_id"],
            route_targeted["shared_global_parameters"]["environment_id"],
            route_targeted_post_repair["shared_global_parameters"]["environment_id"],
            fresh_post_repair["shared_global_parameters"]["environment_id"],
            fresh_post_balance["shared_global_parameters"]["environment_id"],
            fresh_post_third_pass["shared_global_parameters"]["environment_id"],
            fresh_post_fourth_pass["shared_global_parameters"]["environment_id"],
            fresh_post_fifth_pass["shared_global_parameters"]["environment_id"],
        }

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(config["cold_start"]["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["total_timesteps"], 50_000)
        self.assertEqual(curriculum.stage, "route_disruption")
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.seed, 42)
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(
            shared["environment_id"],
            "joint_curriculum_v4_fresh_route_disruption_post_sixth_pass_50k_diagnostic",
        )
        self.assertNotIn(shared["environment_id"], previous_environment_ids)
        self.assertNotIn("explicit_overrides", config["curriculum"])
        self.assertEqual(_disallowed_resume_path_keys(config), [])
        self.assertEqual(production["shared_global_parameters"]["total_timesteps"], 10_000_000)
        self.assertEqual(one_m["shared_global_parameters"]["total_timesteps"], 1_000_000)
        self.assertEqual(
            fresh_post_fifth_pass["shared_global_parameters"]["environment_id"],
            "joint_curriculum_v4_fresh_route_disruption_post_fifth_pass_50k_diagnostic",
        )

    def test_fresh_route_disruption_post_seventh_pass_50k_diagnostic_config_loads(self) -> None:
        production = load_config(Path("configs/training_joint_curriculum.json"))
        one_m = load_config(Path("configs/training_joint_curriculum_clean_cold_start_1m.json"))
        route_smoke = load_config(Path("configs/training_joint_curriculum_smoke_route_disruption.json"))
        route_targeted = load_config(Path("configs/training_joint_curriculum_1m_targeted_route_disruption_10k.json"))
        route_targeted_post_repair = load_config(
            Path("configs/training_joint_curriculum_1m_targeted_route_disruption_10k_post_repair.json")
        )
        fresh_post_repair = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_repair_50k.json")
        )
        fresh_post_balance = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_balance_50k.json")
        )
        fresh_post_third_pass = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_third_pass_50k.json")
        )
        fresh_post_fourth_pass = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_fourth_pass_50k.json")
        )
        fresh_post_fifth_pass = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_fifth_pass_50k.json")
        )
        fresh_post_sixth_pass = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_sixth_pass_50k.json")
        )
        config_path = Path("configs/training_joint_curriculum_fresh_route_disruption_post_seventh_pass_50k.json")

        self.assertTrue(config_path.exists())
        config = load_config(config_path)
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        previous_environment_ids = {
            production["shared_global_parameters"]["environment_id"],
            one_m["shared_global_parameters"]["environment_id"],
            route_smoke["shared_global_parameters"]["environment_id"],
            route_targeted["shared_global_parameters"]["environment_id"],
            route_targeted_post_repair["shared_global_parameters"]["environment_id"],
            fresh_post_repair["shared_global_parameters"]["environment_id"],
            fresh_post_balance["shared_global_parameters"]["environment_id"],
            fresh_post_third_pass["shared_global_parameters"]["environment_id"],
            fresh_post_fourth_pass["shared_global_parameters"]["environment_id"],
            fresh_post_fifth_pass["shared_global_parameters"]["environment_id"],
            fresh_post_sixth_pass["shared_global_parameters"]["environment_id"],
        }

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(config["cold_start"]["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["total_timesteps"], 50_000)
        self.assertEqual(curriculum.stage, "route_disruption")
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.seed, 42)
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(
            shared["environment_id"],
            "joint_curriculum_v4_fresh_route_disruption_post_seventh_pass_50k_diagnostic",
        )
        self.assertNotIn(shared["environment_id"], previous_environment_ids)
        self.assertNotIn("explicit_overrides", config["curriculum"])
        self.assertEqual(_disallowed_resume_path_keys(config), [])
        self.assertEqual(production["shared_global_parameters"]["total_timesteps"], 10_000_000)
        self.assertEqual(one_m["shared_global_parameters"]["total_timesteps"], 1_000_000)
        self.assertEqual(
            fresh_post_sixth_pass["shared_global_parameters"]["environment_id"],
            "joint_curriculum_v4_fresh_route_disruption_post_sixth_pass_50k_diagnostic",
        )

    def test_fresh_route_disruption_post_architecture_fix_50k_diagnostic_config_loads(self) -> None:
        production = load_config(Path("configs/training_joint_curriculum.json"))
        one_m = load_config(Path("configs/training_joint_curriculum_clean_cold_start_1m.json"))
        route_smoke = load_config(Path("configs/training_joint_curriculum_smoke_route_disruption.json"))
        route_targeted = load_config(Path("configs/training_joint_curriculum_1m_targeted_route_disruption_10k.json"))
        route_targeted_post_repair = load_config(
            Path("configs/training_joint_curriculum_1m_targeted_route_disruption_10k_post_repair.json")
        )
        fresh_post_repair = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_repair_50k.json")
        )
        fresh_post_balance = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_balance_50k.json")
        )
        fresh_post_third_pass = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_third_pass_50k.json")
        )
        fresh_post_fourth_pass = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_fourth_pass_50k.json")
        )
        fresh_post_fifth_pass = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_fifth_pass_50k.json")
        )
        fresh_post_sixth_pass = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_sixth_pass_50k.json")
        )
        fresh_post_seventh_pass = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_seventh_pass_50k.json")
        )
        config_path = Path(
            "configs/training_joint_curriculum_fresh_route_disruption_post_architecture_fix_50k.json"
        )

        self.assertTrue(config_path.exists())
        config = load_config(config_path)
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        previous_environment_ids = {
            production["shared_global_parameters"]["environment_id"],
            one_m["shared_global_parameters"]["environment_id"],
            route_smoke["shared_global_parameters"]["environment_id"],
            route_targeted["shared_global_parameters"]["environment_id"],
            route_targeted_post_repair["shared_global_parameters"]["environment_id"],
            fresh_post_repair["shared_global_parameters"]["environment_id"],
            fresh_post_balance["shared_global_parameters"]["environment_id"],
            fresh_post_third_pass["shared_global_parameters"]["environment_id"],
            fresh_post_fourth_pass["shared_global_parameters"]["environment_id"],
            fresh_post_fifth_pass["shared_global_parameters"]["environment_id"],
            fresh_post_sixth_pass["shared_global_parameters"]["environment_id"],
            fresh_post_seventh_pass["shared_global_parameters"]["environment_id"],
        }

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(config["cold_start"]["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["total_timesteps"], 50_000)
        self.assertEqual(curriculum.stage, "route_disruption")
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.seed, 42)
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(
            shared["environment_id"],
            "joint_curriculum_v4_fresh_route_disruption_post_architecture_fix_50k_diagnostic",
        )
        self.assertEqual(
            shared["experiment_name"],
            "joint_curriculum_v4_fresh_route_disruption_post_architecture_fix_50k_diagnostic",
        )
        self.assertEqual(
            shared["team_id"],
            "joint_curriculum_v4_fresh_route_post_architecture_fix_diagnostic",
        )
        self.assertNotIn(shared["environment_id"], previous_environment_ids)
        self.assertNotIn("explicit_overrides", config["curriculum"])
        self.assertEqual(_disallowed_resume_path_keys(config), [])
        self.assertEqual(production["shared_global_parameters"]["total_timesteps"], 10_000_000)
        self.assertEqual(one_m["shared_global_parameters"]["total_timesteps"], 1_000_000)
        self.assertEqual(
            fresh_post_seventh_pass["shared_global_parameters"]["environment_id"],
            "joint_curriculum_v4_fresh_route_disruption_post_seventh_pass_50k_diagnostic",
        )

    def test_fresh_route_disruption_post_candidate_alignment_fix_50k_diagnostic_config_loads(self) -> None:
        production = load_config(Path("configs/training_joint_curriculum.json"))
        one_m = load_config(Path("configs/training_joint_curriculum_clean_cold_start_1m.json"))
        route_smoke = load_config(Path("configs/training_joint_curriculum_smoke_route_disruption.json"))
        route_targeted = load_config(Path("configs/training_joint_curriculum_1m_targeted_route_disruption_10k.json"))
        route_targeted_post_repair = load_config(
            Path("configs/training_joint_curriculum_1m_targeted_route_disruption_10k_post_repair.json")
        )
        fresh_post_repair = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_repair_50k.json")
        )
        fresh_post_balance = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_balance_50k.json")
        )
        fresh_post_third_pass = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_third_pass_50k.json")
        )
        fresh_post_fourth_pass = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_fourth_pass_50k.json")
        )
        fresh_post_fifth_pass = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_fifth_pass_50k.json")
        )
        fresh_post_sixth_pass = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_sixth_pass_50k.json")
        )
        fresh_post_seventh_pass = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_seventh_pass_50k.json")
        )
        fresh_post_architecture_fix = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_architecture_fix_50k.json")
        )
        config_path = Path(
            "configs/training_joint_curriculum_fresh_route_disruption_post_candidate_alignment_fix_50k.json"
        )

        self.assertTrue(config_path.exists())
        config = load_config(config_path)
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        previous_environment_ids = {
            production["shared_global_parameters"]["environment_id"],
            one_m["shared_global_parameters"]["environment_id"],
            route_smoke["shared_global_parameters"]["environment_id"],
            route_targeted["shared_global_parameters"]["environment_id"],
            route_targeted_post_repair["shared_global_parameters"]["environment_id"],
            fresh_post_repair["shared_global_parameters"]["environment_id"],
            fresh_post_balance["shared_global_parameters"]["environment_id"],
            fresh_post_third_pass["shared_global_parameters"]["environment_id"],
            fresh_post_fourth_pass["shared_global_parameters"]["environment_id"],
            fresh_post_fifth_pass["shared_global_parameters"]["environment_id"],
            fresh_post_sixth_pass["shared_global_parameters"]["environment_id"],
            fresh_post_seventh_pass["shared_global_parameters"]["environment_id"],
            fresh_post_architecture_fix["shared_global_parameters"]["environment_id"],
        }

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(config["cold_start"]["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["total_timesteps"], 50_000)
        self.assertEqual(curriculum.stage, "route_disruption")
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.seed, 42)
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(
            shared["environment_id"],
            "joint_curriculum_v4_fresh_route_disruption_post_candidate_alignment_fix_50k_diagnostic",
        )
        self.assertEqual(
            shared["experiment_name"],
            "joint_curriculum_v4_fresh_route_disruption_post_candidate_alignment_fix_50k_diagnostic",
        )
        self.assertEqual(
            shared["team_id"],
            "joint_curriculum_v4_fresh_route_post_candidate_alignment_fix_diagnostic",
        )
        self.assertNotIn(shared["environment_id"], previous_environment_ids)
        self.assertNotIn("explicit_overrides", config["curriculum"])
        self.assertEqual(_disallowed_resume_path_keys(config), [])
        self.assertEqual(production["shared_global_parameters"]["total_timesteps"], 10_000_000)
        self.assertEqual(one_m["shared_global_parameters"]["total_timesteps"], 1_000_000)
        self.assertEqual(
            fresh_post_architecture_fix["shared_global_parameters"]["environment_id"],
            "joint_curriculum_v4_fresh_route_disruption_post_architecture_fix_50k_diagnostic",
        )

    def test_fresh_route_disruption_post_hold_neutralization_50k_diagnostic_config_loads(self) -> None:
        production = load_config(Path("configs/training_joint_curriculum.json"))
        one_m = load_config(Path("configs/training_joint_curriculum_clean_cold_start_1m.json"))
        route_smoke = load_config(Path("configs/training_joint_curriculum_smoke_route_disruption.json"))
        route_targeted = load_config(Path("configs/training_joint_curriculum_1m_targeted_route_disruption_10k.json"))
        route_targeted_post_repair = load_config(
            Path("configs/training_joint_curriculum_1m_targeted_route_disruption_10k_post_repair.json")
        )
        fresh_post_repair = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_repair_50k.json")
        )
        fresh_post_balance = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_balance_50k.json")
        )
        fresh_post_third_pass = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_third_pass_50k.json")
        )
        fresh_post_fourth_pass = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_fourth_pass_50k.json")
        )
        fresh_post_fifth_pass = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_fifth_pass_50k.json")
        )
        fresh_post_sixth_pass = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_sixth_pass_50k.json")
        )
        fresh_post_seventh_pass = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_seventh_pass_50k.json")
        )
        fresh_post_architecture_fix = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_architecture_fix_50k.json")
        )
        fresh_post_candidate_alignment_fix = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_candidate_alignment_fix_50k.json")
        )
        config_path = Path(
            "configs/training_joint_curriculum_fresh_route_disruption_post_hold_neutralization_50k.json"
        )

        self.assertTrue(config_path.exists())
        config = load_config(config_path)
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        previous_environment_ids = {
            production["shared_global_parameters"]["environment_id"],
            one_m["shared_global_parameters"]["environment_id"],
            route_smoke["shared_global_parameters"]["environment_id"],
            route_targeted["shared_global_parameters"]["environment_id"],
            route_targeted_post_repair["shared_global_parameters"]["environment_id"],
            fresh_post_repair["shared_global_parameters"]["environment_id"],
            fresh_post_balance["shared_global_parameters"]["environment_id"],
            fresh_post_third_pass["shared_global_parameters"]["environment_id"],
            fresh_post_fourth_pass["shared_global_parameters"]["environment_id"],
            fresh_post_fifth_pass["shared_global_parameters"]["environment_id"],
            fresh_post_sixth_pass["shared_global_parameters"]["environment_id"],
            fresh_post_seventh_pass["shared_global_parameters"]["environment_id"],
            fresh_post_architecture_fix["shared_global_parameters"]["environment_id"],
            fresh_post_candidate_alignment_fix["shared_global_parameters"]["environment_id"],
        }

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(config["cold_start"]["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["total_timesteps"], 50_000)
        self.assertEqual(curriculum.stage, "route_disruption")
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.seed, 42)
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(
            shared["environment_id"],
            "joint_curriculum_v4_fresh_route_disruption_post_hold_neutralization_50k_diagnostic",
        )
        self.assertEqual(
            shared["experiment_name"],
            "joint_curriculum_v4_fresh_route_disruption_post_hold_neutralization_50k_diagnostic",
        )
        self.assertEqual(
            shared["team_id"],
            "joint_curriculum_v4_fresh_route_post_hold_neutralization_diagnostic",
        )
        self.assertNotIn(shared["environment_id"], previous_environment_ids)
        self.assertNotIn("explicit_overrides", config["curriculum"])
        self.assertEqual(_disallowed_resume_path_keys(config), [])
        self.assertEqual(production["shared_global_parameters"]["total_timesteps"], 10_000_000)
        self.assertEqual(one_m["shared_global_parameters"]["total_timesteps"], 1_000_000)
        self.assertEqual(
            fresh_post_candidate_alignment_fix["shared_global_parameters"]["environment_id"],
            "joint_curriculum_v4_fresh_route_disruption_post_candidate_alignment_fix_50k_diagnostic",
        )

    def test_fresh_route_disruption_post_shortest_dispatch_context_50k_diagnostic_config_loads(self) -> None:
        production = load_config(Path("configs/training_joint_curriculum.json"))
        one_m = load_config(Path("configs/training_joint_curriculum_clean_cold_start_1m.json"))
        route_smoke = load_config(Path("configs/training_joint_curriculum_smoke_route_disruption.json"))
        route_targeted = load_config(Path("configs/training_joint_curriculum_1m_targeted_route_disruption_10k.json"))
        route_targeted_post_repair = load_config(
            Path("configs/training_joint_curriculum_1m_targeted_route_disruption_10k_post_repair.json")
        )
        fresh_post_repair = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_repair_50k.json")
        )
        fresh_post_balance = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_balance_50k.json")
        )
        fresh_post_third_pass = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_third_pass_50k.json")
        )
        fresh_post_fourth_pass = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_fourth_pass_50k.json")
        )
        fresh_post_fifth_pass = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_fifth_pass_50k.json")
        )
        fresh_post_sixth_pass = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_sixth_pass_50k.json")
        )
        fresh_post_seventh_pass = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_seventh_pass_50k.json")
        )
        fresh_post_architecture_fix = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_architecture_fix_50k.json")
        )
        fresh_post_candidate_alignment_fix = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_candidate_alignment_fix_50k.json")
        )
        fresh_post_hold_neutralization = load_config(
            Path("configs/training_joint_curriculum_fresh_route_disruption_post_hold_neutralization_50k.json")
        )
        config_path = Path(
            "configs/training_joint_curriculum_fresh_route_disruption_post_shortest_dispatch_context_50k.json"
        )

        self.assertTrue(config_path.exists())
        config = load_config(config_path)
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        previous_environment_ids = {
            production["shared_global_parameters"]["environment_id"],
            one_m["shared_global_parameters"]["environment_id"],
            route_smoke["shared_global_parameters"]["environment_id"],
            route_targeted["shared_global_parameters"]["environment_id"],
            route_targeted_post_repair["shared_global_parameters"]["environment_id"],
            fresh_post_repair["shared_global_parameters"]["environment_id"],
            fresh_post_balance["shared_global_parameters"]["environment_id"],
            fresh_post_third_pass["shared_global_parameters"]["environment_id"],
            fresh_post_fourth_pass["shared_global_parameters"]["environment_id"],
            fresh_post_fifth_pass["shared_global_parameters"]["environment_id"],
            fresh_post_sixth_pass["shared_global_parameters"]["environment_id"],
            fresh_post_seventh_pass["shared_global_parameters"]["environment_id"],
            fresh_post_architecture_fix["shared_global_parameters"]["environment_id"],
            fresh_post_candidate_alignment_fix["shared_global_parameters"]["environment_id"],
            fresh_post_hold_neutralization["shared_global_parameters"]["environment_id"],
        }

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(config["cold_start"]["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["total_timesteps"], 50_000)
        self.assertEqual(curriculum.stage, "route_disruption")
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.seed, 42)
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(
            shared["environment_id"],
            "joint_curriculum_v4_fresh_route_disruption_post_shortest_dispatch_context_50k_diagnostic",
        )
        self.assertEqual(
            shared["experiment_name"],
            "joint_curriculum_v4_fresh_route_disruption_post_shortest_dispatch_context_50k_diagnostic",
        )
        self.assertEqual(
            shared["team_id"],
            "joint_curriculum_v4_fresh_route_post_shortest_dispatch_context_diagnostic",
        )
        self.assertNotIn(shared["environment_id"], previous_environment_ids)
        self.assertNotIn("explicit_overrides", config["curriculum"])
        self.assertEqual(_disallowed_resume_path_keys(config), [])
        self.assertEqual(production["shared_global_parameters"]["total_timesteps"], 10_000_000)
        self.assertEqual(
            production["shared_global_parameters"]["environment_id"],
            "joint_curriculum_v4_rolling_5pl_288",
        )
        self.assertEqual(one_m["shared_global_parameters"]["total_timesteps"], 1_000_000)
        self.assertEqual(
            one_m["shared_global_parameters"]["environment_id"],
            "joint_curriculum_v4_clean_cold_start_1m_5pl_288",
        )
        self.assertEqual(
            fresh_post_hold_neutralization["shared_global_parameters"]["environment_id"],
            "joint_curriculum_v4_fresh_route_disruption_post_hold_neutralization_50k_diagnostic",
        )

    def test_fresh_route_disruption_v5_candidate_visibility_50k_diagnostic_config_loads(self) -> None:
        production = load_config(Path("configs/training_joint_curriculum.json"))
        one_m = load_config(Path("configs/training_joint_curriculum_clean_cold_start_1m.json"))
        route_smoke = load_config(Path("configs/training_joint_curriculum_smoke_route_disruption.json"))
        route_targeted = load_config(Path("configs/training_joint_curriculum_1m_targeted_route_disruption_10k.json"))
        route_targeted_post_repair = load_config(
            Path("configs/training_joint_curriculum_1m_targeted_route_disruption_10k_post_repair.json")
        )
        prior_route_diagnostic_paths = (
            "configs/training_joint_curriculum_fresh_route_disruption_post_repair_50k.json",
            "configs/training_joint_curriculum_fresh_route_disruption_post_balance_50k.json",
            "configs/training_joint_curriculum_fresh_route_disruption_post_third_pass_50k.json",
            "configs/training_joint_curriculum_fresh_route_disruption_post_fourth_pass_50k.json",
            "configs/training_joint_curriculum_fresh_route_disruption_post_fifth_pass_50k.json",
            "configs/training_joint_curriculum_fresh_route_disruption_post_sixth_pass_50k.json",
            "configs/training_joint_curriculum_fresh_route_disruption_post_seventh_pass_50k.json",
            "configs/training_joint_curriculum_fresh_route_disruption_post_architecture_fix_50k.json",
            "configs/training_joint_curriculum_fresh_route_disruption_post_candidate_alignment_fix_50k.json",
            "configs/training_joint_curriculum_fresh_route_disruption_post_hold_neutralization_50k.json",
            "configs/training_joint_curriculum_fresh_route_disruption_post_shortest_dispatch_context_50k.json",
        )
        prior_route_diagnostics = [load_config(Path(path)) for path in prior_route_diagnostic_paths]
        config_path = Path(
            "configs/training_joint_curriculum_fresh_route_disruption_v5_candidate_visibility_50k.json"
        )

        self.assertTrue(config_path.exists())
        config = load_config(config_path)
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        prior_environment_ids = {
            production["shared_global_parameters"]["environment_id"],
            one_m["shared_global_parameters"]["environment_id"],
            route_smoke["shared_global_parameters"]["environment_id"],
            route_targeted["shared_global_parameters"]["environment_id"],
            route_targeted_post_repair["shared_global_parameters"]["environment_id"],
            *(item["shared_global_parameters"]["environment_id"] for item in prior_route_diagnostics),
        }

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(config["cold_start"]["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], 73)
        self.assertEqual(shared["total_timesteps"], 50_000)
        self.assertEqual(curriculum.stage, "route_disruption")
        self.assertTrue(curriculum.enabled)
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(
            shared["environment_id"],
            "joint_curriculum_v5_fresh_route_disruption_candidate_visibility_50k_diagnostic",
        )
        self.assertEqual(
            shared["experiment_name"],
            "joint_curriculum_v5_fresh_route_disruption_candidate_visibility_50k_diagnostic",
        )
        self.assertNotIn(shared["environment_id"], prior_environment_ids)
        self.assertNotIn("explicit_overrides", config["curriculum"])
        self.assertEqual(_disallowed_resume_path_keys(config), [])
        self.assertEqual(production["shared_global_parameters"]["total_timesteps"], 10_000_000)
        self.assertEqual(one_m["shared_global_parameters"]["total_timesteps"], 1_000_000)

        stale_v4_config = {
            "mdp_contract_version": "physical_reality_v4_real_world_stress_visibility",
            "shared_global_parameters": {"observation_dim": 57},
        }
        stale_v4_checkpoint = {
            "config": stale_v4_config,
            "observation_dim": 57,
        }
        with self.assertRaisesRegex(ValueError, "Config MDP contract mismatch"):
            validate_active_mdp_contract(stale_v4_config)
        with self.assertRaisesRegex(ValueError, MDP_CONTRACT_MISMATCH_MESSAGE):
            validate_checkpoint_mdp_contract(stale_v4_checkpoint)

    def test_fresh_route_disruption_v5_candidate_visibility_100k_diagnostic_config_loads(self) -> None:
        production = load_config(Path("configs/training_joint_curriculum.json"))
        one_m = load_config(Path("configs/training_joint_curriculum_clean_cold_start_1m.json"))
        route_smoke = load_config(Path("configs/training_joint_curriculum_smoke_route_disruption.json"))
        route_targeted = load_config(Path("configs/training_joint_curriculum_1m_targeted_route_disruption_10k.json"))
        route_targeted_post_repair = load_config(
            Path("configs/training_joint_curriculum_1m_targeted_route_disruption_10k_post_repair.json")
        )
        prior_route_diagnostic_paths = (
            "configs/training_joint_curriculum_fresh_route_disruption_post_repair_50k.json",
            "configs/training_joint_curriculum_fresh_route_disruption_post_balance_50k.json",
            "configs/training_joint_curriculum_fresh_route_disruption_post_third_pass_50k.json",
            "configs/training_joint_curriculum_fresh_route_disruption_post_fourth_pass_50k.json",
            "configs/training_joint_curriculum_fresh_route_disruption_post_fifth_pass_50k.json",
            "configs/training_joint_curriculum_fresh_route_disruption_post_sixth_pass_50k.json",
            "configs/training_joint_curriculum_fresh_route_disruption_post_seventh_pass_50k.json",
            "configs/training_joint_curriculum_fresh_route_disruption_post_architecture_fix_50k.json",
            "configs/training_joint_curriculum_fresh_route_disruption_post_candidate_alignment_fix_50k.json",
            "configs/training_joint_curriculum_fresh_route_disruption_post_hold_neutralization_50k.json",
            "configs/training_joint_curriculum_fresh_route_disruption_post_shortest_dispatch_context_50k.json",
            "configs/training_joint_curriculum_fresh_route_disruption_v5_candidate_visibility_50k.json",
        )
        prior_route_diagnostics = [load_config(Path(path)) for path in prior_route_diagnostic_paths]
        config_path = Path(
            "configs/training_joint_curriculum_fresh_route_disruption_v5_candidate_visibility_100k.json"
        )

        self.assertTrue(config_path.exists())
        config = load_config(config_path)
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        config_text = config_path.read_text(encoding="utf-8")
        prior_environment_ids = {
            production["shared_global_parameters"]["environment_id"],
            one_m["shared_global_parameters"]["environment_id"],
            route_smoke["shared_global_parameters"]["environment_id"],
            route_targeted["shared_global_parameters"]["environment_id"],
            route_targeted_post_repair["shared_global_parameters"]["environment_id"],
            *(item["shared_global_parameters"]["environment_id"] for item in prior_route_diagnostics),
        }

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(config["cold_start"]["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], 73)
        self.assertEqual(shared["total_timesteps"], 100_000)
        self.assertEqual(curriculum.stage, "route_disruption")
        self.assertTrue(curriculum.enabled)
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(
            shared["environment_id"],
            "joint_curriculum_v5_fresh_route_disruption_candidate_visibility_100k_diagnostic",
        )
        self.assertEqual(
            shared["experiment_name"],
            "joint_curriculum_v5_fresh_route_disruption_candidate_visibility_100k_diagnostic",
        )
        self.assertNotIn(shared["environment_id"], prior_environment_ids)
        self.assertNotIn("explicit_overrides", config["curriculum"])
        self.assertEqual(_disallowed_resume_path_keys(config), [])
        self.assertNotIn("joint_torch_fresh_route_disruption_v5_candidate_visibility_50k_diagnostic", config_text)
        self.assertEqual(production["shared_global_parameters"]["total_timesteps"], 10_000_000)
        self.assertEqual(one_m["shared_global_parameters"]["total_timesteps"], 1_000_000)

        stale_v4_config = {
            "mdp_contract_version": "physical_reality_v4_real_world_stress_visibility",
            "shared_global_parameters": {"observation_dim": 57},
        }
        stale_v4_checkpoint = {
            "config": stale_v4_config,
            "observation_dim": 57,
        }
        with self.assertRaisesRegex(ValueError, "Config MDP contract mismatch"):
            validate_active_mdp_contract(stale_v4_config)
        with self.assertRaisesRegex(ValueError, MDP_CONTRACT_MISMATCH_MESSAGE):
            validate_checkpoint_mdp_contract(stale_v4_checkpoint)

    def test_routepremium_mixed_ft_200k_config_loads_and_is_init_safe(self) -> None:
        canonical = load_config(
            Path("configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean.json")
        )
        config_path = Path(
            "configs/training_joint_curriculum_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft_200k.json"
        )

        self.assertTrue(config_path.exists())
        config = load_config(config_path)
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        torch_cfg = config["torch_joint_training"]
        fine_tune = config["fine_tune_initialization"]
        run_safety = config["manual_run_safety"]
        config_text = config_path.read_text(encoding="utf-8")

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], 73)
        self.assertEqual(DISCRETE_ACTION_COUNT, 48)
        self.assertEqual(shared["total_timesteps"], 200_000)
        self.assertEqual(
            shared["experiment_name"],
            "joint_curriculum_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft_200k",
        )
        self.assertEqual(
            shared["environment_id"],
            "joint_curriculum_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft_200k",
        )
        self.assertEqual(
            shared["team_id"],
            "joint_curriculum_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft_200k",
        )
        self.assertFalse(config["cold_start"]["required"])
        self.assertTrue(config["cold_start"]["fine_tune_initialization_required"])
        self.assertEqual(fine_tune["mode"], "init_from_joint_checkpoint")
        self.assertEqual(
            fine_tune["source_checkpoint"],
            "models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608/joint_torch_latest.pt",
        )
        self.assertEqual(fine_tune["source_global_step"], 1_000_000)
        self.assertEqual(
            fine_tune["source_environment_id"],
            "joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean",
        )
        self.assertEqual(fine_tune["source_contract"], MDP_CONTRACT_VERSION)
        self.assertTrue(run_safety["requires_skip_registry"])
        self.assertTrue(run_safety["requires_disable_trace_logging"])
        self.assertEqual(run_safety["torch_num_threads"], 2)
        self.assertEqual(run_safety["torch_num_interop_threads"], 1)
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "normal_v4")
        self.assertEqual(
            [(entry.name, entry.steps) for entry in curriculum.stage_schedule],
            [
                ("normal_v4", 10_000),
                ("route_disruption", 40_000),
                ("premium_sla", 40_000),
                ("mixed_stress", 30_000),
                ("route_disruption", 40_000),
                ("premium_sla", 40_000),
            ],
        )
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(curriculum.baseline_anchor_probability, 0.0)
        self.assertEqual(torch_cfg["dqn_gradient_steps_per_rollout"], canonical["torch_joint_training"]["dqn_gradient_steps_per_rollout"])
        self.assertEqual(torch_cfg["dqn_batch_size"], canonical["torch_joint_training"]["dqn_batch_size"])
        self.assertEqual(config["reward_physics"], canonical["reward_physics"])
        self.assertEqual(canonical["shared_global_parameters"]["total_timesteps"], 1_000_000)
        self.assertEqual(
            canonical["shared_global_parameters"]["environment_id"],
            "joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean",
        )
        self.assertNotIn("joint_torch_v5_clean_cold_start_1m_after_rewardfix/", config_text)
        self.assertNotIn("joint_torch_v5_clean_cold_start_1m/", config_text)
        self.assertNotIn("joint_curriculum_v4", config_text)
        self.assertNotIn("physical_reality_v4", config_text)
        self.assertNotIn("physical_reality_v3", config_text)
        self.assertNotIn("700k", config_text.lower())
        self.assertNotIn("dqn1024", config_text.lower())
        self.assertNotIn("dqn512", config_text.lower())

    def test_balanced_retention_ft_200k_config_loads_and_is_init_safe(self) -> None:
        canonical = load_config(
            Path("configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean.json")
        )
        prior_ft = load_config(
            Path("configs/training_joint_curriculum_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft_200k.json")
        )
        config_path = Path(
            "configs/training_joint_curriculum_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k.json"
        )

        self.assertTrue(config_path.exists())
        config = load_config(config_path)
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        fine_tune = config["fine_tune_initialization"]
        run_safety = config["manual_run_safety"]
        config_text = config_path.read_text(encoding="utf-8")
        identity = "joint_curriculum_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k"
        clean_parent_checkpoint = (
            "models/checkpoints/"
            "joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608/"
            "joint_torch_latest.pt"
        )
        failed_ft_checkpoint = (
            "models/checkpoints/"
            "joint_torch_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft_200k_20260608/"
            "joint_torch_latest.pt"
        )

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], 73)
        self.assertEqual(DISCRETE_ACTION_COUNT, 48)
        self.assertEqual(shared["total_timesteps"], 200_000)
        self.assertEqual(shared["experiment_name"], identity)
        self.assertEqual(shared["environment_id"], identity)
        self.assertEqual(shared["team_id"], identity)
        self.assertNotEqual(identity, canonical["shared_global_parameters"]["environment_id"])
        self.assertNotEqual(identity, prior_ft["shared_global_parameters"]["environment_id"])

        self.assertFalse(config["cold_start"]["required"])
        self.assertTrue(config["cold_start"]["fine_tune_initialization_required"])
        self.assertEqual(fine_tune["mode"], "init_from_joint_checkpoint")
        self.assertEqual(fine_tune["source_checkpoint"], clean_parent_checkpoint)
        self.assertEqual(fine_tune["source_global_step"], 1_000_000)
        self.assertEqual(
            fine_tune["source_environment_id"],
            "joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean",
        )
        self.assertEqual(fine_tune["source_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(fine_tune["source_observation_dim"], 73)
        self.assertEqual(fine_tune["source_discrete_action_count"], 48)

        self.assertTrue(run_safety["requires_skip_registry"])
        self.assertTrue(run_safety["requires_disable_trace_logging"])
        self.assertEqual(run_safety["torch_num_threads"], 2)
        self.assertEqual(run_safety["torch_num_interop_threads"], 1)
        self.assertIn("balanced_retention_ft_200k_20260608", run_safety["output_dir"])
        self.assertNotIn("routepremium_mixed_ft_200k_20260608", run_safety["output_dir"])

        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "normal_v4")
        self.assertEqual(
            [(entry.name, entry.steps) for entry in curriculum.stage_schedule],
            [
                ("normal_v4", 20_000),
                ("route_disruption", 25_000),
                ("premium_sla", 25_000),
                ("lead_time_delay", 25_000),
                ("high_holding_cost", 25_000),
                ("mixed_stress", 20_000),
                ("demand_spike", 10_000),
                ("vehicle_scarcity", 10_000),
                ("route_disruption", 15_000),
                ("premium_sla", 15_000),
                ("normal_v4", 10_000),
            ],
        )
        self.assertEqual(sum(entry.steps for entry in curriculum.stage_schedule), shared["total_timesteps"])
        self.assertEqual(curriculum.stage_schedule[-1].name, "normal_v4")
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(curriculum.baseline_anchor_probability, 0.0)

        self.assertEqual(config["ppo_hyperparameters"], canonical["ppo_hyperparameters"])
        self.assertEqual(config["dqn_hyperparameters"], canonical["dqn_hyperparameters"])
        self.assertEqual(config["torch_joint_training"], canonical["torch_joint_training"])
        self.assertEqual(config["reward_physics"], canonical["reward_physics"])
        self.assertEqual(canonical["shared_global_parameters"]["total_timesteps"], 1_000_000)
        self.assertEqual(
            canonical["shared_global_parameters"]["environment_id"],
            "joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean",
        )

        self.assertNotIn(failed_ft_checkpoint, config_text)
        self.assertNotIn("joint_torch_v5_clean_cold_start_1m_after_rewardfix/joint_torch_latest.pt", config_text)
        self.assertNotIn("joint_torch_v5_clean_cold_start_1m/joint_torch_latest.pt", config_text)
        self.assertNotIn("joint_curriculum_v4", config_text)
        self.assertNotIn("physical_reality_v4", config_text)
        self.assertNotIn("physical_reality_v3", config_text)
        self.assertNotIn("700k", config_text.lower())
        self.assertNotIn("dqn1024", config_text.lower())
        self.assertNotIn("dqn512", config_text.lower())

    def test_prod_balanced_retention_nextgen_1m_config_loads_and_is_init_safe(self) -> None:
        canonical = load_config(
            Path("configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean.json")
        )
        balanced_retention = load_config(
            Path("configs/training_joint_curriculum_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k.json")
        )
        config_path = Path("configs/training_joint_curriculum_v5_prod_balanced_retention_nextgen_1m_20260609.json")

        self.assertTrue(config_path.exists())
        config = load_config(config_path)
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        fine_tune = config["fine_tune_initialization"]
        run_safety = config["manual_run_safety"]
        config_text = config_path.read_text(encoding="utf-8")
        identity = "joint_curriculum_v5_prod_balanced_retention_nextgen_1m_20260609"
        production_parent_checkpoint = (
            "models/production/"
            "joint_torch_v5_balanced_retention_ft_200k_20260608/"
            "joint_torch_latest.pt"
        )
        failed_routepremium_checkpoint = (
            "models/checkpoints/"
            "joint_torch_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft_200k_20260608/"
            "joint_torch_latest.pt"
        )
        stage_schedule = [(entry.name, entry.steps) for entry in curriculum.stage_schedule]

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], 73)
        self.assertEqual(DISCRETE_ACTION_COUNT, 48)
        self.assertEqual(shared["total_timesteps"], 1_000_000)
        self.assertEqual(shared["experiment_name"], identity)
        self.assertEqual(shared["environment_id"], identity)
        self.assertEqual(shared["team_id"], identity)
        self.assertNotEqual(identity, canonical["shared_global_parameters"]["environment_id"])
        self.assertNotEqual(identity, balanced_retention["shared_global_parameters"]["environment_id"])

        self.assertFalse(config["cold_start"]["required"])
        self.assertTrue(config["cold_start"]["fine_tune_initialization_required"])
        self.assertEqual(fine_tune["mode"], "init_from_joint_checkpoint")
        self.assertEqual(fine_tune["source_checkpoint"], production_parent_checkpoint)
        self.assertEqual(fine_tune["source_logical_model_id"], "joint_torch_v5_balanced_retention_ft_200k_20260608")
        self.assertEqual(
            fine_tune["source_environment_id"],
            "joint_curriculum_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k",
        )
        self.assertEqual(fine_tune["source_global_step"], 200_000)
        self.assertEqual(fine_tune["source_parent_global_step"], 1_000_000)
        self.assertEqual(fine_tune["source_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(fine_tune["source_observation_dim"], 73)
        self.assertEqual(fine_tune["source_discrete_action_count"], 48)

        self.assertTrue(run_safety["requires_skip_registry"])
        self.assertTrue(run_safety["requires_disable_trace_logging"])
        self.assertEqual(run_safety["torch_num_threads"], 2)
        self.assertEqual(run_safety["torch_num_interop_threads"], 1)
        self.assertEqual(
            run_safety["output_dir"],
            "models/checkpoints/joint_torch_v5_prod_balanced_retention_nextgen_1m_20260609",
        )
        self.assertNotIn("routepremium_mixed_ft_200k_20260608", run_safety["output_dir"])

        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "normal_v4")
        self.assertEqual(
            stage_schedule,
            [
                ("normal_v4", 75_000),
                ("route_disruption", 100_000),
                ("premium_sla", 100_000),
                ("demand_spike", 125_000),
                ("lead_time_delay", 100_000),
                ("vehicle_scarcity", 75_000),
                ("high_holding_cost", 100_000),
                ("mixed_stress", 125_000),
                ("route_disruption", 75_000),
                ("demand_spike", 50_000),
                ("normal_v4", 75_000),
            ],
        )
        self.assertEqual(sum(steps for _stage, steps in stage_schedule), shared["total_timesteps"])
        self.assertEqual(curriculum.stage_schedule[-1].name, "normal_v4")
        self.assertTrue(set(REAL_WORLD_CURRICULUM_STAGES).issubset({stage for stage, _steps in stage_schedule}))
        self.assertGreater(sum(steps for stage, steps in stage_schedule if stage == "route_disruption"), 100_000)
        self.assertGreater(sum(steps for stage, steps in stage_schedule if stage == "demand_spike"), 125_000)
        self.assertEqual(sum(steps for stage, steps in stage_schedule if stage == "mixed_stress"), 125_000)
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(curriculum.baseline_anchor_probability, 0.0)

        self.assertEqual(config["ppo_hyperparameters"], canonical["ppo_hyperparameters"])
        self.assertEqual(config["dqn_hyperparameters"], canonical["dqn_hyperparameters"])
        self.assertEqual(config["torch_joint_training"], canonical["torch_joint_training"])
        self.assertEqual(config["reward_physics"], canonical["reward_physics"])

        self.assertNotIn(failed_routepremium_checkpoint, config_text)
        self.assertNotIn("joint_torch_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft_200k", config_text)
        self.assertNotIn("joint_torch_v5_clean_cold_start_1m_after_rewardfix/joint_torch_latest.pt", config_text)
        self.assertNotIn("joint_torch_v5_clean_cold_start_1m/joint_torch_latest.pt", config_text)
        self.assertNotIn("joint_curriculum_v4", config_text)
        self.assertNotIn("physical_reality_v4", config_text)
        self.assertNotIn("physical_reality_v3", config_text)
        self.assertNotIn("700k", config_text.lower())
        self.assertNotIn("dqn1024", config_text.lower())
        self.assertNotIn("dqn512", config_text.lower())
        self.assertNotIn("registry_enabled", config_text)

    def test_prod_stability_v2_250k_config_loads_and_is_manual_run_safe(self) -> None:
        canonical = load_config(
            Path("configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean.json")
        )
        balanced_retention = load_config(
            Path("configs/training_joint_curriculum_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k.json")
        )
        nextgen = load_config(Path("configs/training_joint_curriculum_v5_prod_balanced_retention_nextgen_1m_20260609.json"))
        config_path = Path("configs/training_joint_curriculum_v5_prod_stability_v2_250k_20260609.json")

        self.assertTrue(config_path.exists())
        config = load_config(config_path)
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        fine_tune = config["fine_tune_initialization"]
        run_safety = config["manual_run_safety"]
        config_text = config_path.read_text(encoding="utf-8")
        identity = "joint_curriculum_v5_prod_stability_v2_250k_20260609"
        production_parent_checkpoint = (
            "models/production/"
            "joint_torch_v5_balanced_retention_ft_200k_20260608/"
            "joint_torch_latest.pt"
        )
        future_output_dir = Path("models/checkpoints/joint_torch_v5_prod_stability_v2_250k_20260609")
        stage_schedule = [(entry.name, entry.steps) for entry in curriculum.stage_schedule]

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], 73)
        self.assertEqual(DISCRETE_ACTION_COUNT, 48)
        self.assertEqual(shared["total_timesteps"], 250_000)
        self.assertEqual(shared["experiment_name"], identity)
        self.assertEqual(shared["environment_id"], identity)
        self.assertEqual(shared["team_id"], identity)
        self.assertNotEqual(identity, canonical["shared_global_parameters"]["environment_id"])
        self.assertNotEqual(identity, balanced_retention["shared_global_parameters"]["environment_id"])
        self.assertNotEqual(identity, nextgen["shared_global_parameters"]["environment_id"])

        self.assertFalse(config["cold_start"]["required"])
        self.assertTrue(config["cold_start"]["fine_tune_initialization_required"])
        self.assertEqual(fine_tune["mode"], "init_from_joint_checkpoint")
        self.assertEqual(fine_tune["source_checkpoint"], production_parent_checkpoint)
        self.assertEqual(fine_tune["source_logical_model_id"], "joint_torch_v5_balanced_retention_ft_200k_20260608")
        self.assertEqual(
            fine_tune["source_environment_id"],
            "joint_curriculum_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k",
        )
        self.assertEqual(fine_tune["source_global_step"], 200_000)
        self.assertEqual(fine_tune["source_parent_global_step"], 1_000_000)
        self.assertEqual(fine_tune["source_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(fine_tune["source_observation_dim"], 73)
        self.assertEqual(fine_tune["source_discrete_action_count"], 48)

        self.assertTrue(run_safety["requires_skip_registry"])
        self.assertTrue(run_safety["requires_disable_trace_logging"])
        self.assertEqual(run_safety["torch_num_threads"], 2)
        self.assertEqual(run_safety["torch_num_interop_threads"], 1)
        self.assertEqual(run_safety["output_dir"], str(future_output_dir).replace("\\", "/"))

        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "normal_v4")
        self.assertEqual(
            stage_schedule,
            [
                ("normal_v4", 30_000),
                ("premium_sla", 30_000),
                ("route_disruption", 30_000),
                ("lead_time_delay", 25_000),
                ("high_holding_cost", 25_000),
                ("demand_spike", 30_000),
                ("vehicle_scarcity", 20_000),
                ("mixed_stress", 30_000),
                ("normal_v4", 30_000),
            ],
        )
        self.assertEqual(sum(steps for _stage, steps in stage_schedule), shared["total_timesteps"])
        self.assertEqual(stage_schedule[-1][0], "normal_v4")
        self.assertEqual(set(REAL_WORLD_CURRICULUM_STAGES), {stage for stage, _steps in stage_schedule})
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(curriculum.baseline_anchor_probability, 0.0)

        self.assertEqual(config["ppo_hyperparameters"], canonical["ppo_hyperparameters"])
        self.assertEqual(config["dqn_hyperparameters"], canonical["dqn_hyperparameters"])
        self.assertEqual(config["torch_joint_training"], canonical["torch_joint_training"])
        self.assertEqual(config["reward_physics"], canonical["reward_physics"])

        forbidden_fragments = (
            "joint_curriculum_v5_prod_balanced_retention_nextgen_1m_20260609",
            "joint_torch_v5_prod_balanced_retention_nextgen_1m_20260609",
            "joint_torch_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft_200k",
            "joint_torch_v5_clean_cold_start_1m_after_rewardfix/",
            "joint_torch_v5_clean_cold_start_1m/",
            "joint_curriculum_v4",
            "physical_reality_v4",
            "physical_reality_v3",
            "dqn1024",
            "dqn512",
            "registry_enabled",
        )
        for fragment in forbidden_fragments:
            self.assertNotIn(fragment, config_text)
        self.assertNotIn("700k", config_text.lower())

    def test_prod_stability_v2_1_250k_config_loads_and_is_manual_run_safe(self) -> None:
        canonical = load_config(
            Path("configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean.json")
        )
        balanced_retention = load_config(
            Path("configs/training_joint_curriculum_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k.json")
        )
        nextgen = load_config(Path("configs/training_joint_curriculum_v5_prod_balanced_retention_nextgen_1m_20260609.json"))
        v2 = load_config(Path("configs/training_joint_curriculum_v5_prod_stability_v2_250k_20260609.json"))
        config_path = Path("configs/training_joint_curriculum_v5_prod_stability_v2_1_250k_20260609.json")

        self.assertTrue(config_path.exists())
        config = load_config(config_path)
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        fine_tune = config["fine_tune_initialization"]
        run_safety = config["manual_run_safety"]
        config_text = config_path.read_text(encoding="utf-8")
        identity = "joint_curriculum_v5_prod_stability_v2_1_250k_20260609"
        production_parent_checkpoint = (
            "models/production/"
            "joint_torch_v5_balanced_retention_ft_200k_20260608/"
            "joint_torch_latest.pt"
        )
        future_output_dir = Path("models/checkpoints/joint_torch_v5_prod_stability_v2_1_250k_20260609")
        stage_schedule = [(entry.name, entry.steps) for entry in curriculum.stage_schedule]
        v2_schedule = [
            (entry.name, entry.steps)
            for entry in curriculum_config_from_training_config(v2, fallback_seed=99).stage_schedule
        ]
        v2_stage_steps = {
            stage: sum(steps for current_stage, steps in v2_schedule if current_stage == stage)
            for stage in REAL_WORLD_CURRICULUM_STAGES
        }
        v2_1_stage_steps = {
            stage: sum(steps for current_stage, steps in stage_schedule if current_stage == stage)
            for stage in REAL_WORLD_CURRICULUM_STAGES
        }

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], 73)
        self.assertEqual(DISCRETE_ACTION_COUNT, 48)
        self.assertEqual(shared["total_timesteps"], 250_000)
        self.assertEqual(shared["experiment_name"], identity)
        self.assertEqual(shared["environment_id"], identity)
        self.assertEqual(shared["team_id"], identity)
        self.assertNotEqual(identity, canonical["shared_global_parameters"]["environment_id"])
        self.assertNotEqual(identity, balanced_retention["shared_global_parameters"]["environment_id"])
        self.assertNotEqual(identity, nextgen["shared_global_parameters"]["environment_id"])
        self.assertNotEqual(identity, v2["shared_global_parameters"]["environment_id"])

        self.assertFalse(config["cold_start"]["required"])
        self.assertTrue(config["cold_start"]["fine_tune_initialization_required"])
        self.assertEqual(fine_tune["mode"], "init_from_joint_checkpoint")
        self.assertEqual(fine_tune["source_checkpoint"], production_parent_checkpoint)
        self.assertEqual(fine_tune["source_logical_model_id"], "joint_torch_v5_balanced_retention_ft_200k_20260608")
        self.assertEqual(
            fine_tune["source_environment_id"],
            "joint_curriculum_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k",
        )
        self.assertEqual(fine_tune["source_global_step"], 200_000)
        self.assertEqual(fine_tune["source_parent_global_step"], 1_000_000)
        self.assertEqual(fine_tune["source_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(fine_tune["source_observation_dim"], 73)
        self.assertEqual(fine_tune["source_continuous_action_dim"], 5)
        self.assertEqual(fine_tune["source_discrete_action_count"], 48)

        self.assertTrue(run_safety["requires_skip_registry"])
        self.assertTrue(run_safety["requires_disable_trace_logging"])
        self.assertEqual(run_safety["torch_num_threads"], 2)
        self.assertEqual(run_safety["torch_num_interop_threads"], 1)
        self.assertEqual(run_safety["output_dir"], str(future_output_dir).replace("\\", "/"))

        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "normal_v4")
        self.assertEqual(
            stage_schedule,
            [
                ("normal_v4", 25_000),
                ("premium_sla", 35_000),
                ("route_disruption", 35_000),
                ("lead_time_delay", 35_000),
                ("high_holding_cost", 35_000),
                ("demand_spike", 20_000),
                ("vehicle_scarcity", 15_000),
                ("mixed_stress", 25_000),
                ("normal_v4", 25_000),
            ],
        )
        self.assertEqual(sum(steps for _stage, steps in stage_schedule), shared["total_timesteps"])
        self.assertEqual(stage_schedule[-1][0], "normal_v4")
        self.assertEqual(set(REAL_WORLD_CURRICULUM_STAGES), {stage for stage, _steps in stage_schedule})
        for timing_stage in ("premium_sla", "route_disruption", "lead_time_delay", "high_holding_cost"):
            self.assertGreater(v2_1_stage_steps[timing_stage], v2_stage_steps[timing_stage])
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(curriculum.baseline_anchor_probability, 0.0)

        self.assertEqual(config["ppo_hyperparameters"], canonical["ppo_hyperparameters"])
        self.assertEqual(config["dqn_hyperparameters"], canonical["dqn_hyperparameters"])
        self.assertEqual(config["torch_joint_training"], canonical["torch_joint_training"])
        self.assertEqual(config["reward_physics"], canonical["reward_physics"])

        forbidden_fragments = (
            "joint_curriculum_v5_prod_balanced_retention_nextgen_1m_20260609",
            "joint_torch_v5_prod_balanced_retention_nextgen_1m_20260609",
            "joint_curriculum_v5_prod_stability_v2_250k_20260609",
            "joint_torch_v5_prod_stability_v2_250k_20260609",
            "joint_torch_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft_200k",
            "joint_torch_v5_clean_cold_start_1m_after_rewardfix/",
            "joint_torch_v5_clean_cold_start_1m/",
            "joint_curriculum_v4",
            "physical_reality_v4",
            "physical_reality_v3",
            "dqn1024",
            "dqn512",
            "registry_enabled",
        )
        for fragment in forbidden_fragments:
            self.assertNotIn(fragment, config_text)
        self.assertNotIn("700k", config_text.lower())

    def test_prod_stability_v2_2_250k_config_loads_and_is_manual_run_safe(self) -> None:
        canonical = load_config(
            Path("configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean.json")
        )
        balanced_retention = load_config(
            Path("configs/training_joint_curriculum_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k.json")
        )
        nextgen = load_config(Path("configs/training_joint_curriculum_v5_prod_balanced_retention_nextgen_1m_20260609.json"))
        v2 = load_config(Path("configs/training_joint_curriculum_v5_prod_stability_v2_250k_20260609.json"))
        v2_1 = load_config(Path("configs/training_joint_curriculum_v5_prod_stability_v2_1_250k_20260609.json"))
        config_path = Path("configs/training_joint_curriculum_v5_prod_stability_v2_2_250k_20260610.json")

        self.assertTrue(config_path.exists())
        config = load_config(config_path)
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        fine_tune = config["fine_tune_initialization"]
        run_safety = config["manual_run_safety"]
        config_text = config_path.read_text(encoding="utf-8")
        identity = "joint_curriculum_v5_prod_stability_v2_2_250k_20260610"
        production_parent_checkpoint = (
            "models/production/"
            "joint_torch_v5_balanced_retention_ft_200k_20260608/"
            "joint_torch_latest.pt"
        )
        future_output_dir = Path("models/checkpoints/joint_torch_v5_prod_stability_v2_2_250k_20260610")
        stage_schedule = [(entry.name, entry.steps) for entry in curriculum.stage_schedule]

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], 73)
        self.assertEqual(DISCRETE_ACTION_COUNT, 48)
        self.assertEqual(shared["total_timesteps"], 250_000)
        self.assertEqual(shared["experiment_name"], identity)
        self.assertEqual(shared["environment_id"], identity)
        self.assertEqual(shared["team_id"], identity)
        self.assertNotEqual(identity, canonical["shared_global_parameters"]["environment_id"])
        self.assertNotEqual(identity, balanced_retention["shared_global_parameters"]["environment_id"])
        self.assertNotEqual(identity, nextgen["shared_global_parameters"]["environment_id"])
        self.assertNotEqual(identity, v2["shared_global_parameters"]["environment_id"])
        self.assertNotEqual(identity, v2_1["shared_global_parameters"]["environment_id"])

        self.assertFalse(config["cold_start"]["required"])
        self.assertTrue(config["cold_start"]["fine_tune_initialization_required"])
        self.assertEqual(fine_tune["mode"], "init_from_joint_checkpoint")
        self.assertEqual(fine_tune["source_checkpoint"], production_parent_checkpoint)
        self.assertEqual(fine_tune["source_logical_model_id"], "joint_torch_v5_balanced_retention_ft_200k_20260608")
        self.assertEqual(
            fine_tune["source_environment_id"],
            "joint_curriculum_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k",
        )
        self.assertEqual(fine_tune["source_global_step"], 200_000)
        self.assertEqual(fine_tune["source_parent_global_step"], 1_000_000)
        self.assertEqual(fine_tune["source_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(fine_tune["source_observation_dim"], 73)
        self.assertEqual(fine_tune["source_continuous_action_dim"], 5)
        self.assertEqual(fine_tune["source_discrete_action_count"], 48)

        self.assertTrue(run_safety["requires_skip_registry"])
        self.assertTrue(run_safety["requires_disable_trace_logging"])
        self.assertEqual(run_safety["torch_num_threads"], 2)
        self.assertEqual(run_safety["torch_num_interop_threads"], 1)
        self.assertEqual(run_safety["output_dir"], str(future_output_dir).replace("\\", "/"))
        self.assertIn("stability_v2_2_250k_20260610", run_safety["output_dir"])
        self.assertNotIn("stability_v2_250k_20260609", run_safety["output_dir"])
        self.assertNotIn("stability_v2_1_250k_20260609", run_safety["output_dir"])

        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "normal_v4")
        self.assertEqual(
            stage_schedule,
            [
                ("normal_v4", 20_000),
                ("premium_sla", 20_000),
                ("normal_v4", 15_000),
                ("lead_time_delay", 20_000),
                ("route_disruption", 20_000),
                ("normal_v4", 15_000),
                ("high_holding_cost", 20_000),
                ("premium_sla", 20_000),
                ("demand_spike", 20_000),
                ("lead_time_delay", 20_000),
                ("mixed_stress", 20_000),
                ("vehicle_scarcity", 15_000),
                ("normal_v4", 25_000),
            ],
        )
        self.assertEqual(sum(steps for _stage, steps in stage_schedule), shared["total_timesteps"])
        self.assertEqual(stage_schedule[-1][0], "normal_v4")
        self.assertEqual(set(REAL_WORLD_CURRICULUM_STAGES), {stage for stage, _steps in stage_schedule})
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertGreater(curriculum.baseline_anchor_probability, 0.0)
        self.assertEqual(curriculum.baseline_anchor_probability, 0.10)

        self.assertEqual(config["ppo_hyperparameters"], canonical["ppo_hyperparameters"])
        self.assertEqual(config["dqn_hyperparameters"], canonical["dqn_hyperparameters"])
        self.assertEqual(config["torch_joint_training"], canonical["torch_joint_training"])
        self.assertEqual(config["reward_physics"], canonical["reward_physics"])

        forbidden_fragments = (
            "joint_curriculum_v5_prod_balanced_retention_nextgen_1m_20260609",
            "joint_torch_v5_prod_balanced_retention_nextgen_1m_20260609",
            "joint_curriculum_v5_prod_stability_v2_250k_20260609",
            "joint_torch_v5_prod_stability_v2_250k_20260609",
            "joint_curriculum_v5_prod_stability_v2_1_250k_20260609",
            "joint_torch_v5_prod_stability_v2_1_250k_20260609",
            "joint_torch_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft_200k",
            "joint_torch_v5_clean_cold_start_1m_after_rewardfix/",
            "joint_torch_v5_clean_cold_start_1m/",
            "joint_curriculum_v4",
            "physical_reality_v4",
            "physical_reality_v3",
            "dqn1024",
            "dqn512",
            "registry_enabled",
        )
        for fragment in forbidden_fragments:
            self.assertNotIn(fragment, config_text)
        self.assertNotIn("700k", config_text.lower())

    def test_prod_stability_v2_3_250k_config_loads_and_is_manual_run_safe(self) -> None:
        canonical = load_config(
            Path("configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean.json")
        )
        v2_2 = load_config(Path("configs/training_joint_curriculum_v5_prod_stability_v2_2_250k_20260610.json"))
        config_path = Path("configs/training_joint_curriculum_v5_prod_stability_v2_3_250k_20260610.json")

        self.assertTrue(config_path.exists())
        config = load_config(config_path)
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        fine_tune = config["fine_tune_initialization"]
        run_safety = config["manual_run_safety"]
        config_text = config_path.read_text(encoding="utf-8")
        identity = "joint_curriculum_v5_prod_stability_v2_3_250k_20260610"
        production_parent_checkpoint = (
            "models/production/"
            "joint_torch_v5_balanced_retention_ft_200k_20260608/"
            "joint_torch_latest.pt"
        )
        future_output_dir = Path("models/checkpoints/joint_torch_v5_prod_stability_v2_3_250k_20260610")
        stage_schedule = [(entry.name, entry.steps) for entry in curriculum.stage_schedule]

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], 73)
        self.assertEqual(DISCRETE_ACTION_COUNT, 48)
        self.assertEqual(shared["total_timesteps"], 250_000)
        self.assertEqual(shared["experiment_name"], identity)
        self.assertEqual(shared["environment_id"], identity)
        self.assertEqual(shared["team_id"], identity)
        self.assertNotEqual(identity, v2_2["shared_global_parameters"]["environment_id"])

        self.assertFalse(config["cold_start"]["required"])
        self.assertTrue(config["cold_start"]["fine_tune_initialization_required"])
        self.assertEqual(fine_tune["mode"], "init_from_joint_checkpoint")
        self.assertEqual(fine_tune["source_checkpoint"], production_parent_checkpoint)
        self.assertEqual(fine_tune["source_logical_model_id"], "joint_torch_v5_balanced_retention_ft_200k_20260608")
        self.assertEqual(
            fine_tune["source_environment_id"],
            "joint_curriculum_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k",
        )
        self.assertEqual(fine_tune["source_global_step"], 200_000)
        self.assertEqual(fine_tune["source_parent_global_step"], 1_000_000)
        self.assertEqual(fine_tune["source_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(fine_tune["source_observation_dim"], 73)
        self.assertEqual(fine_tune["source_continuous_action_dim"], 5)
        self.assertEqual(fine_tune["source_discrete_action_count"], 48)

        self.assertTrue(run_safety["requires_skip_registry"])
        self.assertTrue(run_safety["requires_disable_trace_logging"])
        self.assertEqual(run_safety["torch_num_threads"], 2)
        self.assertEqual(run_safety["torch_num_interop_threads"], 1)
        self.assertEqual(run_safety["output_dir"], str(future_output_dir).replace("\\", "/"))
        self.assertEqual(
            run_safety["final_ppo_path"],
            "models/checkpoints/joint_torch_v5_prod_stability_v2_3_250k_20260610/"
            "ppo_torch_joint_final_stability_v2_3_250k_20260610.pt",
        )
        self.assertEqual(
            run_safety["final_dqn_path"],
            "models/checkpoints/joint_torch_v5_prod_stability_v2_3_250k_20260610/"
            "dqn_torch_joint_final_stability_v2_3_250k_20260610.pt",
        )
        if future_output_dir.exists():
            self.assertTrue((future_output_dir / "joint_torch_latest.pt").exists())

        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "normal_v4")
        self.assertEqual(
            stage_schedule,
            [
                ("normal_v4", 20_000),
                ("premium_sla", 20_000),
                ("normal_v4", 15_000),
                ("lead_time_delay", 25_000),
                ("high_holding_cost", 20_000),
                ("vehicle_scarcity", 15_000),
                ("route_disruption", 20_000),
                ("normal_v4", 15_000),
                ("premium_sla", 20_000),
                ("lead_time_delay", 25_000),
                ("high_holding_cost", 15_000),
                ("vehicle_scarcity", 10_000),
                ("demand_spike", 10_000),
                ("mixed_stress", 10_000),
                ("normal_v4", 10_000),
            ],
        )
        self.assertEqual(sum(steps for _stage, steps in stage_schedule), shared["total_timesteps"])
        self.assertEqual(stage_schedule[-1][0], "normal_v4")
        self.assertEqual(set(REAL_WORLD_CURRICULUM_STAGES), {stage for stage, _steps in stage_schedule})
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(curriculum.baseline_anchor_probability, 0.12)

        self.assertEqual(config["ppo_hyperparameters"], canonical["ppo_hyperparameters"])
        self.assertEqual(config["dqn_hyperparameters"], canonical["dqn_hyperparameters"])
        self.assertEqual(config["torch_joint_training"], canonical["torch_joint_training"])
        self.assertEqual(config["reward_physics"], canonical["reward_physics"])

        forbidden_fragments = (
            "joint_curriculum_v5_prod_balanced_retention_nextgen_1m_20260609",
            "joint_torch_v5_prod_balanced_retention_nextgen_1m_20260609",
            "joint_curriculum_v5_prod_stability_v2_250k_20260609",
            "joint_torch_v5_prod_stability_v2_250k_20260609",
            "joint_curriculum_v5_prod_stability_v2_1_250k_20260609",
            "joint_torch_v5_prod_stability_v2_1_250k_20260609",
            "joint_curriculum_v5_prod_stability_v2_2_250k_20260610",
            "joint_torch_v5_prod_stability_v2_2_250k_20260610",
            "joint_torch_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft_200k",
            "joint_torch_v5_clean_cold_start_1m_after_rewardfix/",
            "joint_torch_v5_clean_cold_start_1m/",
            "joint_curriculum_v4",
            "physical_reality_v4",
            "physical_reality_v3",
            "dqn1024",
            "dqn512",
            "registry_enabled",
            "700k",
        )
        for fragment in forbidden_fragments:
            self.assertNotIn(fragment, config_text)

    def test_prod_stability_v2_4_250k_config_loads_and_is_manual_run_safe(self) -> None:
        canonical = load_config(
            Path("configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean.json")
        )
        v2_3 = load_config(Path("configs/training_joint_curriculum_v5_prod_stability_v2_3_250k_20260610.json"))
        config_path = Path("configs/training_joint_curriculum_v5_prod_stability_v2_4_250k_20260610.json")

        self.assertTrue(config_path.exists())
        config = load_config(config_path)
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        fine_tune = config["fine_tune_initialization"]
        run_safety = config["manual_run_safety"]
        config_text = config_path.read_text(encoding="utf-8")
        identity = "joint_curriculum_v5_prod_stability_v2_4_250k_20260610"
        production_parent_checkpoint = (
            "models/production/"
            "joint_torch_v5_balanced_retention_ft_200k_20260608/"
            "joint_torch_latest.pt"
        )
        future_output_dir = Path("models/checkpoints/joint_torch_v5_prod_stability_v2_4_250k_20260610")
        stage_schedule = [(entry.name, entry.steps) for entry in curriculum.stage_schedule]

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], 73)
        self.assertEqual(DISCRETE_ACTION_COUNT, 48)
        self.assertEqual(shared["total_timesteps"], 250_000)
        self.assertEqual(shared["experiment_name"], identity)
        self.assertEqual(shared["environment_id"], identity)
        self.assertEqual(shared["team_id"], identity)
        self.assertNotEqual(identity, v2_3["shared_global_parameters"]["environment_id"])

        self.assertFalse(config["cold_start"]["required"])
        self.assertTrue(config["cold_start"]["fine_tune_initialization_required"])
        self.assertEqual(fine_tune["mode"], "init_from_joint_checkpoint")
        self.assertEqual(fine_tune["source_checkpoint"], production_parent_checkpoint)
        self.assertEqual(fine_tune["source_logical_model_id"], "joint_torch_v5_balanced_retention_ft_200k_20260608")
        self.assertEqual(
            fine_tune["source_environment_id"],
            "joint_curriculum_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k",
        )
        self.assertEqual(fine_tune["source_global_step"], 200_000)
        self.assertEqual(fine_tune["source_parent_global_step"], 1_000_000)
        self.assertEqual(fine_tune["source_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(fine_tune["source_observation_dim"], 73)
        self.assertEqual(fine_tune["source_continuous_action_dim"], 5)
        self.assertEqual(fine_tune["source_discrete_action_count"], 48)

        self.assertTrue(run_safety["requires_skip_registry"])
        self.assertTrue(run_safety["requires_disable_trace_logging"])
        self.assertEqual(run_safety["torch_num_threads"], 2)
        self.assertEqual(run_safety["torch_num_interop_threads"], 1)
        self.assertEqual(run_safety["output_dir"], str(future_output_dir).replace("\\", "/"))
        self.assertEqual(
            run_safety["final_ppo_path"],
            "models/checkpoints/joint_torch_v5_prod_stability_v2_4_250k_20260610/"
            "ppo_torch_joint_final_stability_v2_4_250k_20260610.pt",
        )
        self.assertEqual(
            run_safety["final_dqn_path"],
            "models/checkpoints/joint_torch_v5_prod_stability_v2_4_250k_20260610/"
            "dqn_torch_joint_final_stability_v2_4_250k_20260610.pt",
        )
        self.assertFalse(future_output_dir.exists())

        self.assertIn("stability_v2_4_objectives", config)
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "normal_v4")
        self.assertEqual(
            stage_schedule,
            [
                ("normal_v4", 20_000),
                ("premium_sla", 20_000),
                ("normal_v4", 15_000),
                ("lead_time_delay", 25_000),
                ("high_holding_cost", 20_000),
                ("vehicle_scarcity", 15_000),
                ("route_disruption", 20_000),
                ("normal_v4", 15_000),
                ("premium_sla", 20_000),
                ("lead_time_delay", 25_000),
                ("high_holding_cost", 15_000),
                ("vehicle_scarcity", 10_000),
                ("demand_spike", 10_000),
                ("mixed_stress", 10_000),
                ("normal_v4", 10_000),
            ],
        )
        self.assertEqual(sum(steps for _stage, steps in stage_schedule), shared["total_timesteps"])
        self.assertEqual(stage_schedule[-1][0], "normal_v4")
        self.assertEqual(set(REAL_WORLD_CURRICULUM_STAGES), {stage for stage, _steps in stage_schedule})
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(curriculum.baseline_anchor_probability, 0.12)

        self.assertEqual(config["ppo_hyperparameters"], canonical["ppo_hyperparameters"])
        self.assertEqual(config["dqn_hyperparameters"], canonical["dqn_hyperparameters"])
        self.assertEqual(config["torch_joint_training"], canonical["torch_joint_training"])
        self.assertEqual(config["reward_physics"], canonical["reward_physics"])

        forbidden_fragments = (
            "joint_curriculum_v5_prod_balanced_retention_nextgen_1m_20260609",
            "joint_torch_v5_prod_balanced_retention_nextgen_1m_20260609",
            "joint_curriculum_v5_prod_stability_v2_250k_20260609",
            "joint_torch_v5_prod_stability_v2_250k_20260609",
            "joint_curriculum_v5_prod_stability_v2_1_250k_20260609",
            "joint_torch_v5_prod_stability_v2_1_250k_20260609",
            "joint_curriculum_v5_prod_stability_v2_2_250k_20260610",
            "joint_torch_v5_prod_stability_v2_2_250k_20260610",
            "joint_curriculum_v5_prod_stability_v2_3_250k_20260610",
            "joint_torch_v5_prod_stability_v2_3_250k_20260610",
            "joint_torch_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft_200k",
            "joint_torch_v5_clean_cold_start_1m_after_rewardfix/",
            "joint_torch_v5_clean_cold_start_1m/",
            "joint_curriculum_v4",
            "physical_reality_v4",
            "physical_reality_v3",
            "dqn1024",
            "dqn512",
            "registry_enabled",
            "700k",
        )
        for fragment in forbidden_fragments:
            self.assertNotIn(fragment, config_text)

    def test_prod_stability_v2_5_exactresume_250k_config_loads_and_is_manual_run_safe(self) -> None:
        canonical = load_config(
            Path("configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean.json")
        )
        v2_4 = load_config(Path("configs/training_joint_curriculum_v5_prod_stability_v2_4_250k_20260610.json"))
        config_path = Path(
            "configs/training_joint_curriculum_v5_prod_stability_v2_5_exactresume_250k_20260610.json"
        )

        self.assertTrue(config_path.exists())
        config = load_config(config_path)
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        fine_tune = config["fine_tune_initialization"]
        run_safety = config["manual_run_safety"]
        config_text = config_path.read_text(encoding="utf-8")
        identity = "joint_curriculum_v5_prod_stability_v2_5_exactresume_250k_20260610"
        production_parent_checkpoint = (
            "models/production/"
            "joint_torch_v5_balanced_retention_ft_200k_20260608/"
            "joint_torch_latest.pt"
        )
        future_output_dir = Path("models/checkpoints/joint_torch_v5_prod_stability_v2_5_exactresume_250k_20260610")
        stage_schedule = [(entry.name, entry.steps) for entry in curriculum.stage_schedule]
        v2_4_stage_schedule = [(entry["name"], entry["steps"]) for entry in v2_4["curriculum"]["stage_schedule"]]

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], 73)
        self.assertEqual(DISCRETE_ACTION_COUNT, 48)
        self.assertEqual(shared["total_timesteps"], 250_000)
        self.assertEqual(shared["experiment_name"], identity)
        self.assertEqual(shared["environment_id"], identity)
        self.assertEqual(shared["team_id"], identity)
        self.assertNotEqual(identity, v2_4["shared_global_parameters"]["environment_id"])

        self.assertFalse(config["cold_start"]["required"])
        self.assertTrue(config["cold_start"]["fine_tune_initialization_required"])
        self.assertEqual(fine_tune["mode"], "init_from_joint_checkpoint")
        self.assertEqual(fine_tune["source_checkpoint"], production_parent_checkpoint)
        self.assertEqual(fine_tune["source_logical_model_id"], "joint_torch_v5_balanced_retention_ft_200k_20260608")
        self.assertEqual(
            fine_tune["source_environment_id"],
            "joint_curriculum_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k",
        )
        self.assertEqual(fine_tune["source_global_step"], 200_000)
        self.assertEqual(fine_tune["source_parent_global_step"], 1_000_000)
        self.assertEqual(fine_tune["source_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(fine_tune["source_observation_dim"], 73)
        self.assertEqual(fine_tune["source_continuous_action_dim"], 5)
        self.assertEqual(fine_tune["source_discrete_action_count"], 48)

        self.assertTrue(run_safety["requires_skip_registry"])
        self.assertTrue(run_safety["requires_disable_trace_logging"])
        self.assertEqual(run_safety["torch_num_threads"], 2)
        self.assertEqual(run_safety["torch_num_interop_threads"], 1)
        self.assertEqual(run_safety["output_dir"], str(future_output_dir).replace("\\", "/"))
        self.assertEqual(
            run_safety["final_ppo_path"],
            "models/checkpoints/joint_torch_v5_prod_stability_v2_5_exactresume_250k_20260610/"
            "ppo_torch_joint_final_stability_v2_5_exactresume_250k_20260610.pt",
        )
        self.assertEqual(
            run_safety["final_dqn_path"],
            "models/checkpoints/joint_torch_v5_prod_stability_v2_5_exactresume_250k_20260610/"
            "dqn_torch_joint_final_stability_v2_5_exactresume_250k_20260610.pt",
        )
        self.assertTrue(future_output_dir.as_posix().startswith("models/checkpoints/"))
        self.assertIn("v2_5_exactresume_250k_20260610", future_output_dir.as_posix())
        for protected_root in ("models/production", "models/baselines", "models/registry", "db"):
            self.assertFalse(future_output_dir.as_posix().startswith(protected_root))

        self.assertIn("stability_v2_5_exactresume_objectives", config)
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "normal_v4")
        self.assertEqual(stage_schedule, v2_4_stage_schedule)
        self.assertEqual(sum(steps for _stage, steps in stage_schedule), shared["total_timesteps"])
        self.assertEqual(stage_schedule[-1][0], "normal_v4")
        self.assertEqual(set(REAL_WORLD_CURRICULUM_STAGES), {stage for stage, _steps in stage_schedule})
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(curriculum.baseline_anchor_probability, 0.12)

        self.assertEqual(config["ppo_hyperparameters"], canonical["ppo_hyperparameters"])
        self.assertEqual(config["dqn_hyperparameters"], canonical["dqn_hyperparameters"])
        self.assertEqual(config["torch_joint_training"], canonical["torch_joint_training"])
        self.assertEqual(config["reward_physics"], canonical["reward_physics"])

        forbidden_fragments = (
            "joint_curriculum_v5_prod_balanced_retention_nextgen_1m_20260609",
            "joint_torch_v5_prod_balanced_retention_nextgen_1m_20260609",
            "joint_curriculum_v5_prod_stability_v2_250k_20260609",
            "joint_torch_v5_prod_stability_v2_250k_20260609",
            "joint_curriculum_v5_prod_stability_v2_1_250k_20260609",
            "joint_torch_v5_prod_stability_v2_1_250k_20260609",
            "joint_curriculum_v5_prod_stability_v2_2_250k_20260610",
            "joint_torch_v5_prod_stability_v2_2_250k_20260610",
            "joint_curriculum_v5_prod_stability_v2_3_250k_20260610",
            "joint_torch_v5_prod_stability_v2_3_250k_20260610",
            "joint_curriculum_v5_prod_stability_v2_4_250k_20260610",
            "joint_torch_v5_prod_stability_v2_4_250k_20260610",
            "joint_torch_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft_200k",
            "joint_torch_v5_clean_cold_start_1m_after_rewardfix/",
            "joint_torch_v5_clean_cold_start_1m/",
            "joint_curriculum_v4",
            "physical_reality_v4",
            "physical_reality_v3",
            "dqn1024",
            "dqn512",
            "registry_enabled",
            "allow_empty_replay_resume",
            "700k",
        )
        for fragment in forbidden_fragments:
            self.assertNotIn(fragment, config_text)

    def test_prod_teacher_retention_250k_config_loads_and_is_manual_run_safe(self) -> None:
        canonical = load_config(
            Path("configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean.json")
        )
        v2_5_250k = load_config(
            Path("configs/training_joint_curriculum_v5_prod_stability_v2_5_exactresume_250k_20260610.json")
        )
        config_path = Path("configs/training_joint_curriculum_v5_prod_teacher_retention_250k_20260611.json")

        self.assertTrue(config_path.exists())
        config = load_config(config_path)
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        teacher_retention = teacher_retention_config_from_training_config(config)
        shared = config["shared_global_parameters"]
        fine_tune = config["fine_tune_initialization"]
        run_safety = config["manual_run_safety"]
        teacher_section = config["teacher_retention"]
        config_text = config_path.read_text(encoding="utf-8")
        identity = "joint_curriculum_v5_prod_teacher_retention_250k_20260611"
        production_parent_checkpoint = (
            "models/production/"
            "joint_torch_v5_balanced_retention_ft_200k_20260608/"
            "joint_torch_latest.pt"
        )
        candidate_teacher_checkpoint = (
            "models/checkpoints/"
            "joint_torch_v5_prod_stability_v2_5_exactresume_250k_20260610/"
            "joint_torch_latest.pt"
        )
        output_dir = "models/checkpoints/joint_torch_v5_prod_teacher_retention_250k_20260611"
        future_output_dir = Path(output_dir)
        future_eval_dir = Path("models/eval/joint_torch_v5_prod_teacher_retention_250k_20260611_offline_scenarios")
        stage_schedule = [(entry.name, entry.steps) for entry in curriculum.stage_schedule]
        v2_5_stage_schedule = [(entry["name"], entry["steps"]) for entry in v2_5_250k["curriculum"]["stage_schedule"]]

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], 73)
        self.assertEqual(DISCRETE_ACTION_COUNT, 48)
        self.assertEqual(shared["total_timesteps"], 250_000)
        self.assertEqual(shared["experiment_name"], identity)
        self.assertEqual(shared["environment_id"], identity)
        self.assertEqual(shared["team_id"], identity)
        self.assertNotEqual(identity, v2_5_250k["shared_global_parameters"]["environment_id"])

        self.assertFalse(config["cold_start"]["required"])
        self.assertTrue(config["cold_start"]["fine_tune_initialization_required"])
        self.assertEqual(fine_tune["mode"], "init_from_joint_checkpoint")
        self.assertEqual(fine_tune["source_checkpoint"], production_parent_checkpoint)
        self.assertEqual(fine_tune["source_logical_model_id"], "joint_torch_v5_balanced_retention_ft_200k_20260608")
        self.assertEqual(fine_tune["source_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(fine_tune["source_observation_dim"], 73)
        self.assertEqual(fine_tune["source_discrete_action_count"], 48)

        self.assertTrue(run_safety["requires_skip_registry"])
        self.assertTrue(run_safety["requires_disable_trace_logging"])
        self.assertTrue(run_safety["requires_enable_teacher_retention"])
        self.assertEqual(run_safety["torch_num_threads"], 2)
        self.assertEqual(run_safety["torch_num_interop_threads"], 1)
        self.assertEqual(run_safety["output_dir"], output_dir)
        self.assertEqual(
            run_safety["final_ppo_path"],
            "models/checkpoints/joint_torch_v5_prod_teacher_retention_250k_20260611/"
            "ppo_torch_joint_final_teacher_retention_250k_20260611.pt",
        )
        self.assertEqual(
            run_safety["final_dqn_path"],
            "models/checkpoints/joint_torch_v5_prod_teacher_retention_250k_20260611/"
            "dqn_torch_joint_final_teacher_retention_250k_20260611.pt",
        )
        if future_output_dir.exists():
            self.assertTrue((future_output_dir / "joint_torch_latest.pt").exists())
        if future_eval_dir.exists():
            self.assertTrue((future_eval_dir / "scenario_summary.json").exists())
        for protected_root in ("models/production", "models/baselines", "models/registry", "db"):
            self.assertFalse(future_output_dir.as_posix().startswith(protected_root))

        self.assertTrue(teacher_retention.enabled)
        self.assertTrue(teacher_section["explicit_enable"])
        self.assertEqual(teacher_section["production_teacher_checkpoint"], production_parent_checkpoint)
        self.assertEqual(teacher_section["candidate_teacher_checkpoint"], candidate_teacher_checkpoint)
        self.assertEqual(str(teacher_retention.production_teacher_checkpoint).replace("\\", "/"), production_parent_checkpoint)
        self.assertEqual(str(teacher_retention.candidate_teacher_checkpoint).replace("\\", "/"), candidate_teacher_checkpoint)
        self.assertEqual(teacher_section["bad_pocket_action_ids"], [32, 33, 41, 45, 46])
        self.assertEqual(teacher_section["state_bank_scenarios"], [
            "baseline",
            "lead_time",
            "route_disruption",
            "premium_sla",
            "high_holding",
            "vehicle_scarcity",
            "mixed_stress",
            "demand_spike",
        ])
        self.assertEqual(teacher_section["teacher_preference_by_scenario"]["baseline"], "production")
        self.assertEqual(teacher_section["teacher_preference_by_scenario"]["lead_time"], "candidate")
        self.assertLessEqual(teacher_section["max_loss"], 0.10)
        self.assertGreater(teacher_section["loss_weight"], 0.0)

        self.assertIn("teacher_retention_objectives", config)
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "normal_v4")
        self.assertEqual(stage_schedule, v2_5_stage_schedule)
        self.assertEqual(sum(steps for _stage, steps in stage_schedule), shared["total_timesteps"])
        self.assertEqual(set(REAL_WORLD_CURRICULUM_STAGES), {stage for stage, _steps in stage_schedule})
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(curriculum.baseline_anchor_probability, 0.12)

        self.assertEqual(config["ppo_hyperparameters"], canonical["ppo_hyperparameters"])
        self.assertEqual(config["dqn_hyperparameters"], canonical["dqn_hyperparameters"])
        self.assertEqual(config["torch_joint_training"], canonical["torch_joint_training"])
        self.assertEqual(config["reward_physics"], canonical["reward_physics"])

        forbidden_fragments = (
            "joint_curriculum_v5_prod_balanced_retention_nextgen_1m_20260609",
            "joint_torch_v5_prod_balanced_retention_nextgen_1m_20260609",
            "joint_curriculum_v5_prod_stability_v2_250k_20260609",
            "joint_torch_v5_prod_stability_v2_250k_20260609",
            "joint_curriculum_v5_prod_stability_v2_1_250k_20260609",
            "joint_torch_v5_prod_stability_v2_1_250k_20260609",
            "joint_curriculum_v5_prod_stability_v2_2_250k_20260610",
            "joint_torch_v5_prod_stability_v2_2_250k_20260610",
            "joint_curriculum_v5_prod_stability_v2_3_250k_20260610",
            "joint_torch_v5_prod_stability_v2_3_250k_20260610",
            "joint_curriculum_v5_prod_stability_v2_4_500k_20260610",
            "joint_torch_v5_prod_stability_v2_4_500k_20260610",
            "joint_curriculum_v5_prod_stability_v2_5_exactresume_500k_20260610",
            "joint_torch_v5_prod_stability_v2_5_exactresume_500k_20260610",
            "joint_torch_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft_200k",
            "joint_torch_v5_clean_cold_start_1m_after_rewardfix/",
            "joint_torch_v5_clean_cold_start_1m/",
            "joint_curriculum_v4",
            "physical_reality_v4",
            "physical_reality_v3",
            "dqn1024",
            "dqn512",
            "registry_enabled",
            "allow_empty_replay_resume",
            "700k",
        )
        for fragment in forbidden_fragments:
            self.assertNotIn(fragment, config_text)

    def test_prod_hierarchical_v1_250k_config_loads_and_is_manual_run_safe(self) -> None:
        canonical = load_config(
            Path("configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean.json")
        )
        v2_5_250k = load_config(
            Path("configs/training_joint_curriculum_v5_prod_stability_v2_5_exactresume_250k_20260610.json")
        )
        config_path = Path("configs/training_joint_curriculum_v5_prod_hierarchical_v1_250k_20260611.json")

        self.assertTrue(config_path.exists())
        config = load_config(config_path)
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        hierarchical_init = hierarchical_dqn_initialization_config_from_training_config(config)
        shared = config["shared_global_parameters"]
        fine_tune = config["fine_tune_initialization"]
        run_safety = config["manual_run_safety"]
        torch_joint = config["torch_joint_training"]
        init_section = config["hierarchical_initialization"]
        config_text = config_path.read_text(encoding="utf-8")
        identity = "joint_curriculum_v5_prod_hierarchical_v1_250k_20260611"
        production_parent_checkpoint = (
            "models/production/"
            "joint_torch_v5_balanced_retention_ft_200k_20260608/"
            "joint_torch_latest.pt"
        )
        output_dir = "models/checkpoints/joint_torch_v5_prod_hierarchical_v1_250k_20260611"
        stage_schedule = [(entry.name, entry.steps) for entry in curriculum.stage_schedule]
        v2_5_stage_schedule = [(entry["name"], entry["steps"]) for entry in v2_5_250k["curriculum"]["stage_schedule"]]

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], 73)
        self.assertEqual(DISCRETE_ACTION_COUNT, 48)
        self.assertEqual(shared["total_timesteps"], 250_000)
        self.assertEqual(shared["experiment_name"], identity)
        self.assertEqual(shared["environment_id"], identity)
        self.assertEqual(shared["team_id"], identity)

        self.assertFalse(config["cold_start"]["required"])
        self.assertTrue(config["cold_start"]["fine_tune_initialization_required"])
        self.assertEqual(fine_tune["mode"], "init_from_joint_checkpoint")
        self.assertEqual(fine_tune["source_checkpoint"], production_parent_checkpoint)
        self.assertEqual(fine_tune["source_logical_model_id"], "joint_torch_v5_balanced_retention_ft_200k_20260608")
        self.assertEqual(fine_tune["source_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(fine_tune["source_observation_dim"], 73)
        self.assertEqual(fine_tune["source_discrete_action_count"], 48)
        self.assertEqual(fine_tune["hierarchical_init_method"], HIERARCHICAL_INIT_METHOD_FLAT_TEACHER_DISTILLATION)

        self.assertEqual(torch_joint["dqn_architecture"], "hierarchical_v1")
        self.assertTrue(hierarchical_init.enabled)
        self.assertTrue(init_section["explicit_enable"])
        self.assertEqual(init_section["method"], HIERARCHICAL_INIT_METHOD_FLAT_TEACHER_DISTILLATION)
        self.assertEqual(init_section["teacher_checkpoint"], production_parent_checkpoint)
        self.assertEqual(str(hierarchical_init.teacher_checkpoint).replace("\\", "/"), production_parent_checkpoint)
        self.assertEqual(init_section["observation_sample_count"], 4096)
        self.assertEqual(init_section["distillation_steps"], 512)
        self.assertEqual(init_section["batch_size"], 128)
        self.assertLessEqual(init_section["max_loss"], 1.0)
        self.assertTrue(init_section["transfer_dqn_trunk"])

        self.assertTrue(run_safety["requires_skip_registry"])
        self.assertTrue(run_safety["requires_disable_trace_logging"])
        self.assertTrue(run_safety["requires_hierarchical_initialization"])
        self.assertFalse(run_safety["training_authorized"])
        self.assertEqual(run_safety["torch_num_threads"], 2)
        self.assertEqual(run_safety["torch_num_interop_threads"], 1)
        self.assertEqual(run_safety["output_dir"], output_dir)
        self.assertEqual(
            run_safety["final_ppo_path"],
            "models/checkpoints/joint_torch_v5_prod_hierarchical_v1_250k_20260611/"
            "ppo_torch_joint_final_hierarchical_v1_250k_20260611.pt",
        )
        self.assertEqual(
            run_safety["final_dqn_path"],
            "models/checkpoints/joint_torch_v5_prod_hierarchical_v1_250k_20260611/"
            "dqn_torch_joint_final_hierarchical_v1_250k_20260611.pt",
        )
        for protected_root in ("models/production", "models/baselines", "models/registry", "db"):
            self.assertFalse(output_dir.startswith(protected_root))

        self.assertIn("hierarchical_v1_objectives", config)
        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "normal_v4")
        self.assertEqual(stage_schedule, v2_5_stage_schedule)
        self.assertEqual(sum(steps for _stage, steps in stage_schedule), shared["total_timesteps"])
        self.assertEqual(set(REAL_WORLD_CURRICULUM_STAGES), {stage for stage, _steps in stage_schedule})
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(curriculum.baseline_anchor_probability, 0.12)

        self.assertEqual(config["ppo_hyperparameters"], canonical["ppo_hyperparameters"])
        self.assertEqual(config["dqn_hyperparameters"], canonical["dqn_hyperparameters"])
        canonical_torch = dict(canonical["torch_joint_training"])
        canonical_torch["dqn_architecture"] = "hierarchical_v1"
        self.assertEqual(torch_joint, canonical_torch)
        self.assertEqual(config["reward_physics"], canonical["reward_physics"])

        forbidden_fragments = (
            "joint_curriculum_v5_prod_balanced_retention_nextgen_1m_20260609",
            "joint_torch_v5_prod_balanced_retention_nextgen_1m_20260609",
            "joint_curriculum_v5_prod_stability_v2_250k_20260609",
            "joint_torch_v5_prod_stability_v2_250k_20260609",
            "joint_curriculum_v5_prod_stability_v2_1_250k_20260609",
            "joint_torch_v5_prod_stability_v2_1_250k_20260609",
            "joint_curriculum_v5_prod_stability_v2_2_250k_20260610",
            "joint_torch_v5_prod_stability_v2_2_250k_20260610",
            "joint_curriculum_v5_prod_stability_v2_3_250k_20260610",
            "joint_torch_v5_prod_stability_v2_3_250k_20260610",
            "joint_curriculum_v5_prod_stability_v2_4_500k_20260610",
            "joint_torch_v5_prod_stability_v2_4_500k_20260610",
            "joint_curriculum_v5_prod_stability_v2_5_exactresume_500k_20260610",
            "joint_torch_v5_prod_stability_v2_5_exactresume_500k_20260610",
            "joint_torch_v5_prod_teacher_retention_250k_20260611",
            "joint_torch_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft_200k",
            "joint_torch_v5_clean_cold_start_1m_after_rewardfix/",
            "joint_torch_v5_clean_cold_start_1m/",
            "joint_curriculum_v4",
            "physical_reality_v4",
            "physical_reality_v3",
            "dqn1024",
            "dqn512",
            "registry_enabled",
            "allow_empty_replay_resume",
            "700k",
        )
        for fragment in forbidden_fragments:
            self.assertNotIn(fragment, config_text)

    def test_prod_hierarchical_v1_500k_config_loads_and_requires_exact_resume(self) -> None:
        canonical = load_config(
            Path("configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean.json")
        )
        hierarchical_250k = load_config(
            Path("configs/training_joint_curriculum_v5_prod_hierarchical_v1_250k_20260611.json")
        )
        config_path = Path("configs/training_joint_curriculum_v5_prod_hierarchical_v1_500k_20260611.json")

        self.assertTrue(config_path.exists())
        config = load_config(config_path)
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        fine_tune = config["fine_tune_initialization"]
        continuation = config["ladder_continuation"]
        run_safety = config["manual_run_safety"]
        torch_joint = config["torch_joint_training"]
        config_text = config_path.read_text(encoding="utf-8")
        identity = "joint_curriculum_v5_prod_hierarchical_v1_500k_20260611"
        parent_checkpoint = (
            "models/checkpoints/"
            "joint_torch_v5_prod_hierarchical_v1_250k_20260611/"
            "joint_torch_latest.pt"
        )
        production_parent_checkpoint = (
            "models/production/"
            "joint_torch_v5_balanced_retention_ft_200k_20260608/"
            "joint_torch_latest.pt"
        )
        output_dir = "models/checkpoints/joint_torch_v5_prod_hierarchical_v1_500k_20260611"
        stage_schedule = [(entry.name, entry.steps) for entry in curriculum.stage_schedule]
        base_stage_schedule = [
            (entry["name"], entry["steps"]) for entry in hierarchical_250k["curriculum"]["stage_schedule"]
        ]

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], 73)
        self.assertEqual(DISCRETE_ACTION_COUNT, 48)
        self.assertEqual(shared["total_timesteps"], 500_000)
        self.assertEqual(shared["experiment_name"], identity)
        self.assertEqual(shared["environment_id"], identity)
        self.assertEqual(shared["team_id"], identity)

        self.assertFalse(config["cold_start"]["required"])
        self.assertTrue(config["cold_start"]["reject_stale_contracts"])
        self.assertEqual(config["cold_start"]["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(fine_tune["mode"], "init_from_joint_checkpoint")
        self.assertEqual(fine_tune["source_checkpoint"], production_parent_checkpoint)
        self.assertEqual(fine_tune["source_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(fine_tune["source_observation_dim"], 73)
        self.assertEqual(fine_tune["source_discrete_action_count"], 48)
        self.assertEqual(fine_tune["hierarchical_init_method"], HIERARCHICAL_INIT_METHOD_FLAT_TEACHER_DISTILLATION)

        self.assertEqual(torch_joint["dqn_architecture"], "hierarchical_v1")
        self.assertNotIn("hierarchical_initialization", config)
        self.assertEqual(continuation["mode"], "exact_resume_from_passed_checkpoint")
        self.assertEqual(continuation["parent_checkpoint"], parent_checkpoint)
        self.assertEqual(continuation["parent_global_step"], 250_000)
        self.assertEqual(continuation["target_global_step"], 500_000)
        self.assertEqual(continuation["parent_gate_decision"], "PASS")
        self.assertEqual(continuation["parent_gate_exit_code"], 0)
        self.assertEqual(continuation["parent_dqn_architecture"], "hierarchical_v1")
        self.assertEqual(continuation["parent_hierarchical_init_method"], HIERARCHICAL_INIT_METHOD_FLAT_TEACHER_DISTILLATION)
        self.assertTrue(continuation["requires_replay_state"])
        self.assertTrue(continuation["requires_rng_state"])
        self.assertFalse(continuation["legacy_warm_resume_allowed"])

        self.assertTrue(run_safety["requires_skip_registry"])
        self.assertTrue(run_safety["requires_disable_trace_logging"])
        self.assertTrue(run_safety["requires_resume"])
        self.assertTrue(run_safety["forbids_init_from_joint_checkpoint"])
        self.assertTrue(run_safety["requires_hierarchical_architecture"])
        self.assertEqual(run_safety["resume_checkpoint"], parent_checkpoint)
        self.assertEqual(run_safety["torch_num_threads"], 2)
        self.assertEqual(run_safety["torch_num_interop_threads"], 1)
        self.assertEqual(run_safety["output_dir"], output_dir)
        self.assertEqual(
            run_safety["final_ppo_path"],
            "models/checkpoints/joint_torch_v5_prod_hierarchical_v1_500k_20260611/"
            "ppo_torch_joint_final_hierarchical_v1_500k_20260611.pt",
        )
        self.assertEqual(
            run_safety["final_dqn_path"],
            "models/checkpoints/joint_torch_v5_prod_hierarchical_v1_500k_20260611/"
            "dqn_torch_joint_final_hierarchical_v1_500k_20260611.pt",
        )
        self.assertTrue(output_dir.startswith("models/checkpoints/"))
        for protected_root in ("models/production", "models/baselines", "models/registry", "db"):
            self.assertFalse(output_dir.startswith(protected_root))

        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "normal_v4")
        self.assertEqual(len(stage_schedule), len(base_stage_schedule) * 2)
        self.assertEqual(stage_schedule[: len(base_stage_schedule)], base_stage_schedule)
        self.assertEqual(stage_schedule[len(base_stage_schedule) :], base_stage_schedule)
        self.assertEqual(sum(steps for _stage, steps in stage_schedule), shared["total_timesteps"])
        self.assertEqual(set(REAL_WORLD_CURRICULUM_STAGES), {stage for stage, _steps in stage_schedule})
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(curriculum.baseline_anchor_probability, 0.12)

        self.assertEqual(config["ppo_hyperparameters"], canonical["ppo_hyperparameters"])
        self.assertEqual(config["dqn_hyperparameters"], canonical["dqn_hyperparameters"])
        canonical_torch = dict(canonical["torch_joint_training"])
        canonical_torch["dqn_architecture"] = "hierarchical_v1"
        self.assertEqual(torch_joint, canonical_torch)
        self.assertEqual(config["reward_physics"], canonical["reward_physics"])

        forbidden_fragments = (
            "joint_curriculum_v5_prod_balanced_retention_nextgen_1m_20260609",
            "joint_torch_v5_prod_balanced_retention_nextgen_1m_20260609",
            "joint_curriculum_v5_prod_stability_v2_250k_20260609",
            "joint_torch_v5_prod_stability_v2_250k_20260609",
            "joint_curriculum_v5_prod_stability_v2_1_250k_20260609",
            "joint_torch_v5_prod_stability_v2_1_250k_20260609",
            "joint_curriculum_v5_prod_stability_v2_2_250k_20260610",
            "joint_torch_v5_prod_stability_v2_2_250k_20260610",
            "joint_curriculum_v5_prod_stability_v2_3_250k_20260610",
            "joint_torch_v5_prod_stability_v2_3_250k_20260610",
            "joint_curriculum_v5_prod_stability_v2_4_500k_20260610",
            "joint_torch_v5_prod_stability_v2_4_500k_20260610",
            "joint_curriculum_v5_prod_stability_v2_5_exactresume_250k_20260610",
            "joint_torch_v5_prod_stability_v2_5_exactresume_250k_20260610",
            "joint_curriculum_v5_prod_stability_v2_5_exactresume_500k_20260610",
            "joint_torch_v5_prod_stability_v2_5_exactresume_500k_20260610",
            "joint_torch_v5_prod_teacher_retention_250k_20260611",
            "joint_torch_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft_200k",
            "joint_torch_v5_clean_cold_start_1m_after_rewardfix/",
            "joint_torch_v5_clean_cold_start_1m/",
            "joint_curriculum_v4",
            "physical_reality_v4",
            "physical_reality_v3",
            "dqn1024",
            "dqn512",
            "registry_enabled",
            "allow_empty_replay_resume",
            "700k",
        )
        for fragment in forbidden_fragments:
            self.assertNotIn(fragment, config_text)

    def test_prod_hierarchical_v1_1m_config_loads_and_requires_exact_resume(self) -> None:
        canonical = load_config(
            Path("configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean.json")
        )
        hierarchical_250k = load_config(
            Path("configs/training_joint_curriculum_v5_prod_hierarchical_v1_250k_20260611.json")
        )
        config_path = Path("configs/training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json")

        self.assertTrue(config_path.exists())
        config = load_config(config_path)
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        fine_tune = config["fine_tune_initialization"]
        continuation = config["ladder_continuation"]
        run_safety = config["manual_run_safety"]
        torch_joint = config["torch_joint_training"]
        config_text = config_path.read_text(encoding="utf-8")
        identity = "joint_curriculum_v5_prod_hierarchical_v1_1m_20260611"
        parent_checkpoint = (
            "models/checkpoints/"
            "joint_torch_v5_prod_hierarchical_v1_500k_20260611/"
            "joint_torch_latest.pt"
        )
        production_parent_checkpoint = (
            "models/production/"
            "joint_torch_v5_balanced_retention_ft_200k_20260608/"
            "joint_torch_latest.pt"
        )
        output_dir = "models/checkpoints/joint_torch_v5_prod_hierarchical_v1_1m_20260611"
        stage_schedule = [(entry.name, entry.steps) for entry in curriculum.stage_schedule]
        base_stage_schedule = [
            (entry["name"], entry["steps"]) for entry in hierarchical_250k["curriculum"]["stage_schedule"]
        ]

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], 73)
        self.assertEqual(DISCRETE_ACTION_COUNT, 48)
        self.assertEqual(shared["total_timesteps"], 1_000_000)
        self.assertEqual(shared["experiment_name"], identity)
        self.assertEqual(shared["environment_id"], identity)
        self.assertEqual(shared["team_id"], identity)

        self.assertFalse(config["cold_start"]["required"])
        self.assertTrue(config["cold_start"]["reject_stale_contracts"])
        self.assertEqual(config["cold_start"]["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(fine_tune["mode"], "init_from_joint_checkpoint")
        self.assertEqual(fine_tune["source_checkpoint"], production_parent_checkpoint)
        self.assertEqual(fine_tune["source_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(fine_tune["source_observation_dim"], 73)
        self.assertEqual(fine_tune["source_discrete_action_count"], 48)
        self.assertEqual(fine_tune["hierarchical_init_method"], HIERARCHICAL_INIT_METHOD_FLAT_TEACHER_DISTILLATION)

        self.assertEqual(torch_joint["dqn_architecture"], "hierarchical_v1")
        self.assertNotIn("hierarchical_initialization", config)
        self.assertEqual(continuation["mode"], "exact_resume_from_passed_checkpoint")
        self.assertEqual(continuation["parent_checkpoint"], parent_checkpoint)
        self.assertEqual(continuation["parent_global_step"], 500_000)
        self.assertEqual(continuation["target_global_step"], 1_000_000)
        self.assertEqual(continuation["parent_gate_decision"], "PASS")
        self.assertEqual(continuation["parent_gate_exit_code"], 0)
        self.assertEqual(continuation["parent_dqn_architecture"], "hierarchical_v1")
        self.assertEqual(continuation["parent_hierarchical_init_method"], HIERARCHICAL_INIT_METHOD_FLAT_TEACHER_DISTILLATION)
        self.assertTrue(continuation["requires_replay_state"])
        self.assertTrue(continuation["requires_rng_state"])
        self.assertFalse(continuation["legacy_warm_resume_allowed"])

        self.assertTrue(run_safety["requires_skip_registry"])
        self.assertTrue(run_safety["requires_disable_trace_logging"])
        self.assertTrue(run_safety["requires_resume"])
        self.assertTrue(run_safety["forbids_init_from_joint_checkpoint"])
        self.assertTrue(run_safety["requires_hierarchical_architecture"])
        self.assertEqual(run_safety["resume_checkpoint"], parent_checkpoint)
        self.assertEqual(run_safety["torch_num_threads"], 2)
        self.assertEqual(run_safety["torch_num_interop_threads"], 1)
        self.assertEqual(run_safety["output_dir"], output_dir)
        self.assertEqual(
            run_safety["final_ppo_path"],
            "models/checkpoints/joint_torch_v5_prod_hierarchical_v1_1m_20260611/"
            "ppo_torch_joint_final_hierarchical_v1_1m_20260611.pt",
        )
        self.assertEqual(
            run_safety["final_dqn_path"],
            "models/checkpoints/joint_torch_v5_prod_hierarchical_v1_1m_20260611/"
            "dqn_torch_joint_final_hierarchical_v1_1m_20260611.pt",
        )
        self.assertTrue(output_dir.startswith("models/checkpoints/"))
        for protected_root in ("models/production", "models/baselines", "models/registry", "db"):
            self.assertFalse(output_dir.startswith(protected_root))

        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "normal_v4")
        self.assertEqual(len(stage_schedule), len(base_stage_schedule) * 4)
        for index in range(4):
            start = index * len(base_stage_schedule)
            stop = start + len(base_stage_schedule)
            self.assertEqual(stage_schedule[start:stop], base_stage_schedule)
        self.assertEqual(sum(steps for _stage, steps in stage_schedule), shared["total_timesteps"])
        self.assertEqual(set(REAL_WORLD_CURRICULUM_STAGES), {stage for stage, _steps in stage_schedule})
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(curriculum.baseline_anchor_probability, 0.12)

        self.assertEqual(config["ppo_hyperparameters"], canonical["ppo_hyperparameters"])
        self.assertEqual(config["dqn_hyperparameters"], canonical["dqn_hyperparameters"])
        canonical_torch = dict(canonical["torch_joint_training"])
        canonical_torch["dqn_architecture"] = "hierarchical_v1"
        self.assertEqual(torch_joint, canonical_torch)
        self.assertEqual(config["reward_physics"], canonical["reward_physics"])

        forbidden_fragments = (
            "joint_curriculum_v5_prod_balanced_retention_nextgen_1m_20260609",
            "joint_torch_v5_prod_balanced_retention_nextgen_1m_20260609",
            "joint_curriculum_v5_prod_stability_v2_250k_20260609",
            "joint_torch_v5_prod_stability_v2_250k_20260609",
            "joint_curriculum_v5_prod_stability_v2_1_250k_20260609",
            "joint_torch_v5_prod_stability_v2_1_250k_20260609",
            "joint_curriculum_v5_prod_stability_v2_2_250k_20260610",
            "joint_torch_v5_prod_stability_v2_2_250k_20260610",
            "joint_curriculum_v5_prod_stability_v2_3_250k_20260610",
            "joint_torch_v5_prod_stability_v2_3_250k_20260610",
            "joint_curriculum_v5_prod_stability_v2_4_500k_20260610",
            "joint_torch_v5_prod_stability_v2_4_500k_20260610",
            "joint_curriculum_v5_prod_stability_v2_5_exactresume_250k_20260610",
            "joint_torch_v5_prod_stability_v2_5_exactresume_250k_20260610",
            "joint_curriculum_v5_prod_stability_v2_5_exactresume_500k_20260610",
            "joint_torch_v5_prod_stability_v2_5_exactresume_500k_20260610",
            "joint_torch_v5_prod_teacher_retention_250k_20260611",
            "joint_torch_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft_200k",
            "joint_torch_v5_clean_cold_start_1m_after_rewardfix/",
            "joint_torch_v5_clean_cold_start_1m/",
            "joint_curriculum_v4",
            "physical_reality_v4",
            "physical_reality_v3",
            "dqn1024",
            "dqn512",
            "registry_enabled",
            "allow_empty_replay_resume",
            "700k",
        )
        for fragment in forbidden_fragments:
            self.assertNotIn(fragment, config_text)

    def test_prod_stability_v2_5_exactresume_500k_config_loads_and_requires_exact_resume(self) -> None:
        canonical = load_config(
            Path("configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean.json")
        )
        v2_5_250k = load_config(
            Path("configs/training_joint_curriculum_v5_prod_stability_v2_5_exactresume_250k_20260610.json")
        )
        config_path = Path(
            "configs/training_joint_curriculum_v5_prod_stability_v2_5_exactresume_500k_20260610.json"
        )

        self.assertTrue(config_path.exists())
        config = load_config(config_path)
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        fine_tune = config["fine_tune_initialization"]
        continuation = config["ladder_continuation"]
        run_safety = config["manual_run_safety"]
        config_text = config_path.read_text(encoding="utf-8")
        identity = "joint_curriculum_v5_prod_stability_v2_5_exactresume_500k_20260610"
        parent_checkpoint = (
            "models/checkpoints/"
            "joint_torch_v5_prod_stability_v2_5_exactresume_250k_20260610/"
            "joint_torch_latest.pt"
        )
        production_parent_checkpoint = (
            "models/production/"
            "joint_torch_v5_balanced_retention_ft_200k_20260608/"
            "joint_torch_latest.pt"
        )
        output_dir = "models/checkpoints/joint_torch_v5_prod_stability_v2_5_exactresume_500k_20260610"
        stage_schedule = [(entry.name, entry.steps) for entry in curriculum.stage_schedule]
        base_stage_schedule = [
            (entry["name"], entry["steps"]) for entry in v2_5_250k["curriculum"]["stage_schedule"]
        ]

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], 73)
        self.assertEqual(DISCRETE_ACTION_COUNT, 48)
        self.assertEqual(shared["total_timesteps"], 500_000)
        self.assertEqual(shared["experiment_name"], identity)
        self.assertEqual(shared["environment_id"], identity)
        self.assertEqual(shared["team_id"], identity)

        self.assertFalse(config["cold_start"]["required"])
        self.assertTrue(config["cold_start"]["reject_stale_contracts"])
        self.assertEqual(config["cold_start"]["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(fine_tune["mode"], "init_from_joint_checkpoint")
        self.assertEqual(fine_tune["source_checkpoint"], production_parent_checkpoint)

        self.assertEqual(continuation["mode"], "exact_resume_from_passed_checkpoint")
        self.assertEqual(continuation["parent_checkpoint"], parent_checkpoint)
        self.assertEqual(continuation["parent_global_step"], 250_000)
        self.assertEqual(continuation["target_global_step"], 500_000)
        self.assertEqual(continuation["parent_gate_decision"], "PASS")
        self.assertEqual(continuation["parent_gate_exit_code"], 0)
        self.assertTrue(continuation["requires_replay_state"])
        self.assertTrue(continuation["requires_rng_state"])
        self.assertFalse(continuation["legacy_warm_resume_allowed"])

        self.assertTrue(run_safety["requires_skip_registry"])
        self.assertTrue(run_safety["requires_disable_trace_logging"])
        self.assertEqual(run_safety["torch_num_threads"], 2)
        self.assertEqual(run_safety["torch_num_interop_threads"], 1)
        self.assertEqual(run_safety["output_dir"], output_dir)
        self.assertEqual(
            run_safety["final_ppo_path"],
            "models/checkpoints/joint_torch_v5_prod_stability_v2_5_exactresume_500k_20260610/"
            "ppo_torch_joint_final_stability_v2_5_exactresume_500k_20260610.pt",
        )
        self.assertEqual(
            run_safety["final_dqn_path"],
            "models/checkpoints/joint_torch_v5_prod_stability_v2_5_exactresume_500k_20260610/"
            "dqn_torch_joint_final_stability_v2_5_exactresume_500k_20260610.pt",
        )
        self.assertTrue(output_dir.startswith("models/checkpoints/"))
        self.assertIn("v2_5_exactresume_500k_20260610", output_dir)
        for protected_root in ("models/production", "models/baselines", "models/registry", "db"):
            self.assertFalse(output_dir.startswith(protected_root))

        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "normal_v4")
        self.assertEqual(len(stage_schedule), len(base_stage_schedule) * 2)
        self.assertEqual(stage_schedule[: len(base_stage_schedule)], base_stage_schedule)
        self.assertEqual(stage_schedule[len(base_stage_schedule) :], base_stage_schedule)
        self.assertEqual(sum(steps for _stage, steps in stage_schedule), shared["total_timesteps"])
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(curriculum.baseline_anchor_probability, 0.12)

        self.assertEqual(config["ppo_hyperparameters"], canonical["ppo_hyperparameters"])
        self.assertEqual(config["dqn_hyperparameters"], canonical["dqn_hyperparameters"])
        self.assertEqual(config["torch_joint_training"], canonical["torch_joint_training"])
        self.assertEqual(config["reward_physics"], canonical["reward_physics"])

        forbidden_fragments = (
            "joint_curriculum_v5_prod_balanced_retention_nextgen_1m_20260609",
            "joint_torch_v5_prod_balanced_retention_nextgen_1m_20260609",
            "joint_curriculum_v5_prod_stability_v2_250k_20260609",
            "joint_torch_v5_prod_stability_v2_250k_20260609",
            "joint_curriculum_v5_prod_stability_v2_1_250k_20260609",
            "joint_torch_v5_prod_stability_v2_1_250k_20260609",
            "joint_curriculum_v5_prod_stability_v2_2_250k_20260610",
            "joint_torch_v5_prod_stability_v2_2_250k_20260610",
            "joint_curriculum_v5_prod_stability_v2_3_250k_20260610",
            "joint_torch_v5_prod_stability_v2_3_250k_20260610",
            "joint_curriculum_v5_prod_stability_v2_4_250k_20260610",
            "joint_torch_v5_prod_stability_v2_4_250k_20260610",
            "joint_torch_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft_200k",
            "joint_torch_v5_clean_cold_start_1m_after_rewardfix/",
            "joint_torch_v5_clean_cold_start_1m/",
            "joint_curriculum_v4",
            "physical_reality_v4",
            "physical_reality_v3",
            "dqn1024",
            "dqn512",
            "registry_enabled",
            "allow_empty_replay_resume",
            "700k",
        )
        for fragment in forbidden_fragments:
            self.assertNotIn(fragment, config_text)

    def test_prod_stability_v2_4_500k_config_loads_and_is_manual_run_safe(self) -> None:
        canonical = load_config(
            Path("configs/training_joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean.json")
        )
        v2_4 = load_config(Path("configs/training_joint_curriculum_v5_prod_stability_v2_4_250k_20260610.json"))
        config_path = Path("configs/training_joint_curriculum_v5_prod_stability_v2_4_500k_20260610.json")

        self.assertTrue(config_path.exists())
        config = load_config(config_path)
        validate_active_mdp_contract(config)
        curriculum = curriculum_config_from_training_config(config, fallback_seed=99)
        shared = config["shared_global_parameters"]
        fine_tune = config["fine_tune_initialization"]
        continuation = config["ladder_continuation"]
        run_safety = config["manual_run_safety"]
        config_text = config_path.read_text(encoding="utf-8")
        identity = "joint_curriculum_v5_prod_stability_v2_4_500k_20260610"
        parent_checkpoint = (
            "models/checkpoints/"
            "joint_torch_v5_prod_stability_v2_4_250k_20260610/"
            "joint_torch_latest.pt"
        )
        production_parent_checkpoint = (
            "models/production/"
            "joint_torch_v5_balanced_retention_ft_200k_20260608/"
            "joint_torch_latest.pt"
        )
        output_dir = "models/checkpoints/joint_torch_v5_prod_stability_v2_4_500k_20260610"
        stage_schedule = [(entry.name, entry.steps) for entry in curriculum.stage_schedule]
        v2_4_stage_schedule = [
            (entry["name"], entry["steps"]) for entry in v2_4["curriculum"]["stage_schedule"]
        ]

        self.assertEqual(config["mdp_contract_version"], MDP_CONTRACT_VERSION)
        self.assertEqual(shared["observation_dim"], OBSERVATION_DIM)
        self.assertEqual(shared["observation_dim"], 73)
        self.assertEqual(DISCRETE_ACTION_COUNT, 48)
        self.assertEqual(shared["total_timesteps"], 500_000)
        self.assertEqual(shared["experiment_name"], identity)
        self.assertEqual(shared["environment_id"], identity)
        self.assertEqual(shared["team_id"], identity)
        self.assertNotEqual(identity, v2_4["shared_global_parameters"]["environment_id"])

        self.assertFalse(config["cold_start"]["required"])
        self.assertTrue(config["cold_start"]["reject_stale_contracts"])
        self.assertEqual(config["cold_start"]["allow_resume_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(fine_tune["mode"], "init_from_joint_checkpoint")
        self.assertEqual(fine_tune["source_checkpoint"], production_parent_checkpoint)
        self.assertEqual(fine_tune["source_contract"], MDP_CONTRACT_VERSION)
        self.assertEqual(fine_tune["source_observation_dim"], 73)
        self.assertEqual(fine_tune["source_continuous_action_dim"], 5)
        self.assertEqual(fine_tune["source_discrete_action_count"], 48)

        self.assertEqual(continuation["mode"], "continue_from_passed_checkpoint")
        self.assertEqual(continuation["parent_checkpoint"], parent_checkpoint)
        self.assertEqual(continuation["parent_global_step"], 250_000)
        self.assertEqual(continuation["target_global_step"], 500_000)
        self.assertEqual(continuation["parent_gate_decision"], "PASS")
        self.assertEqual(continuation["parent_gate_exit_code"], 0)

        self.assertTrue(run_safety["requires_skip_registry"])
        self.assertTrue(run_safety["requires_disable_trace_logging"])
        self.assertEqual(run_safety["torch_num_threads"], 2)
        self.assertEqual(run_safety["torch_num_interop_threads"], 1)
        self.assertEqual(run_safety["output_dir"], output_dir)
        self.assertEqual(
            run_safety["final_ppo_path"],
            "models/checkpoints/joint_torch_v5_prod_stability_v2_4_500k_20260610/"
            "ppo_torch_joint_final_stability_v2_4_500k_20260610.pt",
        )
        self.assertEqual(
            run_safety["final_dqn_path"],
            "models/checkpoints/joint_torch_v5_prod_stability_v2_4_500k_20260610/"
            "dqn_torch_joint_final_stability_v2_4_500k_20260610.pt",
        )

        self.assertTrue(curriculum.enabled)
        self.assertEqual(curriculum.stage, "normal_v4")
        self.assertEqual(len(stage_schedule), len(v2_4_stage_schedule) * 2)
        self.assertEqual(stage_schedule[: len(v2_4_stage_schedule)], v2_4_stage_schedule)
        self.assertEqual(stage_schedule[len(v2_4_stage_schedule) :], v2_4_stage_schedule)
        self.assertEqual(sum(steps for _stage, steps in stage_schedule), shared["total_timesteps"])
        self.assertEqual(set(REAL_WORLD_CURRICULUM_STAGES), {stage for stage, _steps in stage_schedule})
        self.assertFalse(curriculum.randomization.enabled)
        self.assertEqual(curriculum.randomization.ranges_profile, "fixed_medium")
        self.assertEqual(curriculum.baseline_anchor_probability, 0.12)

        self.assertEqual(config["ppo_hyperparameters"], canonical["ppo_hyperparameters"])
        self.assertEqual(config["dqn_hyperparameters"], canonical["dqn_hyperparameters"])
        self.assertEqual(config["torch_joint_training"], canonical["torch_joint_training"])
        self.assertEqual(config["reward_physics"], canonical["reward_physics"])

        forbidden_fragments = (
            "joint_curriculum_v5_prod_balanced_retention_nextgen_1m_20260609",
            "joint_torch_v5_prod_balanced_retention_nextgen_1m_20260609",
            "joint_curriculum_v5_prod_stability_v2_250k_20260609",
            "joint_torch_v5_prod_stability_v2_250k_20260609",
            "joint_curriculum_v5_prod_stability_v2_1_250k_20260609",
            "joint_torch_v5_prod_stability_v2_1_250k_20260609",
            "joint_curriculum_v5_prod_stability_v2_2_250k_20260610",
            "joint_torch_v5_prod_stability_v2_2_250k_20260610",
            "joint_curriculum_v5_prod_stability_v2_3_250k_20260610",
            "joint_torch_v5_prod_stability_v2_3_250k_20260610",
            "joint_torch_v5_clean_after_rewardfix_perfclean_routepremium_mixed_ft_200k",
            "joint_torch_v5_clean_cold_start_1m_after_rewardfix/",
            "joint_torch_v5_clean_cold_start_1m/",
            "joint_curriculum_v4",
            "physical_reality_v4",
            "physical_reality_v3",
            "dqn1024",
            "dqn512",
            "registry_enabled",
            "700k",
        )
        for fragment in forbidden_fragments:
            self.assertNotIn(fragment, config_text)

    def test_invalid_stage_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown curriculum stage"):
            CurriculumConfig.from_mapping({"enabled": True, "stage": "not_real"})

    def test_invalid_stage_schedule_step_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "steps must be a positive integer"):
            CurriculumConfig.from_mapping(
                {
                    "enabled": True,
                    "stage": "normal_v4",
                    "stage_schedule": [{"name": "normal_v4", "steps": 0}],
                }
            )

    def test_invalid_baseline_anchor_probability_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "baseline_anchor_probability"):
            CurriculumConfig.from_mapping(
                {"enabled": True, "stage": "normal_v4", "baseline_anchor_probability": 1.5}
            )

    def test_invalid_randomization_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ranges_profile"):
            CurriculumConfig.from_mapping(
                {
                    "enabled": True,
                    "stage": "normal_v4",
                    "randomization": {"enabled": True, "ranges_profile": "all_high_everywhere"},
                }
            )

    def test_invalid_explicit_override_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported key"):
            CurriculumConfig.from_mapping(
                {
                    "enabled": True,
                    "stage": "premium_sla",
                    "explicit_overrides": {"rolling_demand_volume_multiplier": 2.0},
                }
            )

    def test_invalid_explicit_override_numeric_value_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be finite"):
            CurriculumConfig.from_mapping(
                {
                    "enabled": True,
                    "stage": "demand_spike",
                    "explicit_overrides": {"rolling_demand_volume_multiplier": float("nan")},
                }
            )

    def test_disabled_curriculum_returns_no_op_overrides(self) -> None:
        sample = CurriculumSampler(CurriculumConfig.disabled(seed=11)).sample(episode_index=3)

        self.assertFalse(sample.enabled)
        self.assertEqual(sample.environment_overrides, {})
        self.assertEqual(sample.effective_stage, "normal_v4")

    def test_same_seed_stage_and_episode_gives_same_overrides(self) -> None:
        config = CurriculumConfig(
            enabled=True,
            stage="demand_spike",
            randomization=CurriculumRandomizationConfig(enabled=True, ranges_profile="medium_single_stress"),
            seed=123,
        )
        sampler = CurriculumSampler(config)

        self.assertEqual(sampler.sample(5), sampler.sample(5))

    def test_different_episode_can_change_randomized_sample_deterministically(self) -> None:
        config = CurriculumConfig(
            enabled=True,
            stage="demand_spike",
            randomization=CurriculumRandomizationConfig(enabled=True, ranges_profile="medium_single_stress"),
            seed=123,
        )
        sampler = CurriculumSampler(config)

        first = sampler.sample(5).environment_overrides
        second = sampler.sample(6).environment_overrides

        self.assertNotEqual(first, second)
        self.assertEqual(second, sampler.sample(6).environment_overrides)

    def test_stage_override_families_are_distinct_and_supported(self) -> None:
        expected_keys = {
            "normal_v4": set(),
            "high_holding_cost": {
                "holding_cost_multiplier",
                "excess_inventory_penalty_multiplier",
                "planned_replenishment_cost_multiplier",
            },
            "vehicle_scarcity": {
                "vehicle_availability_multiplier",
                "fleet_capacity_multiplier",
                "capacity_shock_severity",
            },
            "route_disruption": {
                "route_disruption_probability",
                "congestion_multiplier",
                "shortest_route_risk_multiplier",
                "traversal_cost_multiplier",
                "disruption_risk_multiplier",
            },
            "premium_sla": {
                "premium_sla_ratio",
                "urgent_due_window_multiplier",
                "premium_lateness_penalty_multiplier",
            },
            "demand_spike": {
                "rolling_demand_probability_multiplier",
                "rolling_demand_volume_multiplier",
                "order_units_multiplier",
                "urgent_order_probability_multiplier",
            },
            "lead_time_delay": {
                "supplier_delay_probability",
                "lead_time_mean_multiplier",
                "lead_time_variance_multiplier",
                "stockout_penalty_multiplier",
            },
        }

        for stage, keys in expected_keys.items():
            config = CurriculumConfig(enabled=True, stage=stage)  # type: ignore[arg-type]
            overrides = CurriculumSampler(config).sample(episode_index=0).environment_overrides
            self.assertEqual(set(overrides), keys)
            self.assertTrue(set(overrides).issubset(SUPPORTED_OVERRIDE_KEYS))

    def test_normal_v4_stage_produces_baseline_like_overrides(self) -> None:
        sample = CurriculumSampler(CurriculumConfig(enabled=True, stage="normal_v4")).sample(episode_index=0)

        self.assertEqual(sample.environment_overrides, {})

    def test_high_holding_cost_stage_produces_holding_cost_overrides(self) -> None:
        overrides = CurriculumSampler(CurriculumConfig(enabled=True, stage="high_holding_cost")).sample(0).environment_overrides

        self.assertGreater(overrides["holding_cost_multiplier"], 1.0)
        self.assertGreater(overrides["excess_inventory_penalty_multiplier"], 1.0)

    def test_vehicle_scarcity_stage_produces_scarcity_overrides(self) -> None:
        overrides = CurriculumSampler(CurriculumConfig(enabled=True, stage="vehicle_scarcity")).sample(0).environment_overrides

        self.assertLess(overrides["vehicle_availability_multiplier"], 1.0)
        self.assertLess(overrides["fleet_capacity_multiplier"], 1.0)
        self.assertGreater(overrides["capacity_shock_severity"], 0.0)

    def test_route_disruption_stage_produces_route_risk_overrides(self) -> None:
        overrides = CurriculumSampler(CurriculumConfig(enabled=True, stage="route_disruption")).sample(0).environment_overrides

        self.assertGreater(overrides["route_disruption_probability"], 0.0)
        self.assertGreater(overrides["congestion_multiplier"], 1.0)
        self.assertGreater(overrides["shortest_route_risk_multiplier"], 1.0)

    def test_premium_sla_stage_produces_urgency_overrides(self) -> None:
        overrides = CurriculumSampler(CurriculumConfig(enabled=True, stage="premium_sla")).sample(0).environment_overrides

        self.assertGreater(overrides["premium_sla_ratio"], 0.0)
        self.assertLess(overrides["urgent_due_window_multiplier"], 1.0)
        self.assertGreater(overrides["premium_lateness_penalty_multiplier"], 1.0)

    def test_demand_spike_stage_produces_demand_overrides(self) -> None:
        overrides = CurriculumSampler(CurriculumConfig(enabled=True, stage="demand_spike")).sample(0).environment_overrides

        self.assertGreater(overrides["rolling_demand_probability_multiplier"], 1.0)
        self.assertGreater(overrides["rolling_demand_volume_multiplier"], 1.0)
        self.assertGreater(overrides["order_units_multiplier"], 1.0)

    def test_lead_time_stage_produces_supplier_and_lead_time_overrides(self) -> None:
        overrides = CurriculumSampler(CurriculumConfig(enabled=True, stage="lead_time_delay")).sample(0).environment_overrides

        self.assertGreater(overrides["supplier_delay_probability"], 0.0)
        self.assertGreater(overrides["lead_time_mean_multiplier"], 1.0)
        self.assertGreater(overrides["lead_time_variance_multiplier"], 1.0)

    def test_mixed_stress_stage_is_moderate_not_all_high(self) -> None:
        config = CurriculumConfig(
            enabled=True,
            stage="mixed_stress",
            randomization=CurriculumRandomizationConfig(enabled=True, ranges_profile="high_late_stress"),
            seed=99,
        )
        overrides = CurriculumSampler(config).sample(episode_index=2).environment_overrides

        self.assertLessEqual(overrides["capacity_shock_severity"], 0.35)
        self.assertLessEqual(overrides["route_disruption_probability"], 0.35)
        self.assertGreaterEqual(overrides["vehicle_availability_multiplier"], 0.70)
        self.assertLessEqual(overrides["rolling_demand_probability_multiplier"], 1.60)

    def test_observation_dim_is_73_and_old_v3_checkpoint_is_rejected(self) -> None:
        self.assertEqual(OBSERVATION_DIM, 73)
        self.assertEqual(MDP_CONTRACT_VERSION, "physical_reality_v5_route_candidate_visibility")

        with self.assertRaisesRegex(ValueError, MDP_CONTRACT_MISMATCH_MESSAGE):
            validate_checkpoint_mdp_contract(
                {
                    "config": {
                        "mdp_contract_version": "physical_reality_v3_dispatch_feasibility",
                        "shared_global_parameters": {"observation_dim": 49},
                    }
                }
            )


if __name__ == "__main__":
    unittest.main()


def _first_non_anchor_sample(sampler: CurriculumSampler, *, global_step: int | None = None):
    for episode_index in range(20):
        sample = sampler.sample(episode_index, global_step=global_step)
        if not sample.baseline_anchor_used:
            return sample
    raise AssertionError("expected at least one non-anchor curriculum sample")

