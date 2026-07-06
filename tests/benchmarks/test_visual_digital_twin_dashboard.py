from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts import visual_digital_twin_dashboard as dashboard


ROOT = Path(__file__).resolve().parents[2]


class VisualDigitalTwinDashboardV3Tests(unittest.TestCase):
    def test_v3_turkish_control_room_tabs_and_forbidden_phrases(self) -> None:
        corpus = dashboard.dashboard_text_corpus()

        for label in [
            "Kontrol Odası",
            "Canlı Senaryo",
            "Dijital İkiz Haritası",
            "PPO + DQN Karar Akışı",
            "48 Aksiyon Sözlüğü",
            "Kamu Replay Analizi",
            "KPI ve Zaman Çizgisi",
            "Kanıtlar ve Sınırlar",
            "Kamu replay betimleyici proxy analizdir; nedensel OPE veya şirket verisi doğrulaması değildir.",
            "CANLI_SIMULASYON_MODU",
            "GÜVENLİ_SENTETİK_MOD",
            "PUBLIC_REPLAY_MODU",
        ]:
            self.assertIn(label, corpus)

        for old_label in [
            "Genel Bakış",
            "Senaryo Çalıştırıcı",
            "PPO + DQN Karar İzi",
            "48 Aksiyon Çözücü",
            "Kamu Replay Dağılımı",
            "synthetic visual demo trajectory",
        ]:
            self.assertNotIn(old_label, corpus)

        for phrase in dashboard.FORBIDDEN_DASHBOARD_PHRASES:
            self.assertNotIn(phrase, corpus.lower())

    def test_live_backend_mode_steps_real_env_or_reports_safe_fallback(self) -> None:
        backend = dashboard.create_dashboard_backend(
            ROOT,
            scenario_id="baseline_normal",
            mode=dashboard.CANLI_SIMULASYON_MODU,
            use_production_policy=False,
        )
        initial = backend.current_state()
        advanced = backend.step()

        self.assertIn(advanced["mode"], {dashboard.CANLI_SIMULASYON_MODU, dashboard.GUVENLI_SENTETIK_MOD})
        self.assertGreaterEqual(advanced["step_index"], initial["step_index"])
        self.assertIn("backend_source", advanced)
        self.assertIn("trajectory_label", advanced)
        self.assertIn("observation_summary", advanced)
        self.assertIn("talep/SLA", advanced["observation_summary"])
        self.assertIn("filo/kapasite", advanced["observation_summary"])
        self.assertIn("rota/sıkışıklık", advanced["observation_summary"])
        self.assertIn("stok/ikmal", advanced["observation_summary"])
        self.assertIn("senaryo stresi", advanced["observation_summary"])
        self.assertEqual(set(advanced["ppo_controls"]), set(dashboard.PPO_CONTROLS))
        self.assertTrue(0 <= advanced["selected_action_id"] < 48)
        self.assertIn("operational_interpretation_tr", advanced["decoded_action"])

        if advanced["mode"] == dashboard.CANLI_SIMULASYON_MODU:
            self.assertIn("env_5pl", advanced["backend_source"])
            self.assertFalse(advanced["safe_fallback_active"])
        else:
            self.assertTrue(advanced["safe_fallback_active"])
            self.assertIn("güvenli sentetik", advanced["fallback_reason"].lower())

    def test_synthetic_fallback_is_clearly_labeled_when_live_root_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = dashboard.create_dashboard_backend(
                Path(tmp),
                scenario_id="baseline_normal",
                mode=dashboard.CANLI_SIMULASYON_MODU,
                use_production_policy=False,
            )
            state = backend.current_state()

        self.assertEqual(state["mode"], dashboard.GUVENLI_SENTETIK_MOD)
        self.assertTrue(state["safe_fallback_active"])
        self.assertIn("güvenli sentetik", state["fallback_reason"].lower())
        self.assertIn("canlı env", state["fallback_reason"].lower())

    def test_auto_run_session_helpers_and_speed_mapping(self) -> None:
        session = dashboard.initial_dashboard_session_state("mixed_stress")

        self.assertEqual(session["scenario_id"], "mixed_stress")
        self.assertFalse(session["running"])
        self.assertEqual(session["mode"], dashboard.CANLI_SIMULASYON_MODU)
        self.assertTrue(dashboard.should_auto_advance(True, dashboard.CANLI_SIMULASYON_MODU))
        self.assertFalse(dashboard.should_auto_advance(False, dashboard.CANLI_SIMULASYON_MODU))
        self.assertLess(
            dashboard.autorun_interval_ms("5x"),
            dashboard.autorun_interval_ms("1x"),
        )
        self.assertGreater(
            dashboard.autorun_interval_ms("0.25x"),
            dashboard.autorun_interval_ms("1x"),
        )

    def test_scenario_discovery_includes_all_configs_and_documented_aliases(self) -> None:
        scenarios = dashboard.discover_scenarios(ROOT / "configs" / "eval_scenarios")
        by_id = {scenario["scenario_id"]: scenario for scenario in scenarios}

        config_ids = {path.stem for path in (ROOT / "configs" / "eval_scenarios").glob("*.json")}
        self.assertTrue(config_ids)
        self.assertTrue(config_ids.issubset(by_id))

        for alias in dashboard.DOCUMENTED_SCENARIO_ALIASES:
            self.assertIn(alias, by_id)
            self.assertIn(by_id[alias]["status"], {"runnable", "documented / not runnable in current workspace"})

        self.assertEqual(by_id["baseline_normal"]["status"], "runnable")

    def test_action_decoder_exposes_all_48_actions_and_watch_labels(self) -> None:
        actions = dashboard.build_action_decoder_rows()

        self.assertEqual(len(actions), 48)
        self.assertEqual({row["action_id"] for row in actions}, set(range(48)))
        self.assertEqual(actions[24]["watch"], "watched action")
        self.assertEqual(actions[32]["watch"], "watched action")
        self.assertEqual(actions[24]["route_family"], "shortest")
        self.assertEqual(actions[32]["route_family"], "low_congestion")
        self.assertNotIn("whole model", actions[24]["operational_interpretation"].lower())

    def test_public_replay_distribution_preserves_expected_totals(self) -> None:
        replay = dashboard.load_public_replay_distribution(
            ROOT / "Tez" / "FINAL_EKLER" / "EK_A_48_ACTION_MAPPING_AND_PUBLIC_REPLAY_COUNTS.md",
            ROOT / "Tez" / "FINAL_EKLER" / "EK_A2_ACTION_FAMILY_DISTRIBUTION_SUMMARY.md",
        )

        self.assertEqual(replay["dataset_totals"]["LaDe"], 31_415)
        self.assertEqual(replay["dataset_totals"]["NYC HVFHS"], 100_000)
        self.assertEqual(replay["dataset_totals"]["Olist"], 96_476)
        self.assertEqual(replay["global_total"], 227_891)
        self.assertEqual(len(replay["actions"]), 48)
        self.assertEqual(sum(row["global_count"] for row in replay["actions"]), 227_891)
        self.assertTrue({row["action_id"] for row in replay["zero_count_actions"]})

    def test_html_export_contains_v3_static_preview_and_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dashboard.html"
            state = dashboard.build_dashboard_state(ROOT)

            dashboard.export_html_dashboard(state, output)
            html = output.read_text(encoding="utf-8")

            self.assertTrue(output.exists())
            self.assertIn("Bu statik önizlemedir. Canlı ürün demosu için Streamlit çalıştırın.", html)
            self.assertIn("Kontrol Odası", html)
            self.assertIn("Başlat", html)
            self.assertIn("Durdur", html)
            self.assertIn("Tek Adım", html)
            self.assertIn("Dijital İkiz Haritası", html)
            self.assertIn("PPO + DQN Karar Akışı", html)
            self.assertIn("48 Aksiyon Sözlüğü", html)
            self.assertIn("Kamu Replay Analizi", html)
            self.assertIn("id='dataset-filter'", html)
            self.assertIn("id='scenario-select'", html)
            self.assertIn("id='route-filter'", html)
            self.assertIn("CANLI_SIMULASYON_MODU", html)
            self.assertNotIn("decision-support system", html.lower())

    def test_cli_export_runs_to_v3_path_when_invoked_by_script_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "reports" / "demo_dashboard_v3" / "index.html"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "visual_digital_twin_dashboard.py"),
                    "--export-html",
                    str(output),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.exists())
            self.assertIn("LIVE_TURKISH_DIGITAL_TWIN_DASHBOARD", result.stdout)


if __name__ == "__main__":
    unittest.main()
