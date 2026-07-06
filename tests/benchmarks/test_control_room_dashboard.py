from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock

from scripts import control_room_server as control_room


ROOT = Path(__file__).resolve().parents[2]


class ControlRoomDashboardV4Tests(unittest.TestCase):
    def test_get_state_is_read_only_even_when_running(self) -> None:
        app = control_room.ControlRoomApp(ROOT)

        app.control_start()
        first = app.dispatch_api("GET", "/api/state")["state"]["step"]
        second = app.dispatch_api("GET", "/api/state")["state"]["step"]
        third = app.dispatch_api("GET", "/api/status")["state"]["step"]

        self.assertTrue(app.get_state()["running"])
        self.assertEqual(first, 0)
        self.assertEqual(second, 0)
        self.assertEqual(third, 0)

    def test_tick_and_step_are_the_only_simulation_mutations(self) -> None:
        app = control_room.ControlRoomApp(ROOT)

        app.control_start()
        ticked = app.dispatch_api("POST", "/api/control/tick")["state"]
        self.assertEqual(ticked["step"], 1)
        self.assertTrue(ticked["running"])

        app.control_pause()
        paused_step = app.get_state()["step"]
        no_tick = app.dispatch_api("POST", "/api/control/tick")["state"]
        self.assertEqual(no_tick["step"], paused_step)
        stepped = app.dispatch_api("POST", "/api/control/step")["state"]
        self.assertEqual(stepped["step"], paused_step + 1)
        self.assertFalse(stepped["running"])

    def test_start_pause_and_reset_are_idempotent_and_deterministic(self) -> None:
        app = control_room.ControlRoomApp(ROOT)

        first_start = app.control_start()["step"]
        second_start = app.control_start()["step"]
        self.assertEqual(first_start, second_start)
        self.assertTrue(app.get_state()["running"])

        app.control_pause()
        paused_once = app.get_state()["step"]
        app.control_pause()
        self.assertEqual(app.get_state()["step"], paused_once)
        self.assertFalse(app.get_state()["running"])

        app.control_step()
        reset_a = app.control_reset()
        app.control_step()
        reset_b = app.control_reset()
        for key in ["step", "simulated_time", "selected_scenario", "mode", "dqn_action", "kpis"]:
            self.assertEqual(reset_a[key], reset_b[key])

    def test_demo_akisi_is_default_and_does_not_load_live_policy(self) -> None:
        app = control_room.ControlRoomApp(ROOT)
        state = app.get_state()

        self.assertEqual(state["mode"], control_room.SUNUM_MODU)
        self.assertFalse(state["production_policy_read_only"])
        self.assertFalse(state["env_connected"])
        self.assertIn("Demo Akışı", state["presentation_notice_tr"])
        self.assertEqual(state["step"], 0)

    def test_presentation_golden_traces_are_deterministic(self) -> None:
        traces_a = {
            scenario: control_room.build_presentation_trace(ROOT, scenario, steps=120)
            for scenario in control_room.PRESENTATION_TRACE_SCENARIOS
        }
        traces_b = {
            scenario: control_room.build_presentation_trace(ROOT, scenario, steps=120)
            for scenario in control_room.PRESENTATION_TRACE_SCENARIOS
        }

        self.assertEqual(traces_a, traces_b)
        for scenario, trace in traces_a.items():
            self.assertEqual(trace["scenario_id"], scenario)
            self.assertEqual(len(trace["steps"]), 120)
            first = trace["steps"][0]
            for field in [
                "step",
                "simulated_time",
                "scenario_id",
                "vehicles",
                "orders",
                "routes",
                "active_route",
                "ppo_controls",
                "dqn_action_id",
                "decoded_action",
                "kpis",
                "event_log_line",
            ]:
                self.assertIn(field, first)
            self.assertTrue(0 <= first["dqn_action_id"] < 48)

    def test_v5_static_assets_and_golden_traces_exist_after_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "demo_v5" / "index.html"
            control_room.export_static(ROOT, output)
            base = output.parent

            for path in [
                base / "index.html",
                base / "app.js",
                base / "styles.css",
                base / "golden_traces" / "baseline_normal_trace.json",
                base / "golden_traces" / "mixed_stress_trace.json",
                base / "golden_traces" / "route_disruption_congestion_trace.json",
            ]:
                self.assertTrue(path.exists(), str(path))
            trace = json.loads((base / "golden_traces" / "baseline_normal_trace.json").read_text(encoding="utf-8"))
            self.assertEqual(len(trace["steps"]), 120)

    def test_v6_operation_traces_are_deterministic_and_schema_valid(self) -> None:
        required = {
            "trace_id",
            "scenario_id",
            "step",
            "simulated_time",
            "operation_phase",
            "operation_phase_tr",
            "hubs",
            "vehicles",
            "orders",
            "route_candidates",
            "active_routes",
            "congestion_zones",
            "disruption_zones",
            "observation_summary",
            "ppo_controls",
            "ppo_source",
            "dqn_action_id",
            "decoded_action",
            "kpis",
            "event_log",
            "action_history",
        }
        traces_a = {
            scenario: control_room.build_operation_trace(ROOT, scenario, steps=180)
            for scenario in control_room.PRESENTATION_TRACE_SCENARIOS
        }
        traces_b = {
            scenario: control_room.build_operation_trace(ROOT, scenario, steps=180)
            for scenario in control_room.PRESENTATION_TRACE_SCENARIOS
        }

        self.assertEqual(traces_a, traces_b)
        for scenario, trace in traces_a.items():
            self.assertEqual(trace["trace_id"], scenario)
            self.assertEqual(trace["scenario_id"], scenario)
            self.assertEqual(trace["playback_mode"], control_room.OPERASYON_KAYDI_OYNATICI)
            self.assertGreaterEqual(len(trace["steps"]), 180)
            first = trace["steps"][0]
            self.assertFalse(required - set(first))
            self.assertEqual(set(first["ppo_controls"]), set(control_room.PPO_CONTROLS))
            self.assertIn(first["ppo_source"], {"policy_read_only", "golden_trace", "deterministic_demo_policy", "fallback_static"})
            self.assertTrue(0 <= first["dqn_action_id"] < 48)
            self.assertTrue(first["operation_phase_tr"])
            self.assertNotIn("_", first["operation_phase_tr"])

    def test_v6_operation_trace_entities_are_stable_and_vehicle_movement_is_continuous(self) -> None:
        trace = control_room.build_operation_trace(ROOT, "route_disruption_congestion", steps=180)
        steps = trace["steps"]

        hub_positions = {
            hub["id"]: (hub["x"], hub["y"])
            for hub in steps[0]["hubs"]
        }
        order_positions = {
            order["id"]: (order["x"], order["y"])
            for order in steps[0]["orders"]
        }
        route_paths = {
            route["id"]: tuple((point["x"], point["y"]) for point in route["path"])
            for route in steps[0]["route_candidates"]
        }
        vehicle_ids = {vehicle["id"] for vehicle in steps[0]["vehicles"]}
        statuses_seen = {order["id"]: {order["status"]} for order in steps[0]["orders"]}

        previous_vehicle_positions = {
            vehicle["id"]: (float(vehicle["x"]), float(vehicle["y"]))
            for vehicle in steps[0]["vehicles"]
        }
        max_jump = 0.0
        for row in steps[1:]:
            self.assertEqual({hub["id"]: (hub["x"], hub["y"]) for hub in row["hubs"]}, hub_positions)
            self.assertEqual({order["id"]: (order["x"], order["y"]) for order in row["orders"]}, order_positions)
            self.assertEqual(
                {route["id"]: tuple((point["x"], point["y"]) for point in route["path"]) for route in row["route_candidates"]},
                route_paths,
            )
            self.assertEqual({vehicle["id"] for vehicle in row["vehicles"]}, vehicle_ids)
            for order in row["orders"]:
                statuses_seen[order["id"]].add(order["status"])
            for vehicle in row["vehicles"]:
                previous = previous_vehicle_positions[vehicle["id"]]
                current = (float(vehicle["x"]), float(vehicle["y"]))
                jump = ((current[0] - previous[0]) ** 2 + (current[1] - previous[1]) ** 2) ** 0.5
                max_jump = max(max_jump, jump)
                previous_vehicle_positions[vehicle["id"]] = current

        self.assertLess(max_jump, 9.0)
        lifecycle_values = set().union(*statuses_seen.values())
        self.assertTrue({"waiting", "assigned", "in_transit", "delivered"}.issubset(lifecycle_values))

    def test_v6_playback_api_is_read_only_and_tick_seek_work(self) -> None:
        app = control_room.ControlRoomApp(ROOT)
        first = app.dispatch_api("GET", "/api/playback/state")["state"]
        second = app.dispatch_api("GET", "/api/playback/state")["state"]
        third = app.dispatch_api("GET", f"/api/trace/{first['loaded_trace_id']}")["trace"]

        self.assertEqual(first["playback_mode"], control_room.OPERASYON_KAYDI_OYNATICI)
        self.assertEqual(first["step"], second["step"])
        self.assertEqual(first["loaded_trace_id"], third["trace_id"])
        self.assertEqual(first["ppo_source"], "deterministic_demo_policy")
        self.assertIn("ppo_variance_summary", first)

        app.dispatch_api("POST", "/api/playback/play")
        ticked = app.dispatch_api("POST", "/api/playback/tick")["state"]
        self.assertEqual(ticked["step"], first["step"] + 1)
        app.dispatch_api("POST", "/api/playback/pause")
        paused_step = app.dispatch_api("GET", "/api/playback/state")["state"]["step"]
        paused_tick = app.dispatch_api("POST", "/api/playback/tick")["state"]
        self.assertEqual(paused_tick["step"], paused_step)

        seeked = app.dispatch_api("POST", "/api/playback/seek", {"step": 44})["state"]
        self.assertEqual(seeked["step"], 44)
        loaded = app.dispatch_api("POST", "/api/playback/load", {"trace_id": "mixed_stress"})["state"]
        self.assertEqual(loaded["loaded_trace_id"], "mixed_stress")
        self.assertEqual(loaded["step"], 0)

    def test_v8_static_export_contains_no_map_product_dashboard_and_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "demo_v9" / "index.html"
            control_room.export_static(ROOT, output)
            base = output.parent
            html = output.read_text(encoding="utf-8")
            app_js = (base / "app.js").read_text(encoding="utf-8")

            for path in [
                base / "index.html",
                base / "app.js",
                base / "styles.css",
                base / "traces" / "baseline_normal_trace.json",
                base / "traces" / "mixed_stress_trace.json",
                base / "traces" / "route_disruption_congestion_trace.json",
            ]:
                self.assertTrue(path.exists(), str(path))
            trace = json.loads((base / "traces" / "baseline_normal_trace.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(trace["steps"]), 180)
            for marker in [
                "20260616v9",
                "5PL Dijital İkiz Karar Orkestrasyon Paneli",
                "Bu statik önizlemedir; canlı dashboard için start_control_room.bat çalıştırın.",
                "Operasyon Özeti",
                "Operasyon Akışı",
                "PPO Sürekli Kontroller",
                "DQN Karar Kartı",
                "48 Aksiyon Dağılımı",
                "KPI Zaman Çizgisi",
                "Senaryo Karşılaştırması",
                "Public Replay Analizi",
                "Benchmark / Kanıt Skor Kartı",
                "Olay Akışı",
                "Şirket Verisi Gereksinimleri",
                "Sınırlar ve İddia Kontrolü",
                "actionSearch",
                "dispatchFilter",
                "routeFilter",
                "fleetFilter",
                "reorderFilter",
                "timelineSlider",
                "ppoHistory",
                "ppoSource",
                "decisionEventId",
                "scorecardGrid",
                "companyRequirements",
            ]:
                self.assertIn(marker, html)
            combined = html + "\n" + app_js
            for forbidden_map_marker in [
                "mapSvg",
                "renderMap",
                "grid-line",
                "map-card",
                "Dijital İkiz Haritası",
                "Stres bölgelerini göster",
                "route polyline",
                "coordinate grid",
            ]:
                self.assertNotIn(forbidden_map_marker, combined)
            for forbidden_debug_marker in ["<pre", "<table", str(ROOT), "raw JSON", "debug table"]:
                self.assertNotIn(forbidden_debug_marker.lower(), combined.lower())

    def test_v9_static_and_live_mode_copy_are_polished_and_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            live_dir = base / "live"
            static_output = base / "static" / "index.html"

            control_room.write_static_assets(live_dir, root=ROOT)
            control_room.export_static(ROOT, static_output)

            live_html = (live_dir / "index.html").read_text(encoding="utf-8")
            static_html = static_output.read_text(encoding="utf-8")
            static_app = (static_output.parent / "app.js").read_text(encoding="utf-8")

        self.assertIn("20260616v9", live_html)
        self.assertIn("Demo Akışı", live_html)
        self.assertIn("Teknik Simülatör", live_html)
        self.assertIn("Kamu Replay", live_html)
        self.assertNotIn("Sunum Modu", live_html)
        self.assertNotIn("Operasyon Kaydı Oynatıcı", live_html)
        self.assertNotIn("Canlı Simülasyon Modu", live_html)
        self.assertNotIn("Güvenli Görsel Demo", live_html)

        static_notice = "Bu statik önizlemedir; canlı dashboard için start_control_room.bat çalıştırın."
        self.assertNotIn(static_notice, live_html)
        self.assertIn(static_notice, static_html)
        self.assertIn('class="static-preview"', static_html)
        self.assertIn("Demo Akışı", static_html)

        combined = live_html + "\n" + static_html + "\n" + static_app
        self.assertIn("demo trace", combined)
        self.assertIn("simülatör metriği", combined)
        self.assertIn("public replay metriği", combined)
        self.assertIn("Henüz hesaplanmadı", combined)
        self.assertIn("Başlangıç", combined)
        self.assertIn("çok düşük olay", combined)

    def test_v9_primary_ui_polish_labels_and_boundaries_are_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "demo_v9" / "index.html"
            control_room.export_static(ROOT, output)
            html = output.read_text(encoding="utf-8")
            app_js = (output.parent / "app.js").read_text(encoding="utf-8")

        combined = html + "\n" + app_js
        for required in [
            "Operasyon Akışı",
            "Bekleyen",
            "Karar verildi",
            "Sevk edildi",
            "Tamamlandı",
            "Gecikti / riskli",
            "PPO çevrimiçi öğrenmez; karar anında mevcut gözleme göre beş sürekli kontrol çıktısı üretir.",
            "DQN, 48 taktik aksiyon içinden dispatch/rota/filo/ikmal bileşimini seçer.",
            "Yeni karar yok; önceki karar uygulanıyor.",
            "Bu skorlar kanıt olgunluğudur; global üstünlük skoru değildir.",
            "score / 5",
            "Public replay olgunluğu",
            "Filo/dispatch kanıtı",
            "Simülatör-production hazırlığı",
            "Eski production karşılaştırması",
            "Rota/stokastik kanıt",
            "Şirket verisi doğrulama hazırlığı",
            "Araştırma yolu olgunluğu",
            "Gerçek dünya devreye alma hazırlığı",
            "Dispatch: beklet",
            "Rota ailesi",
            "Filo modu",
            "İkmal modu",
            "Filo / taşıyıcı",
            "Anonim join anahtarları",
        ]:
            self.assertIn(required, combined)

        for forbidden_visible in [
            "public historical replay maturity",
            "fleet/dispatch evidence",
            "old-production comparison",
            "route/stochastic evidence",
            "company-data validation readiness",
            "research pathway maturity",
            "real-world deployment readiness",
            "row.dimension",
            "JSON.stringify(a)",
        ]:
            self.assertNotIn(forbidden_visible, combined)

        for forbidden_phrase in [
            "decision-support system",
            "live company deployment",
            "company-data validation is complete",
            "public replay proves",
            "causal public replay superiority",
            "global state-of-the-art",
        ]:
            self.assertNotIn(forbidden_phrase, combined.lower())

    def test_v9_dashboard_payload_defaults_to_demo_flow_and_operation_pipeline(self) -> None:
        app = control_room.ControlRoomApp(ROOT)
        state = app.get_state()

        self.assertEqual(state["mode_label_tr"], "Demo Akışı")
        dashboard = state["dashboard"]
        self.assertEqual(
            [stage["label_tr"] for stage in dashboard["operation_pipeline"]],
            ["Bekleyen", "Karar verildi", "Sevk edildi", "Tamamlandı", "Gecikti / riskli"],
        )
        self.assertEqual(dashboard["operation_summary"]["mode_label_tr"], "Demo Akışı")
        self.assertEqual(dashboard["operation_summary"]["metric_context_tr"], "demo trace")
        self.assertEqual(dashboard["action_analytics"]["action_count"], 48)
        self.assertEqual(dashboard["public_replay"]["global_total"], 227_891)

    def test_v7_trace_binds_decisions_to_assignments_and_movement_frames(self) -> None:
        trace = control_room.build_operation_trace(ROOT, "mixed_stress", steps=72)
        rows = trace["steps"]

        frame_types = {row["frame_type"] for row in rows}
        self.assertIn("decision_event", frame_types)
        self.assertIn("movement_frame", frame_types)

        decision_rows = [row for row in rows if row["frame_type"] == "decision_event"]
        self.assertTrue(decision_rows)
        for row in decision_rows:
            assignment = next(item for item in row["assignments"] if item["assignment_id"] == row["active_assignment_id"])
            self.assertEqual(row["dqn_action_id"], assignment["bound_dqn_action_id"])
            self.assertEqual(row["ppo_controls"], assignment["bound_ppo_controls"])
            self.assertEqual(set(assignment["bound_ppo_controls"]), set(control_room.PPO_CONTROLS))

        for previous, current in zip(rows, rows[1:], strict=False):
            current_assignment = next(
                item for item in current["assignments"] if item["assignment_id"] == current["active_assignment_id"]
            )
            if current["frame_type"] == "movement_frame" and current["active_assignment_id"] == previous["active_assignment_id"]:
                previous_assignment = next(
                    item for item in previous["assignments"] if item["assignment_id"] == previous["active_assignment_id"]
                )
                self.assertEqual(current_assignment["bound_dqn_action_id"], previous_assignment["bound_dqn_action_id"])
                self.assertEqual(current["dqn_action_id"], previous["dqn_action_id"])
                self.assertEqual(current["ppo_controls"], previous["ppo_controls"])

    def test_v7_state_panel_uses_active_assignment_bound_decision(self) -> None:
        app = control_room.ControlRoomApp(ROOT)
        decision_state = app.dispatch_api("POST", "/api/playback/seek", {"step": 18})["state"]
        movement_state = app.dispatch_api("POST", "/api/playback/seek", {"step": 19})["state"]

        self.assertEqual(decision_state["frame_type"], "decision_event")
        self.assertEqual(movement_state["frame_type"], "movement_frame")
        self.assertEqual(decision_state["active_assignment_id"], movement_state["active_assignment_id"])

        panel = movement_state["decision_panel"]
        self.assertIn(panel["panel_title_tr"], {"Son karar olayı", "Seçili aktif operasyon"})
        self.assertEqual(panel["assignment_id"], movement_state["active_assignment_id"])
        self.assertEqual(panel["dqn_action_id"], movement_state["dqn_action"]["action_id"])
        self.assertEqual(panel["dqn_action_id"], decision_state["decision_panel"]["dqn_action_id"])
        self.assertEqual(panel["bound_ppo_controls"], movement_state["ppo_controls"])
        self.assertIn("karar anında inference", movement_state["ppo_explanation_tr"])
        self.assertEqual(movement_state["ppo_source_label_tr"], "Kaynak: deterministik sunum izi")

    def test_v7_event_log_and_action_history_are_operational_turkish_text(self) -> None:
        trace = control_room.build_operation_trace(ROOT, "route_disruption_congestion", steps=36)
        decision = next(row for row in trace["steps"] if row["frame_type"] == "decision_event")
        movement = next(row for row in trace["steps"] if row["frame_type"] == "movement_frame")

        decision_text = " ".join(item["text"] for item in decision["event_log"])
        movement_text = " ".join(item["text"] for item in movement["event_log"])
        self.assertIn("DQN", decision_text)
        self.assertIn("seçildi", decision_text)
        self.assertIn("yeni karar üretilmedi", movement_text)
        for chip in movement["action_history"]:
            self.assertIn("tooltip_tr", chip)
            self.assertIn("DQN", chip["tooltip_tr"])
            self.assertIn(chip["interpretation_tr"], chip["tooltip_tr"])

    def test_v7_stress_zones_are_labeled_toggleable_and_hidden_when_irrelevant(self) -> None:
        baseline = control_room.build_operation_trace(ROOT, "baseline_normal", steps=4)["steps"][0]
        route = control_room.build_operation_trace(ROOT, "route_disruption_congestion", steps=4)["steps"][0]

        self.assertFalse(baseline["stress_zones_visible"])
        self.assertEqual(baseline["congestion_zones"] + baseline["disruption_zones"], [])
        self.assertTrue(route["stress_zones_visible"])
        for zone in route["congestion_zones"] + route["disruption_zones"]:
            self.assertIn(zone["label"], {"Sıkışıklık bölgesi", "Kesinti riski"})
            self.assertIn("legend_label_tr", zone)

        app = control_room.ControlRoomApp(ROOT)
        app.dispatch_api("POST", "/api/playback/load", {"trace_id": "route_disruption_congestion"})
        hidden = app.dispatch_api("POST", "/api/playback/set_stress_zones", {"enabled": False})["state"]
        self.assertFalse(hidden["stress_zones_visible"])
        self.assertEqual(hidden["zones"], [])

        shown = app.dispatch_api("POST", "/api/playback/set_stress_zones", {"enabled": True})["state"]
        self.assertTrue(shown["stress_zones_visible"])
        self.assertGreater(len(shown["zones"]), 0)

    def test_v7_assignment_api_is_read_only_and_select_assignment_is_explicit(self) -> None:
        app = control_room.ControlRoomApp(ROOT)
        state = app.dispatch_api("POST", "/api/playback/seek", {"step": 18})["state"]
        assignment_id = state["active_assignment_id"]

        first = app.dispatch_api("GET", f"/api/assignment/{assignment_id}")["assignment"]
        second = app.dispatch_api("GET", f"/api/assignment/{assignment_id}")["assignment"]
        self.assertEqual(first, second)
        self.assertEqual(app.get_state()["step"], 18)

        selected = app.dispatch_api("POST", "/api/playback/select_assignment", {"assignment_id": assignment_id})["state"]
        self.assertEqual(selected["selected_assignment_id"], assignment_id)
        self.assertEqual(selected["decision_panel"]["assignment_id"], assignment_id)

    def test_run_server_uses_v9_static_directory(self) -> None:
        captured: dict[str, object] = {}

        class FakeServer:
            def __init__(self, address: tuple[str, int], handler: object) -> None:
                captured["address"] = address
                captured["handler"] = handler

            def serve_forever(self) -> None:
                raise KeyboardInterrupt()

            def server_close(self) -> None:
                captured["closed"] = True

        def fake_write_static_assets(target_dir: Path, *, root: Path | None = None) -> None:
            captured["static_dir"] = target_dir
            captured["root"] = root

        with (
            mock.patch.object(control_room, "write_static_assets", side_effect=fake_write_static_assets),
            mock.patch.object(control_room, "ThreadingHTTPServer", FakeServer),
            mock.patch.object(control_room, "find_available_port", return_value=9876),
        ):
            result = control_room.run_server(ROOT, port=9876, open_browser=False)

        self.assertEqual(result, 0)
        self.assertEqual(captured["static_dir"], ROOT / "reports" / "demo_control_room_v9")
        self.assertEqual(captured["root"], ROOT)
        self.assertTrue(captured["closed"])

    def test_scenario_discovery_includes_configs_and_aliases(self) -> None:
        scenarios = control_room.discover_scenarios(ROOT)
        by_id = {row["scenario_id"]: row for row in scenarios}
        config_ids = {path.stem for path in (ROOT / "configs" / "eval_scenarios").glob("*.json")}

        self.assertTrue(config_ids)
        self.assertTrue(config_ids.issubset(by_id))
        for alias in control_room.DOCUMENTED_SCENARIO_ALIASES:
            self.assertIn(alias, by_id)
        self.assertEqual(by_id["baseline_normal"]["status"], "runnable")

    def test_action_decoder_and_public_replay_totals(self) -> None:
        actions = control_room.build_action_rows(ROOT)
        replay = control_room.load_public_replay(ROOT)

        self.assertEqual(len(actions), 48)
        self.assertEqual({row["action_id"] for row in actions}, set(range(48)))
        self.assertEqual(actions[24]["watch_tr"], "izlenen aksiyon")
        self.assertEqual(actions[32]["watch_tr"], "izlenen aksiyon")
        self.assertEqual(replay["global_total"], 227_891)
        self.assertEqual(replay["dataset_totals"]["LaDe"], 31_415)
        self.assertEqual(replay["dataset_totals"]["NYC HVFHS"], 100_000)
        self.assertEqual(replay["dataset_totals"]["Olist"], 96_476)
        self.assertEqual(len(replay["actions"]), 48)

    def test_control_room_state_advances_and_contains_orchestration_dashboard_entities(self) -> None:
        app = control_room.ControlRoomApp(ROOT)
        initial = app.get_state()
        app.control_step()
        advanced = app.get_state()

        self.assertEqual(initial["running"], False)
        self.assertGreater(advanced["step"], initial["step"])
        self.assertIn(advanced["mode"], control_room.MODES)
        self.assertIn("dashboard", advanced)
        dashboard = advanced["dashboard"]
        self.assertTrue(dashboard["operation_pipeline"])
        self.assertEqual(
            [stage["label_tr"] for stage in dashboard["operation_pipeline"]],
            ["Bekleyen", "Karar verildi", "Sevk edildi", "Tamamlandı", "Gecikti / riskli"],
        )
        self.assertTrue(dashboard["decision_timeline"])
        self.assertTrue(dashboard["kpi_timeline"])
        self.assertTrue(dashboard["scenario_comparison"])
        self.assertEqual(dashboard["action_analytics"]["action_count"], 48)
        self.assertEqual(dashboard["public_replay"]["global_total"], 227_891)
        self.assertIn("ppo_controls", advanced)
        self.assertEqual(set(advanced["ppo_controls"]), set(control_room.PPO_CONTROLS))
        self.assertTrue(0 <= advanced["dqn_action"]["action_id"] < 48)
        self.assertTrue(advanced["event_log"])

    def test_start_pause_reset_controls_change_status(self) -> None:
        app = control_room.ControlRoomApp(ROOT)

        app.control_start()
        self.assertTrue(app.get_state()["running"])
        app.tick_if_running()
        self.assertGreater(app.get_state()["step"], 0)

        app.control_pause()
        paused_step = app.get_state()["step"]
        app.tick_if_running()
        self.assertFalse(app.get_state()["running"])
        self.assertEqual(app.get_state()["step"], paused_step)

        app.control_reset()
        self.assertFalse(app.get_state()["running"])
        self.assertEqual(app.get_state()["step"], 0)

    def test_control_settings_for_scenario_speed_and_mode(self) -> None:
        app = control_room.ControlRoomApp(ROOT)

        app.set_scenario("mixed_stress")
        app.set_speed("5x")
        app.set_mode(control_room.CANLI_SIMULASYON_MODU)
        state = app.get_state()

        self.assertEqual(state["selected_scenario"], "mixed_stress")
        self.assertEqual(state["speed"], "5x")
        self.assertEqual(state["mode"], control_room.CANLI_SIMULASYON_MODU)
        self.assertEqual(state["mode_label_tr"], "Teknik Simülatör")

    def test_api_dispatch_returns_json_payloads(self) -> None:
        app = control_room.ControlRoomApp(ROOT)

        status = app.dispatch_api("GET", "/api/status")
        scenarios = app.dispatch_api("GET", "/api/scenarios")
        actions = app.dispatch_api("GET", "/api/actions")
        replay = app.dispatch_api("GET", "/api/replay")
        scorecard = app.dispatch_api("GET", "/api/scorecard")
        requirements = app.dispatch_api("GET", "/api/company-data-requirements")
        step = app.dispatch_api("POST", "/api/control/step")

        self.assertEqual(status["status"], "ok")
        self.assertTrue(scenarios["scenarios"])
        self.assertEqual(len(actions["actions"]), 48)
        self.assertEqual(replay["replay"]["global_total"], 227_891)
        self.assertEqual(replay["analytics"]["action_count"], 48)
        self.assertIn(24, replay["analytics"]["watch_action_ids"])
        self.assertIn(32, replay["analytics"]["watch_action_ids"])
        self.assertIn("dispatch", replay["analytics"]["filters"])
        self.assertIn("route_family", replay["analytics"]["filters"])
        self.assertEqual(scorecard["scorecard"]["classification"], "FINAL_INTEGRATED_SCORECARD_READY_WITH_LIMITATIONS")
        dimensions = {row["dimension"] for row in scorecard["scorecard"]["dimensions"]}
        self.assertIn("simulator-production readiness", dimensions)
        self.assertIn("real-world deployment readiness", dimensions)
        self.assertIn("orders", {row["family"] for row in requirements["requirements"]["table_families"]})
        self.assertIn("decision_timestamps", {row["family"] for row in requirements["requirements"]["table_families"]})
        self.assertGreater(step["state"]["step"], 0)
        self.assertIsInstance(json.dumps(step, ensure_ascii=False), str)

    def test_v8_get_state_is_read_only_and_tick_advances_dashboard_step(self) -> None:
        app = control_room.ControlRoomApp(ROOT)

        app.control_start()
        first = app.dispatch_api("GET", "/api/state")["state"]
        second = app.dispatch_api("GET", "/api/state")["state"]
        ticked = app.dispatch_api("POST", "/api/control/tick")["state"]

        self.assertEqual(first["step"], 0)
        self.assertEqual(second["step"], 0)
        self.assertEqual(ticked["step"], 1)
        self.assertIn("dashboard", ticked)
        self.assertTrue(ticked["dashboard"]["kpi_timeline"])

    def test_static_preview_and_primary_ui_are_turkish_and_not_debug_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "index.html"
            control_room.export_static(ROOT, output)
            html = output.read_text(encoding="utf-8")

        for label in [
            "5PL Dijital İkiz Karar Orkestrasyon Paneli",
            "Operasyon Özeti",
            "Operasyon Akışı",
            "PPO Sürekli Kontroller",
            "DQN Karar Kartı",
            "48 Aksiyon Dağılımı",
            "KPI Zaman Çizgisi",
            "Senaryo Karşılaştırması",
            "Public Replay Analizi",
            "Benchmark / Kanıt Skor Kartı",
            "Olay Akışı",
            "Şirket Verisi Gereksinimleri",
            "Sınırlar ve İddia Kontrolü",
            "Oynat",
            "Duraklat",
            "Baştan al",
            "Tek adım",
            "Demo Akışı",
            "Teknik Simülatör",
            "Kamu Replay",
            "Kamu replay betimleyici proxy analizdir",
        ]:
            self.assertIn(label, html)
        for technical_marker in [
            "connectionStatus",
            "tabWarning",
            "errorPanel",
            "20260616v9",
            "decisionEventId",
            "timelineSlider",
        ]:
            self.assertIn(technical_marker, html)
        for removed_marker in [
            "Dijital İkiz Haritası",
            "mapSvg",
            "stressZoneToggle",
            "Stres bölgelerini göster",
            "Geri sar",
            "İleri sar",
            "Sunum Modu",
            "Operasyon Kaydı Oynatıcı",
            "Canlı Simülasyon Modu",
            "Güvenli Görsel Demo",
        ]:
            self.assertNotIn(removed_marker, html)

        for forbidden in control_room.FORBIDDEN_PHRASES:
            self.assertNotIn(forbidden, html.lower())
        self.assertNotIn("Streamlit", html)
        self.assertNotIn("Choose options", html)
        self.assertNotIn(str(ROOT), html)
        self.assertNotIn("<pre", html.lower())

    def test_static_assets_and_launchers_exist(self) -> None:
        for path in [
            ROOT / "reports" / "demo_control_room_v4" / "index.html",
            ROOT / "reports" / "demo_control_room_v4" / "app.js",
            ROOT / "reports" / "demo_control_room_v4" / "styles.css",
            ROOT / "reports" / "demo_control_room_v5" / "index.html",
            ROOT / "reports" / "demo_control_room_v5" / "app.js",
            ROOT / "reports" / "demo_control_room_v5" / "styles.css",
            ROOT / "reports" / "demo_control_room_v6" / "index.html",
            ROOT / "reports" / "demo_control_room_v6" / "app.js",
            ROOT / "reports" / "demo_control_room_v6" / "styles.css",
            ROOT / "reports" / "demo_control_room_v7" / "index.html",
            ROOT / "reports" / "demo_control_room_v7" / "app.js",
            ROOT / "reports" / "demo_control_room_v7" / "styles.css",
            ROOT / "reports" / "demo_control_room_v8" / "index.html",
            ROOT / "reports" / "demo_control_room_v8" / "app.js",
            ROOT / "reports" / "demo_control_room_v8" / "styles.css",
            ROOT / "start_control_room.bat",
            ROOT / "start_control_room.ps1",
        ]:
            self.assertTrue(path.exists(), str(path))

    def test_batch_launcher_is_ascii_only_and_has_no_unquoted_text_fragments(self) -> None:
        path = ROOT / "start_control_room.bat"
        raw = path.read_bytes()

        self.assertTrue(raw)
        self.assertTrue(all(byte < 128 for byte in raw), "BAT launcher must stay ASCII-only")
        text = raw.decode("ascii")

        forbidden_fragments = ["tfen", "ütfen", "kontrol", "Başlatma", "N_CMD", "orlevel"]
        for fragment in forbidden_fragments:
            self.assertNotIn(fragment, text)

        py_probe = text.index("where py")
        python_probe = text.index("where python")
        self.assertLess(py_probe, python_probe)

        allowed_prefixes = (
            "@echo ",
            "setlocal",
            "cd ",
            "set ",
            "where ",
            "if ",
            "echo",
            "pause",
            "exit ",
            "%py_run%",
            "rem ",
            ")",
        )
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped == "(":
                continue
            self.assertTrue(
                stripped.lower().startswith(allowed_prefixes),
                f"Unexpected BAT line, likely un-echoed text: {stripped!r}",
            )


if __name__ == "__main__":
    unittest.main()
