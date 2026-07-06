from __future__ import annotations

import argparse
import json
import math
import mimetypes
import socket
import sys
import threading
import time
import webbrowser
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.act.discrete_action_mapper import DiscreteActionMapper


TITLE_TR = "5PL Dijital İkiz Karar Orkestrasyon Paneli"
MODEL_ID = "joint_torch_v5_prod_hierarchical_v1_1m_20260611"
MODEL_ID_SHORT = "hierarchical_v1_1m"
CONTRACT = "physical_reality_v5_route_candidate_visibility"
DEFAULT_PORT = 8765
DEFAULT_SCENARIO = "baseline_normal"
DEFAULT_TRAINING_CONFIG = "training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json"
STATIC_VERSION = "20260616v9"

SUNUM_MODU = "SUNUM_MODU"
OPERASYON_KAYDI_OYNATICI = "OPERASYON_KAYDI_OYNATICI"
CANLI_SIMULASYON_MODU = "CANLI_SIMULASYON_MODU"
GUVENLI_GORSEL_DEMO_MODU = "GUVENLI_GORSEL_DEMO_MODU"
KAMU_REPLAY_MODU = "KAMU_REPLAY_MODU"
MODES = (SUNUM_MODU, CANLI_SIMULASYON_MODU, GUVENLI_GORSEL_DEMO_MODU, KAMU_REPLAY_MODU)
MODE_LABELS_TR = {
    SUNUM_MODU: "Demo Akışı",
    CANLI_SIMULASYON_MODU: "Teknik Simülatör",
    GUVENLI_GORSEL_DEMO_MODU: "Demo Akışı",
    KAMU_REPLAY_MODU: "Kamu Replay",
}
PRESENTATION_NOTICE_TR = "Demo Akışı: deterministik operasyon izi oynatılır; canlı şirket verisi değildir."
PRESENTATION_TRACE_SCENARIOS = (
    "baseline_normal",
    "mixed_stress",
    "route_disruption_congestion",
)
OPERATION_TRACE_STEPS = 180
DECISION_EVENT_INTERVAL = 18
PPO_CONTROLS = [
    "reorder_fraction",
    "dispatch_intensity",
    "speed_multiplier",
    "safety_stock_multiplier",
    "capacity_buffer_fraction",
]
DOCUMENTED_SCENARIO_ALIASES = [
    "baseline_normal",
    "demand_spike_volatility",
    "premium_sla_pressure",
    "route_disruption_congestion",
    "vehicle_scarcity_capacity_shock",
    "high_holding_cost",
    "lead_time_volatility",
    "mixed_stress",
    "low_congestion",
]
FORBIDDEN_PHRASES = [
    "decision-support system",
    "decision support system",
    "live company deployment",
    "company-data validation is complete",
    "public replay proves",
    "causal public replay superiority",
    "global state-of-the-art",
]
PUBLIC_REPLAY_WARNING_TR = "Kamu replay betimleyici proxy analizdir; nedensel OPE veya şirket verisi doğrulaması değildir."
MAP_COORDINATE_NOTICE_TR = "Harita koordinatları görselleştirme amaçlı sentetik koordinatlardır; gerçek şirket lokasyonu içermez."

TURKISH_DISPATCH = {"hold": "Beklet", "dispatch": "Sevk et"}
TURKISH_ROUTE = {
    "shortest": "En kısa rota",
    "low_congestion": "Düşük sıkışıklık",
    "high_resilience": "Dayanıklı rota",
}
TURKISH_FLEET = {"secondary_fleet": "İkincil filo", "primary_fleet": "Birincil filo"}
TURKISH_REORDER = {
    "none": "İkmal yok",
    "conservative": "Temkinli",
    "aggressive": "Agresif",
    "emergency": "Acil",
}


def repo_root() -> Path:
    return ROOT


def discover_scenarios(root: Path) -> list[dict[str, Any]]:
    scenario_dir = root / "configs" / "eval_scenarios"
    scenarios: dict[str, dict[str, Any]] = {}
    if scenario_dir.exists():
        for path in sorted(scenario_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            scenario_id = str(payload.get("scenario_id") or path.stem)
            overrides = payload.get("environment_overrides", {})
            scenarios[scenario_id] = {
                "scenario_id": scenario_id,
                "name_tr": scenario_name_tr(scenario_id),
                "description_tr": scenario_description_tr(scenario_id, str(payload.get("description", ""))),
                "status": "runnable",
                "status_tr": "çalıştırılabilir",
                "stress_type_tr": infer_stress_type_tr(scenario_id, overrides),
                "expected_pressure_tr": infer_expected_pressure_tr(scenario_id),
                "episode_count": int(payload.get("episode_count", 0) or 0),
                "source_kind": "scenario_config",
                "runnable": True,
            }
    for alias in DOCUMENTED_SCENARIO_ALIASES:
        scenarios.setdefault(
            alias,
            {
                "scenario_id": alias,
                "name_tr": scenario_name_tr(alias),
                "description_tr": "Dokümante senaryo alias'ı; bu çalışma alanında runnable JSON yoksa güvenli görsel demo kullanılır.",
                "status": "documented / not runnable in current workspace",
                "status_tr": "dokümante",
                "stress_type_tr": infer_stress_type_tr(alias, {}),
                "expected_pressure_tr": infer_expected_pressure_tr(alias),
                "episode_count": 0,
                "source_kind": "documented_alias",
                "runnable": False,
            },
        )
    return [scenarios[key] for key in sorted(scenarios)]


def scenario_name_tr(scenario_id: str) -> str:
    return {
        "baseline_normal": "Normal Operasyon",
        "demand_spike_volatility": "Talep Sıçraması",
        "premium_sla_pressure": "Premium SLA Baskısı",
        "route_disruption_congestion": "Rota Kesintisi ve Sıkışıklık",
        "vehicle_scarcity_capacity_shock": "Araç Kıtlığı",
        "high_holding_cost": "Yüksek Stok Maliyeti",
        "lead_time_volatility": "Tedarik Süresi Oynaklığı",
        "mixed_stress": "Karma Stres",
        "low_congestion": "Düşük Sıkışıklık Rota Baskısı",
    }.get(scenario_id, scenario_id.replace("_", " ").title())


def scenario_description_tr(scenario_id: str, fallback: str) -> str:
    return {
        "baseline_normal": "Normal 5PL akışında servis, sevk ve stok dengesini izler.",
        "demand_spike_volatility": "Talep artışı altında sevk yoğunluğu ve bekleyen sipariş basıncını gösterir.",
        "premium_sla_pressure": "Dar teslim penceresi ve premium servis baskısını öne çıkarır.",
        "route_disruption_congestion": "Rota kesintisi, sıkışıklık ve alternatif rota seçimini görünür yapar.",
        "vehicle_scarcity_capacity_shock": "Araç kıtlığı altında filo ve dispatch kararlarını gösterir.",
        "high_holding_cost": "Stok tutma maliyeti yükseldiğinde ikmal davranışını izler.",
        "lead_time_volatility": "Tedarik süresi oynaklığında güvenlik stoğu ve servis dengesini gösterir.",
        "mixed_stress": "Talep, rota, filo ve stok baskılarını aynı ekranda birleştirir.",
        "low_congestion": "Düşük sıkışıklık rota tercihini dokümante alias olarak gösterir.",
    }.get(scenario_id, fallback)


def infer_stress_type_tr(scenario_id: str, overrides: dict[str, Any]) -> str:
    text = f"{scenario_id} {' '.join(overrides)}".lower()
    if "route" in text or "congestion" in text:
        return "rota/sıkışıklık"
    if "vehicle" in text or "capacity" in text or "fleet" in text:
        return "filo/kapasite"
    if "holding" in text or "inventory" in text:
        return "stok/ikmal"
    if "lead" in text:
        return "tedarik süresi"
    if "premium" in text or "sla" in text:
        return "SLA"
    if "demand" in text:
        return "talep"
    if "mixed" in text:
        return "karma stres"
    return "normal operasyon"


def infer_expected_pressure_tr(scenario_id: str) -> str:
    return {
        "baseline_normal": "Servis yüksek, gecikme düşük, sert blokajlar sıfır kalmalı.",
        "demand_spike_volatility": "Sevk yoğunluğu ve bekleyen sipariş baskısı artar.",
        "premium_sla_pressure": "Gecikme riski ve hız kontrolü baskın hale gelir.",
        "route_disruption_congestion": "Rota ailesi ve sıkışıklık bayrakları daha belirgin olur.",
        "vehicle_scarcity_capacity_shock": "Filo bulunurluğu dispatch başarısını sınar.",
        "high_holding_cost": "İkmal ve stok tutma dengesi görünür olur.",
        "lead_time_volatility": "Güvenlik stoğu ve tedarik süresi baskısı artar.",
        "mixed_stress": "Talep, rota, filo ve stok baskısı birlikte yükselir.",
        "low_congestion": "Düşük sıkışıklık rota bileşeni izlenir.",
    }.get(scenario_id, "Senaryo baskısı canlı ekranda izlenir.")


def build_action_rows(root: Path | None = None) -> list[dict[str, Any]]:
    replay = load_public_replay(root or repo_root())
    replay_by_id = {row["action_id"]: row for row in replay["actions"]}
    mapper = DiscreteActionMapper()
    rows: list[dict[str, Any]] = []
    for action_id in range(mapper.action_count):
        decoded = mapper.map(action_id).as_dict()
        replay_row = replay_by_id.get(action_id, {})
        rows.append(
            {
                "action_id": action_id,
                "dispatch": decoded["dispatch"],
                "dispatch_tr": TURKISH_DISPATCH[decoded["dispatch"]],
                "route_family": decoded["route"],
                "route_family_tr": TURKISH_ROUTE[decoded["route"]],
                "fleet_mode": decoded["mode"],
                "fleet_mode_tr": TURKISH_FLEET[decoded["mode"]],
                "reorder_mode": decoded["reorder"],
                "reorder_mode_tr": TURKISH_REORDER[decoded["reorder"]],
                "global_count": int(replay_row.get("global_count", 0)),
                "global_rate": float(replay_row.get("global_rate", 0.0)),
                "watch_tr": "izlenen aksiyon" if action_id in {24, 32} else "",
                "interpretation_tr": action_interpretation_tr(decoded),
            }
        )
    return rows


def action_interpretation_tr(decoded: dict[str, str]) -> str:
    return (
        f"{TURKISH_DISPATCH[decoded['dispatch']]} / {TURKISH_ROUTE[decoded['route']]} / "
        f"{TURKISH_FLEET[decoded['mode']]} / {TURKISH_REORDER[decoded['reorder']]}"
    )


def load_public_replay(root: Path) -> dict[str, Any]:
    appendix = root / "Tez" / "FINAL_EKLER" / "EK_A_48_ACTION_MAPPING_AND_PUBLIC_REPLAY_COUNTS.md"
    text = appendix.read_text(encoding="utf-8") if appendix.exists() else ""
    global_total = _first_int_after(text, "Global public replay rows:", default=0)
    dataset_totals = {
        "LaDe": _first_int_after(text, "LaDe", default=0),
        "NYC HVFHS": _first_int_after(text, "NYC HVFHS", default=0),
        "Olist": _first_int_after(text, "Olist", default=0),
    }
    actions: list[dict[str, Any]] = []
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 14 or not cells[0].isdigit():
            continue
        actions.append(
            {
                "action_id": int(cells[0]),
                "dispatch": cells[1],
                "route_family": cells[2],
                "fleet_mode": cells[3],
                "reorder_mode": cells[4],
                "lade_count": _to_int(cells[5]),
                "nyc_count": _to_int(cells[7]),
                "olist_count": _to_int(cells[9]),
                "global_count": _to_int(cells[11]),
                "global_rate": float(cells[12]),
            }
        )
    family_distribution = _family_distribution(actions)
    return {
        "global_total": global_total,
        "dataset_totals": dataset_totals,
        "actions": actions,
        "top_10_actions": sorted(actions, key=lambda row: row["global_count"], reverse=True)[:10],
        "zero_count_actions": [row for row in actions if row["global_count"] == 0],
        "family_distribution": family_distribution,
        "warning_tr": PUBLIC_REPLAY_WARNING_TR,
    }


def _first_int_after(text: str, marker: str, *, default: int) -> int:
    index = text.find(marker)
    if index < 0:
        return default
    tail = text[index : index + 120]
    digits = []
    started = False
    for char in tail:
        if char.isdigit() or (started and char == ","):
            digits.append(char)
            started = True
        elif started:
            break
    return _to_int("".join(digits)) if digits else default


def _to_int(value: str) -> int:
    return int(value.replace(",", "").strip())


def _family_distribution(actions: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counters = {
        "dispatch": Counter(),
        "route_family": Counter(),
        "fleet_mode": Counter(),
        "reorder_mode": Counter(),
    }
    for row in actions:
        count = int(row["global_count"])
        counters["dispatch"][row["dispatch"]] += count
        counters["route_family"][row["route_family"]] += count
        counters["fleet_mode"][row["fleet_mode"]] += count
        counters["reorder_mode"][row["reorder_mode"]] += count
    return {key: dict(value) for key, value in counters.items()}


def build_presentation_trace(root: Path, scenario_id: str, *, steps: int = 120) -> dict[str, Any]:
    actions = build_action_rows(Path(root))
    sequence = scenario_action_sequence(scenario_id)
    trace_steps: list[dict[str, Any]] = []
    for step in range(steps):
        action = deepcopy(actions[sequence[step % len(sequence)]])
        visual = build_visual_state(scenario_id, step, action)
        active_route = next((route for route in visual["routes"] if route.get("selected")), visual["routes"][0])
        trace_steps.append(
            {
                "step": step,
                "simulated_time": simulated_time(step),
                "scenario_id": scenario_id,
                "vehicles": visual["vehicles"],
                "orders": visual["orders"],
                "routes": visual["routes"],
                "active_route": active_route,
                "ppo_controls": build_ppo_controls(scenario_id, step),
                "dqn_action_id": int(action["action_id"]),
                "decoded_action": action,
                "kpis": build_kpis(scenario_id, step),
                "event_log_line": presentation_event_line(scenario_id, step, action),
            }
        )
    return {
        "trace_version": "control_room_v5_presentation_trace_v1",
        "scenario_id": scenario_id,
        "step_count": steps,
        "deterministic": True,
        "notice_tr": PRESENTATION_NOTICE_TR,
        "steps": trace_steps,
    }


def write_golden_traces(root: Path, target_dir: Path, *, steps: int = 120) -> None:
    trace_dir = Path(target_dir) / "golden_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    for scenario_id in PRESENTATION_TRACE_SCENARIOS:
        trace = build_presentation_trace(root, scenario_id, steps=steps)
        path = trace_dir / f"{scenario_id}_trace.json"
        path.write_text(json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def build_operation_trace(root: Path, scenario_id: str, *, steps: int = OPERATION_TRACE_STEPS) -> dict[str, Any]:
    actions = build_action_rows(Path(root))
    sequence = scenario_action_sequence(scenario_id)
    hubs = _operation_hubs()
    orders_template = _operation_orders()
    routes = _operation_route_candidates(scenario_id)
    trace_steps: list[dict[str, Any]] = []
    for step in range(steps):
        assignments = _operation_assignments_for_step(actions, sequence, routes, scenario_id, step)
        active_assignment = _active_assignment_for_step(assignments)
        action = deepcopy(active_assignment["bound_decoded_action"])
        selected_route = next((route for route in routes if route["id"] == active_assignment["route_id"]), routes[0])
        vehicles = _operation_vehicles(step, routes[0], active_assignment)
        orders = _operation_orders_for_assignments(orders_template, step, assignments)
        kpis = _operation_kpis(scenario_id, step, orders)
        ppo_controls = deepcopy(active_assignment["bound_ppo_controls"])
        operation_phase = _operation_phase(step)
        operation_phase_tr = _operation_phase_label(operation_phase)
        frame_type = _operation_frame_type(scenario_id, step, active_assignment)
        latest_decision_event_id = active_assignment["decision_event_id"]
        current_decision_event_id = latest_decision_event_id if frame_type == "decision_event" else None
        event = _operation_event(scenario_id, step, frame_type, operation_phase, active_assignment, selected_route, orders)
        event_log = [{"time": simulated_time(step), "text": event}]
        history = _operation_action_history(assignments)
        stress_zones_visible = _stress_zones_visible(scenario_id)
        congestion_zones = _operation_congestion_zones(scenario_id) if stress_zones_visible else []
        disruption_zones = _operation_disruption_zones(scenario_id) if stress_zones_visible else []
        trace_steps.append(
            {
                "trace_id": scenario_id,
                "scenario_id": scenario_id,
                "step": step,
                "simulated_time": simulated_time(step),
                "frame_type": frame_type,
                "current_decision_event_id": current_decision_event_id,
                "latest_decision_event_id": latest_decision_event_id,
                "active_assignment_id": active_assignment["assignment_id"],
                "assignments": deepcopy(assignments),
                "operation_phase": operation_phase,
                "operation_phase_tr": operation_phase_tr,
                "hubs": deepcopy(hubs),
                "vehicles": vehicles,
                "orders": orders,
                "route_candidates": deepcopy(routes),
                "active_routes": [
                    {
                        **deepcopy(selected_route),
                        "route_id": selected_route["id"],
                        "assignment_id": active_assignment["assignment_id"],
                        "bound_dqn_action_id": active_assignment["bound_dqn_action_id"],
                        "selected": True,
                    }
                ],
                "congestion_zones": deepcopy(congestion_zones),
                "disruption_zones": deepcopy(disruption_zones),
                "stress_zones_visible": stress_zones_visible,
                "stress_zone_legend": _stress_zone_legend(congestion_zones, disruption_zones),
                "observation_summary": _operation_observation_summary(scenario_id, step, orders, selected_route),
                "ppo_controls": ppo_controls,
                "ppo_source": active_assignment["ppo_source"],
                "ppo_source_label_tr": "Kaynak: deterministik sunum izi",
                "dqn_action_id": int(action["action_id"]),
                "decoded_action": action,
                "kpis": kpis,
                "event_log": event_log,
                "event_log_line": event,
                "action_history": history,
            }
        )
    return {
        "trace_version": "control_room_v7_assignment_bound_trace_v1",
        "trace_id": scenario_id,
        "scenario_id": scenario_id,
        "scenario_name_tr": scenario_name_tr(scenario_id),
        "playback_mode": OPERASYON_KAYDI_OYNATICI,
        "step_count": steps,
        "deterministic": True,
        "ppo_explanation_tr": "PPO çevrimiçi öğrenmez; karar anında inference çıktısı üretir ve hareket karelerinde bağlı atama üzerinde taşınır.",
        "notice_tr": PRESENTATION_NOTICE_TR,
        "steps": trace_steps,
    }


def write_operation_traces(root: Path, target_dir: Path, *, steps: int = OPERATION_TRACE_STEPS) -> None:
    trace_dir = Path(target_dir) / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    for scenario_id in PRESENTATION_TRACE_SCENARIOS:
        trace = build_operation_trace(root, scenario_id, steps=steps)
        path = trace_dir / f"{scenario_id}_trace.json"
        path.write_text(json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _operation_hubs() -> list[dict[str, Any]]:
    return [
        {"id": "H1", "x": 14.0, "y": 24.0, "label": "Kuzey Hub"},
        {"id": "H2", "x": 78.0, "y": 72.0, "label": "Güney Hub"},
    ]


def _operation_orders() -> list[dict[str, Any]]:
    return [
        {"id": "S-201", "x": 32.0, "y": 18.0, "priority": "standart", "release_step": 0},
        {"id": "S-204", "x": 58.0, "y": 29.0, "priority": "premium", "release_step": 8},
        {"id": "S-309", "x": 45.0, "y": 82.0, "priority": "acil", "release_step": 16},
        {"id": "S-411", "x": 86.0, "y": 47.0, "priority": "standart", "release_step": 24},
        {"id": "S-512", "x": 22.0, "y": 74.0, "priority": "premium", "release_step": 32},
        {"id": "S-618", "x": 67.0, "y": 88.0, "priority": "standart", "release_step": 40},
    ]


def _operation_route_candidates(scenario_id: str) -> list[dict[str, Any]]:
    stress = scenario_stress_profile(scenario_id)
    return [
        {
            "id": "R-short",
            "family": "shortest",
            "family_tr": "En kısa rota",
            "path": _points([(14, 24), (31, 28), (50, 42), (78, 72)]),
            "congestion": "orta" if stress["rota"] < 0.6 else "yüksek",
        },
        {
            "id": "R-low",
            "family": "low_congestion",
            "family_tr": "Düşük sıkışıklık",
            "path": _points([(14, 24), (21, 42), (37, 58), (58, 67), (78, 72)]),
            "congestion": "düşük",
        },
        {
            "id": "R-res",
            "family": "high_resilience",
            "family_tr": "Dayanıklı rota",
            "path": _points([(14, 24), (28, 16), (54, 24), (72, 48), (78, 72)]),
            "congestion": "orta",
        },
    ]


def _points(values: list[tuple[float, float]]) -> list[dict[str, float]]:
    return [{"x": float(x), "y": float(y)} for x, y in values]


def _operation_congestion_zones(scenario_id: str) -> list[dict[str, Any]]:
    stress = scenario_stress_profile(scenario_id)
    if not _stress_zones_visible(scenario_id):
        return []
    return [
        {
            "id": "CZ-1",
            "x": 58.0,
            "y": 44.0,
            "r": round(6.5 + stress["rota"] * 7.0, 2),
            "label": "Sıkışıklık bölgesi",
            "legend_label_tr": "Sıkışıklık bölgesi",
            "kind": "congestion",
        },
    ]


def _operation_disruption_zones(scenario_id: str) -> list[dict[str, Any]]:
    stress = scenario_stress_profile(scenario_id)
    if not _stress_zones_visible(scenario_id) or stress["rota"] < 0.55:
        return []
    return [
        {
            "id": "DZ-1",
            "x": 50.0,
            "y": 41.0,
            "r": round(4.5 + stress["rota"] * 3.5, 2),
            "label": "Kesinti riski",
            "legend_label_tr": "Kesinti riski",
            "kind": "disruption",
        }
    ]


def _stress_zones_visible(scenario_id: str) -> bool:
    if scenario_id == "baseline_normal":
        return False
    stress = scenario_stress_profile(scenario_id)
    return stress["rota"] >= 0.5 or stress["talep"] >= 0.7


def _stress_zone_legend(
    congestion_zones: list[dict[str, Any]], disruption_zones: list[dict[str, Any]]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if congestion_zones:
        rows.append({"kind": "congestion", "label_tr": "Sıkışıklık bölgesi"})
    if disruption_zones:
        rows.append({"kind": "disruption", "label_tr": "Kesinti riski"})
    return rows


def _operation_assignments_for_step(
    actions: list[dict[str, Any]],
    sequence: list[int],
    routes: list[dict[str, Any]],
    scenario_id: str,
    step: int,
) -> list[dict[str, Any]]:
    order_cycle = ["S-201", "S-204", "S-309", "S-411", "S-512", "S-618"]
    vehicle_cycle = ["A1", "A2", "A3", "A4"]
    last_decision_index = step // DECISION_EVENT_INTERVAL
    assignments: list[dict[str, Any]] = []
    for decision_index in range(last_decision_index + 1):
        created_step = decision_index * DECISION_EVENT_INTERVAL
        action = deepcopy(actions[sequence[decision_index % len(sequence)]])
        route = next((item for item in routes if item["family"] == action["route_family"]), routes[0])
        order_id = order_cycle[decision_index % len(order_cycle)]
        vehicle_id = vehicle_cycle[decision_index % len(vehicle_cycle)]
        age = step - created_step
        completed_step = created_step + DECISION_EVENT_INTERVAL - 1
        if age <= 2:
            status = "assigned"
        elif age < DECISION_EVENT_INTERVAL - 1:
            status = "in_transit"
        elif scenario_id == "route_disruption_congestion" and decision_index % 3 == 2:
            status = "delayed"
        else:
            status = "delivered"
        bound_orders = _operation_orders_for_assignment_preview(order_id, "assigned")
        ppo_controls = build_trace_ppo_controls(scenario_id, created_step, bound_orders, route)
        assignments.append(
            {
                "assignment_id": f"{scenario_id}-ASG-{decision_index + 1:02d}",
                "decision_event_id": f"{scenario_id}-DEC-{decision_index + 1:02d}",
                "order_id": order_id,
                "vehicle_id": vehicle_id,
                "route_id": route["id"],
                "route_family_tr": route["family_tr"],
                "created_step": created_step,
                "completed_step": completed_step if step >= completed_step else None,
                "bound_dqn_action_id": int(action["action_id"]),
                "bound_decoded_action": action,
                "bound_ppo_controls": ppo_controls,
                "ppo_source": "deterministic_demo_policy",
                "status": status,
                "status_tr": _assignment_status_tr(status),
                "last_decision_step": created_step,
            }
        )
    return assignments


def _operation_orders_for_assignment_preview(order_id: str, status: str) -> list[dict[str, Any]]:
    return [{"id": order_id, "status": status}]


def _assignment_status_tr(status: str) -> str:
    return {
        "assigned": "atanmış",
        "in_transit": "yolda",
        "delivered": "teslim edildi",
        "delayed": "gecikmiş",
        "cancelled": "iptal edildi",
    }.get(status, status)


def _active_assignment_for_step(assignments: list[dict[str, Any]]) -> dict[str, Any]:
    for assignment in reversed(assignments):
        if assignment["status"] in {"assigned", "in_transit", "delayed"}:
            return assignment
    return assignments[-1]


def _operation_orders_for_assignments(
    orders_template: list[dict[str, Any]], step: int, assignments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    latest_by_order = {assignment["order_id"]: assignment for assignment in assignments}
    rows: list[dict[str, Any]] = []
    for order in orders_template:
        assignment = latest_by_order.get(order["id"])
        if assignment is None:
            status = "waiting"
            assigned_vehicle = ""
            assignment_id = ""
        else:
            status = assignment["status"]
            assigned_vehicle = assignment["vehicle_id"]
            assignment_id = assignment["assignment_id"]
        status_tr = {
            "waiting": "bekleyen",
            "assigned": "atanmış",
            "in_transit": "yolda",
            "delivered": "teslim edildi",
            "delayed": "gecikmiş",
            "cancelled": "iptal edildi",
        }[status]
        row = dict(order)
        row.update(
            {
                "status": status,
                "status_tr": status_tr,
                "assigned_vehicle": assigned_vehicle,
                "assignment_id": assignment_id,
                "pulse": status in {"waiting", "assigned"},
            }
        )
        rows.append(row)
    return rows


def _operation_orders_for_step(
    orders_template: list[dict[str, Any]], step: int, vehicles: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    vehicle_by_order = {vehicle["target_order"]: vehicle["id"] for vehicle in vehicles}
    rows: list[dict[str, Any]] = []
    for index, order in enumerate(orders_template):
        age = step - int(order["release_step"])
        if age < 0:
            status = "waiting"
        elif age < 18:
            status = "assigned"
        elif age < 58:
            status = "in_transit"
        elif age < 122:
            status = "delivered"
        else:
            status = "delayed" if index in {2, 4} and step % 36 > 24 else "delivered"
        status_tr = {
            "waiting": "bekliyor",
            "assigned": "atandı",
            "in_transit": "yolda",
            "delivered": "teslim edildi",
            "delayed": "gecikti",
        }[status]
        row = dict(order)
        row.update(
            {
                "status": status,
                "status_tr": status_tr,
                "assigned_vehicle": vehicle_by_order.get(order["id"], ""),
                "pulse": status in {"waiting", "assigned"},
            }
        )
        rows.append(row)
    return rows


def _operation_vehicles(step: int, route: dict[str, Any], active_assignment: dict[str, Any]) -> list[dict[str, Any]]:
    route_path = route["path"]
    vehicles: list[dict[str, Any]] = []
    for index, vehicle_id in enumerate(["A1", "A2", "A3", "A4"]):
        phase = ((step * (0.010 + index * 0.0018)) + index * 0.21) % 2.0
        progress = phase if phase <= 1.0 else 2.0 - phase
        point = _point_on_polyline(route_path, progress)
        is_active = vehicle_id == active_assignment["vehicle_id"]
        vehicles.append(
            {
                "id": vehicle_id,
                "x": round(point["x"], 2),
                "y": round(point["y"], 2),
                "fleet_tr": "Birincil filo" if index in {0, 3} else "İkincil filo",
                "target_order": active_assignment["order_id"] if is_active else "",
                "assignment_id": active_assignment["assignment_id"] if is_active else "",
                "status": "moving" if progress > 0.04 else "at_hub",
                "status_tr": "rotada" if progress > 0.04 else "hub'da",
                "route_id": active_assignment["route_id"] if is_active else route["id"],
                "bound_dqn_action_id": active_assignment["bound_dqn_action_id"] if is_active else None,
                "action_badge_tr": f"DQN {active_assignment['bound_dqn_action_id']}" if is_active else "",
                "route_progress": round(progress, 3),
            }
        )
    return vehicles


def _point_on_polyline(path: list[dict[str, float]], progress: float) -> dict[str, float]:
    progress = _clamp(progress, 0.0, 1.0)
    lengths: list[float] = []
    total = 0.0
    for start, end in zip(path, path[1:], strict=False):
        length = math.hypot(end["x"] - start["x"], end["y"] - start["y"])
        lengths.append(length)
        total += length
    if total <= 0.0:
        return dict(path[0])
    target = progress * total
    travelled = 0.0
    for index, length in enumerate(lengths):
        if travelled + length >= target:
            local = (target - travelled) / length if length else 0.0
            start = path[index]
            end = path[index + 1]
            return {
                "x": start["x"] + ((end["x"] - start["x"]) * local),
                "y": start["y"] + ((end["y"] - start["y"]) * local),
            }
        travelled += length
    return dict(path[-1])


def build_trace_ppo_controls(
    scenario_id: str, step: int, orders: list[dict[str, Any]], active_route: dict[str, Any]
) -> dict[str, float]:
    stress = scenario_stress_profile(scenario_id)
    waiting = sum(1 for order in orders if order["status"] in {"waiting", "assigned"})
    in_transit = sum(1 for order in orders if order["status"] == "in_transit")
    delivered = sum(1 for order in orders if order["status"] == "delivered")
    route_bias = {"shortest": 0.0, "low_congestion": 0.025, "high_resilience": 0.04}.get(active_route["family"], 0.0)
    slow_wave = math.sin(step / 18.0)
    phase_wave = math.cos(step / 29.0)
    return {
        "reorder_fraction": round(_clamp(0.10 + stress["stok"] * 0.42 + slow_wave * 0.025, 0, 1), 3),
        "dispatch_intensity": round(_clamp(0.24 + stress["talep"] * 0.42 + waiting * 0.035 - delivered * 0.012, 0, 1), 3),
        "speed_multiplier": round(_clamp(0.92 + stress["SLA"] * 0.23 + stress["rota"] * 0.07 + route_bias + phase_wave * 0.018, 0.75, 1.35), 3),
        "safety_stock_multiplier": round(_clamp(1.0 + stress["stok"] * 0.58 + waiting * 0.018 + slow_wave * 0.015, 1.0, 2.0), 3),
        "capacity_buffer_fraction": round(_clamp(0.035 + stress["filo"] * 0.34 + in_transit * 0.012 + route_bias, 0, 0.5), 3),
    }


def operation_ppo_variance_summary(trace: dict[str, Any], step: int) -> dict[str, dict[str, Any]]:
    rows = trace["steps"]
    current = rows[_trace_step_index(step, rows)]["ppo_controls"]
    summary: dict[str, dict[str, Any]] = {}
    for key in PPO_CONTROLS:
        values = [float(row["ppo_controls"][key]) for row in rows]
        minimum = round(min(values), 3)
        maximum = round(max(values), 3)
        delta = round(maximum - minimum, 3)
        stable = delta <= 0.035
        summary[key] = {
            "min": minimum,
            "max": maximum,
            "current": current[key],
            "delta": delta,
            "stable": stable,
            "message_tr": "Bu kontrolde senaryo boyunca düşük varyans var; bu UI hatası değildir." if stable else "",
        }
    return summary


def _trace_step_index(step: int, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    return max(0, min(int(step), len(rows) - 1))


def _operation_kpis(scenario_id: str, step: int, orders: list[dict[str, Any]]) -> dict[str, float]:
    kpis = build_kpis(scenario_id, step)
    delivered = sum(1 for order in orders if order["status"] == "delivered")
    delayed = sum(1 for order in orders if order["status"] == "delayed")
    kpis["service_level"] = round(_clamp(kpis["service_level"] + delivered * 0.002 - delayed * 0.004, 0, 1), 3)
    kpis["lateness"] = round(_clamp(kpis["lateness"] + delayed * 0.003, 0, 1), 4)
    return kpis


def _operation_phase(step: int) -> str:
    phase = (step // 45) % 4
    return ["demand_intake", "route_assignment", "dispatch_execution", "delivery_closure"][phase]


def _operation_phase_label(phase: str) -> str:
    labels = {
        "demand_intake": "Talep alımı",
        "route_assignment": "Rota ataması",
        "dispatch_execution": "Sevk yürütme",
        "delivery_closure": "Teslimat kapanışı",
        "live_optional": "Teknik mod",
    }
    return labels.get(phase, "Operasyon izi")


def _operation_observation_summary(
    scenario_id: str, step: int, orders: list[dict[str, Any]], active_route: dict[str, Any]
) -> dict[str, Any]:
    stress = scenario_stress_profile(scenario_id)
    waiting = sum(1 for order in orders if order["status"] in {"waiting", "assigned"})
    in_transit = sum(1 for order in orders if order["status"] == "in_transit")
    delivered = sum(1 for order in orders if order["status"] == "delivered")
    return {
        "obs_dim": 73,
        "demand_pressure": round(stress["talep"], 3),
        "route_pressure": round(stress["rota"], 3),
        "fleet_pressure": round(stress["filo"], 3),
        "waiting_orders": waiting,
        "in_transit_orders": in_transit,
        "delivered_orders": delivered,
        "active_route_family": active_route["family"],
        "note_tr": "Özet 73 boyutlu gözlem vektörünün sunum amaçlı iz açıklamasıdır.",
    }


def _operation_frame_type(scenario_id: str, step: int, active_assignment: dict[str, Any]) -> str:
    if step % DECISION_EVENT_INTERVAL == 0:
        return "decision_event"
    if active_assignment["status"] in {"delivered", "delayed"} and step == active_assignment["completed_step"]:
        return "delivery_event"
    if scenario_id == "route_disruption_congestion" and step % DECISION_EVENT_INTERVAL == 9:
        return "disruption_event"
    if active_assignment["bound_decoded_action"].get("reorder_mode") != "none" and step % DECISION_EVENT_INTERVAL == 6:
        return "reorder_event"
    return "movement_frame"


def _operation_event(
    scenario_id: str,
    step: int,
    frame_type: str,
    operation_phase: str,
    active_assignment: dict[str, Any],
    active_route: dict[str, Any],
    orders: list[dict[str, Any]],
) -> str:
    action = active_assignment["bound_decoded_action"]
    if frame_type == "decision_event":
        return (
            f"DQN {action['action_id']} seçildi: {action['interpretation_tr']}. "
            f"{active_assignment['vehicle_id']} aracı {active_assignment['order_id']} siparişi için "
            f"{active_route['family_tr']} rotasına bağlandı."
        )
    if frame_type == "delivery_event":
        return (
            f"{active_assignment['order_id']} teslimat durumuna geçti; "
            f"bağlı karar DQN {active_assignment['bound_dqn_action_id']} olarak korunuyor."
        )
    if frame_type == "disruption_event":
        return (
            f"Kesinti riski izleniyor; {active_assignment['vehicle_id']} aynı atama rotasında ilerliyor, "
            "yeni karar üretilmedi."
        )
    if frame_type == "reorder_event":
        return (
            f"İkmal kontrolü {action['reorder_mode_tr']} seviyesinde taşınıyor; "
            f"DQN {action['action_id']} kararı değişmedi."
        )
    return (
        f"Araç {active_assignment['vehicle_id']}, {active_assignment['order_id']} siparişi için "
        f"aynı rota üzerinde ilerliyor; yeni karar üretilmedi."
    )


def _operation_action_history(assignments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for assignment in assignments[-14:]:
        action = assignment["bound_decoded_action"]
        tooltip = f"DQN {action['action_id']}: {action['interpretation_tr']}"
        rows.append(
            {
                "step": assignment["created_step"],
                "action_id": int(action["action_id"]),
                "label": action["interpretation_tr"],
                "interpretation_tr": action["interpretation_tr"],
                "watch_tr": action["watch_tr"],
                "assignment_id": assignment["assignment_id"],
                "tooltip_tr": tooltip,
            }
        )
    return rows


def _find_assignment(assignments: list[dict[str, Any]], assignment_id: str | None) -> dict[str, Any] | None:
    if not assignment_id:
        return None
    return next((assignment for assignment in assignments if assignment["assignment_id"] == assignment_id), None)


def _decision_panel_payload(assignment: dict[str, Any], selected: bool) -> dict[str, Any]:
    action = assignment["bound_decoded_action"]
    return {
        "panel_title_tr": "Seçili aktif operasyon" if selected else "Son karar olayı",
        "assignment_id": assignment["assignment_id"],
        "order_id": assignment["order_id"],
        "vehicle_id": assignment["vehicle_id"],
        "route_id": assignment["route_id"],
        "route_family_tr": assignment["route_family_tr"],
        "dqn_action_id": int(action["action_id"]),
        "decoded_action_tr": action["interpretation_tr"],
        "dispatch_tr": action["dispatch_tr"],
        "route_family": action["route_family_tr"],
        "fleet_mode_tr": action["fleet_mode_tr"],
        "reorder_mode_tr": action["reorder_mode_tr"],
        "bound_ppo_controls": deepcopy(assignment["bound_ppo_controls"]),
        "ppo_source": assignment["ppo_source"],
        "last_decision_step": assignment["last_decision_step"],
        "status_tr": assignment["status_tr"],
    }


def _routes_for_legacy_map(row: dict[str, Any]) -> list[dict[str, Any]]:
    active_ids = {route["id"] for route in row["active_routes"]}
    routes: list[dict[str, Any]] = []
    for route in row["route_candidates"]:
        path = deepcopy(route["path"])
        start = path[0]
        end = path[-1]
        routes.append(
            {
                **deepcopy(route),
                "path": path,
                "x1": start["x"],
                "y1": start["y"],
                "x2": end["x"],
                "y2": end["y"],
                "selected": route["id"] in active_ids,
            }
        )
    return routes


def presentation_event_line(scenario_id: str, step: int, action: dict[str, Any]) -> str:
    if step > 0 and step % 2 == 1:
        return f"Ara\u00e7 A{(step % 4) + 1} sipari\u015f S-{200 + step} i\u00e7in rota aday\u0131 se\u00e7ti."
    if step == 0:
        return f"{scenario_name_tr(scenario_id)} sunum izi haz\u0131r."
    if step % 2 == 0:
        return f"DQN aksiyon {action['action_id']}: {action['interpretation_tr']}"
    return f"Ara\u00e7 A{(step % 4) + 1} sipari\u015f S-{200 + step} i\u00e7in rota aday\u0131 se\u00e7ti."


@dataclass
class ControlRoomApp:
    root: Path
    selected_scenario: str = DEFAULT_SCENARIO
    mode: str = SUNUM_MODU
    speed: str = "1x"
    production_policy_read_only: bool = False
    running: bool = False
    step: int = 0
    selected_assignment_id: str | None = None
    show_stress_zones: bool = True
    event_log: list[dict[str, str]] = field(default_factory=list)
    action_history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self._lock = threading.RLock()
        self.scenarios = discover_scenarios(self.root)
        self.actions = build_action_rows(self.root)
        self.replay = load_public_replay(self.root)
        self._presentation_traces: dict[str, dict[str, Any]] = {}
        self._operation_traces: dict[str, dict[str, Any]] = {}
        self._live_adapter = LiveSimulationAdapter(
            self.root,
            self.selected_scenario,
            enabled=self.mode == CANLI_SIMULASYON_MODU,
        )
        self._env_available = self._live_adapter.available
        self._env_blocker = self._live_adapter.blocker
        self._append_event("Kontrol odası hazır.")

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return self._build_state()

    def control_start(self) -> dict[str, Any]:
        with self._lock:
            if not self.running:
                self.running = True
            return self._build_state()

    def control_pause(self) -> dict[str, Any]:
        with self._lock:
            if self.running:
                self.running = False
            return self._build_state()

    def control_reset(self) -> dict[str, Any]:
        with self._lock:
            self.running = False
            self.step = 0
            self.event_log.clear()
            self.action_history.clear()
            self._live_adapter = LiveSimulationAdapter(
                self.root,
                self.selected_scenario,
                enabled=self.mode == CANLI_SIMULASYON_MODU,
            )
            self._env_available = self._live_adapter.available
            self._env_blocker = self._live_adapter.blocker
            self._append_event("Senaryo sıfırlandı.")
            return self._build_state()

    def control_step(self) -> dict[str, Any]:
        with self._lock:
            self._advance()
            return self._build_state()

    def tick_if_running(self) -> dict[str, Any]:
        with self._lock:
            if self.running:
                self._advance()
            return self._build_state()

    def set_scenario(self, scenario_id: str) -> dict[str, Any]:
        with self._lock:
            known = {scenario["scenario_id"] for scenario in self.scenarios}
            if scenario_id not in known:
                raise ValueError(f"unknown scenario: {scenario_id}")
            self.selected_scenario = scenario_id
            return self.control_reset()

    def set_speed(self, speed: str) -> dict[str, Any]:
        with self._lock:
            if speed not in {"0.25x", "0.5x", "1x", "2x", "5x"}:
                raise ValueError(f"unsupported speed: {speed}")
            self.speed = speed
            self._append_event(f"Hız {speed} olarak ayarlandı.")
            return self._build_state()

    def set_mode(self, mode: str) -> dict[str, Any]:
        with self._lock:
            if mode not in MODES:
                raise ValueError(f"unsupported mode: {mode}")
            self.mode = mode
            if mode == CANLI_SIMULASYON_MODU:
                self._live_adapter = LiveSimulationAdapter(self.root, self.selected_scenario, enabled=True)
            elif mode == SUNUM_MODU:
                self.production_policy_read_only = False
                self._live_adapter = LiveSimulationAdapter(self.root, self.selected_scenario, enabled=False)
            self._env_available = self._live_adapter.available
            self._env_blocker = self._live_adapter.blocker
            self._append_event(f"Mod {MODE_LABELS_TR[self.mode]} olarak ayarlandı.")
            return self._build_state()

    def set_policy_enabled(self, enabled: bool) -> dict[str, Any]:
        with self._lock:
            self.production_policy_read_only = bool(enabled)
            self._append_event("Üretim politikası salt-okunur göstergesi güncellendi.")
            return self._build_state()

    def dispatch_api(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        body = body or {}
        parsed = urlparse(path)
        route = parsed.path
        if method == "GET" and route == "/api/status":
            return {"status": "ok", "state": self.get_state()}
        if method == "GET" and route == "/api/scenarios":
            return {"scenarios": self.scenarios}
        if method == "GET" and route == "/api/actions":
            return {"actions": self.actions, "replay": self.replay}
        if method == "GET" and route == "/api/replay":
            return {
                "replay": self.replay,
                "actions": self.actions,
                "analytics": build_action_analytics(self.actions, self.replay),
            }
        if method == "GET" and route == "/api/scorecard":
            return {"scorecard": build_evidence_scorecard()}
        if method == "GET" and route == "/api/company-data-requirements":
            return {"requirements": build_company_data_requirements()}
        if method == "GET" and route == "/api/traces":
            return {"traces": self.list_traces()}
        if method == "GET" and route.startswith("/api/trace/"):
            trace_id = route.rsplit("/", 1)[-1]
            return {"trace": self.get_trace(trace_id)}
        if method == "GET" and route.startswith("/api/assignment/"):
            assignment_id = route.rsplit("/", 1)[-1]
            return {"assignment": self.get_assignment(assignment_id)}
        if method == "GET" and route == "/api/state":
            return {"state": self.get_state()}
        if method == "GET" and route == "/api/playback/state":
            return {"state": self.get_state()}
        if method == "POST" and route == "/api/control/start":
            return {"state": self.control_start()}
        if method == "POST" and route == "/api/control/pause":
            return {"state": self.control_pause()}
        if method == "POST" and route == "/api/control/reset":
            return {"state": self.control_reset()}
        if method == "POST" and route == "/api/control/step":
            return {"state": self.control_step()}
        if method == "POST" and route == "/api/control/tick":
            return {"state": self.tick_if_running()}
        if method == "POST" and route == "/api/control/set_scenario":
            return {"state": self.set_scenario(str(body.get("scenario_id", DEFAULT_SCENARIO)))}
        if method == "POST" and route == "/api/control/set_speed":
            return {"state": self.set_speed(str(body.get("speed", "1x")))}
        if method == "POST" and route == "/api/control/set_mode":
            return {"state": self.set_mode(str(body.get("mode", SUNUM_MODU)))}
        if method == "POST" and route == "/api/control/set_policy":
            return {"state": self.set_policy_enabled(bool(body.get("enabled", False)))}
        if method == "POST" and route == "/api/playback/load":
            return {"state": self.load_trace(str(body.get("trace_id", DEFAULT_SCENARIO)))}
        if method == "POST" and route == "/api/playback/play":
            return {"state": self.control_start()}
        if method == "POST" and route == "/api/playback/pause":
            return {"state": self.control_pause()}
        if method == "POST" and route == "/api/playback/reset":
            return {"state": self.control_reset()}
        if method == "POST" and route == "/api/playback/seek":
            return {"state": self.seek_playback(int(body.get("step", 0)))}
        if method == "POST" and route == "/api/playback/tick":
            return {"state": self.tick_if_running()}
        if method == "POST" and route == "/api/playback/select_assignment":
            return {"state": self.select_assignment(str(body.get("assignment_id", "")))}
        if method == "POST" and route == "/api/playback/set_stress_zones":
            return {"state": self.set_stress_zones(bool(body.get("enabled", True)))}
        if method == "POST" and route == "/api/trace/generate_live":
            return {
                "ok": False,
                "reason_tr": "Canlı simülasyonu trace'e kaydetme bu sunum paketinde kapalıdır; varsayılan paket deterministik golden trace kullanır.",
                "state": self.get_state(),
            }
        raise KeyError(f"unknown endpoint: {method} {route}")

    def _advance(self) -> None:
        if self.mode == SUNUM_MODU:
            rows = self._operation_trace()["steps"]
            if self.step < len(rows) - 1:
                self.step += 1
            else:
                self.running = False
            return
        self.step += 1
        if self.mode == CANLI_SIMULASYON_MODU and self._live_adapter.available:
            live = self._live_adapter.step(self.production_policy_read_only)
            if not live.get("ok", False):
                self._env_blocker = str(live.get("error", "unknown live-env error"))
                self._append_event(f"Canlı env blocker: {self._env_blocker}")
        action = self._current_action()
        self.action_history.append(
            {
                "step": self.step,
                "action_id": action["action_id"],
                "label": action["interpretation_tr"],
                "watch_tr": action["watch_tr"],
            }
        )
        self.action_history = self.action_history[-14:]
        if self.mode == SUNUM_MODU:
            self._append_event(self._presentation_step()["event_log_line"])
            return
        if self.step % 2 == 0:
            self._append_event(
                f"DQN aksiyon {action['action_id']}: {action['interpretation_tr']}"
            )
        else:
            self._append_event(
                f"Araç A{(self.step % 4) + 1} sipariş S-{200 + self.step} için rota adayı seçti."
            )

    def _presentation_trace(self) -> dict[str, Any]:
        if self.selected_scenario not in self._presentation_traces:
            self._presentation_traces[self.selected_scenario] = build_presentation_trace(
                self.root,
                self.selected_scenario,
                steps=120,
            )
        return self._presentation_traces[self.selected_scenario]

    def _presentation_step(self) -> dict[str, Any]:
        trace = self._presentation_trace()
        steps = trace["steps"]
        return deepcopy(steps[self.step % len(steps)])

    def list_traces(self) -> list[dict[str, Any]]:
        rows = []
        for scenario_id in PRESENTATION_TRACE_SCENARIOS:
            trace = self._operation_trace_for(scenario_id)
            rows.append(
                {
                    "trace_id": trace["trace_id"],
                    "scenario_id": trace["scenario_id"],
                    "scenario_name_tr": trace["scenario_name_tr"],
                    "step_count": trace["step_count"],
                    "deterministic": trace["deterministic"],
                    "playback_mode": trace["playback_mode"],
                }
            )
        return rows

    def get_trace(self, trace_id: str) -> dict[str, Any]:
        return deepcopy(self._operation_trace_for(trace_id))

    def get_assignment(self, assignment_id: str) -> dict[str, Any]:
        row = self._operation_step()
        assignment = _find_assignment(row["assignments"], assignment_id)
        if assignment is None:
            raise ValueError(f"unknown assignment: {assignment_id}")
        return deepcopy(assignment)

    def load_trace(self, trace_id: str) -> dict[str, Any]:
        with self._lock:
            if trace_id not in PRESENTATION_TRACE_SCENARIOS:
                raise ValueError(f"unknown trace: {trace_id}")
            self.selected_scenario = trace_id
            self.mode = SUNUM_MODU
            self.running = False
            self.step = 0
            self.selected_assignment_id = None
            self.event_log.clear()
            self.action_history.clear()
            return self._build_state()

    def seek_playback(self, step: int) -> dict[str, Any]:
        with self._lock:
            rows = self._operation_trace()["steps"]
            self.step = _trace_step_index(step, rows)
            return self._build_state()

    def select_assignment(self, assignment_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._operation_step()
            if _find_assignment(row["assignments"], assignment_id) is None:
                raise ValueError(f"unknown assignment: {assignment_id}")
            self.selected_assignment_id = assignment_id
            return self._build_state()

    def set_stress_zones(self, enabled: bool) -> dict[str, Any]:
        with self._lock:
            self.show_stress_zones = bool(enabled)
            return self._build_state()

    def _operation_trace_for(self, scenario_id: str) -> dict[str, Any]:
        if scenario_id not in PRESENTATION_TRACE_SCENARIOS:
            raise ValueError(f"unknown trace: {scenario_id}")
        if scenario_id not in self._operation_traces:
            self._operation_traces[scenario_id] = build_operation_trace(
                self.root,
                scenario_id,
                steps=OPERATION_TRACE_STEPS,
            )
        return self._operation_traces[scenario_id]

    def _operation_trace(self) -> dict[str, Any]:
        return self._operation_trace_for(self.selected_scenario)

    def _operation_step(self) -> dict[str, Any]:
        trace = self._operation_trace()
        rows = trace["steps"]
        return deepcopy(rows[_trace_step_index(self.step, rows)])

    def _build_state(self) -> dict[str, Any]:
        if self.mode == SUNUM_MODU:
            trace = self._operation_trace()
            row = self._operation_step()
            selected_assignment = _find_assignment(row["assignments"], self.selected_assignment_id)
            if selected_assignment is None:
                selected_assignment = _find_assignment(row["assignments"], row["active_assignment_id"])
            effective_selected_assignment_id = selected_assignment["assignment_id"] if selected_assignment else ""
            action = deepcopy(selected_assignment["bound_decoded_action"] if selected_assignment else row["decoded_action"])
            visible_zones = row["congestion_zones"] + row["disruption_zones"] if self.show_stress_zones else []
            visual = {
                "vehicles": deepcopy(row["vehicles"]),
                "orders": deepcopy(row["orders"]),
                "routes": _routes_for_legacy_map(row),
                "zones": deepcopy(visible_zones),
                "hubs": deepcopy(row["hubs"]),
            }
            kpis = deepcopy(row["kpis"])
            ppo_controls = deepcopy(selected_assignment["bound_ppo_controls"] if selected_assignment else row["ppo_controls"])
            active_route = deepcopy(row["active_routes"][0])
            shown_time = row["simulated_time"]
            event_log = deepcopy(row["event_log"])
            action_history = deepcopy(row["action_history"])
            playback_payload = {
                "loaded_trace_id": trace["trace_id"],
                "trace_step_count": trace["step_count"],
                "playback_mode": trace["playback_mode"],
                "frame_type": row["frame_type"],
                "current_decision_event_id": row["current_decision_event_id"],
                "latest_decision_event_id": row["latest_decision_event_id"],
                "active_assignment_id": row["active_assignment_id"],
                "selected_assignment_id": effective_selected_assignment_id,
                "assignments": deepcopy(row["assignments"]),
                "decision_panel": _decision_panel_payload(
                    selected_assignment,
                    bool(self.selected_assignment_id),
                ),
                "operation_phase": row["operation_phase"],
                "operation_phase_tr": row.get("operation_phase_tr", _operation_phase_label(row["operation_phase"])),
                "route_candidates": deepcopy(row["route_candidates"]),
                "active_routes": deepcopy(row["active_routes"]),
                "congestion_zones": deepcopy(row["congestion_zones"]),
                "disruption_zones": deepcopy(row["disruption_zones"]),
                "stress_zones_visible": bool(visible_zones),
                "stress_zones_enabled": self.show_stress_zones,
                "stress_zone_legend": deepcopy(row["stress_zone_legend"]),
                "observation_summary": deepcopy(row["observation_summary"]),
                "ppo_source": selected_assignment["ppo_source"] if selected_assignment else row["ppo_source"],
                "ppo_source_label_tr": row["ppo_source_label_tr"],
                "ppo_variance_summary": operation_ppo_variance_summary(trace, self.step),
                "ppo_explanation_tr": trace["ppo_explanation_tr"],
            }
        else:
            action = self._current_action()
            visual = build_visual_state(self.selected_scenario, self.step, action)
            live_metrics = self._live_adapter.last_metrics if self._live_adapter.available else {}
            kpis = build_kpis(self.selected_scenario, self.step)
            kpis.update({key: value for key, value in live_metrics.items() if key in kpis})
            ppo_controls = self._current_ppo_controls()
            active_route = next((route for route in visual["routes"] if route.get("selected")), visual["routes"][0])
            shown_time = simulated_time(self.step)
            event_log = list(reversed(self.event_log[-10:]))
            action_history = list(self.action_history)
            playback_payload = {
                "loaded_trace_id": self.selected_scenario,
                "trace_step_count": OPERATION_TRACE_STEPS,
                "playback_mode": "CANLI_TEKNIK_MOD",
                "frame_type": "live_optional",
                "current_decision_event_id": None,
                "latest_decision_event_id": None,
                "active_assignment_id": "",
                "selected_assignment_id": "",
                "assignments": [],
                "decision_panel": {},
                "operation_phase": "live_optional",
                "operation_phase_tr": _operation_phase_label("live_optional"),
                "route_candidates": deepcopy(visual["routes"]),
                "active_routes": [deepcopy(active_route)],
                "congestion_zones": deepcopy(visual["zones"]),
                "disruption_zones": [],
                "stress_zones_visible": bool(visual["zones"]),
                "stress_zones_enabled": self.show_stress_zones,
                "stress_zone_legend": [],
                "observation_summary": {"obs_dim": 73, "note_tr": "Canlı teknik mod özetidir."},
                "ppo_source": "policy_read_only" if self.production_policy_read_only else "fallback_static",
                "ppo_source_label_tr": "policy_read_only output" if self.production_policy_read_only else "fallback static output",
                "ppo_variance_summary": {},
                "ppo_explanation_tr": "PPO çevrim içinde ağırlık güncellemez; mevcut gözlemden beş sürekli kontrol çıkarımı üretir.",
            }
        state = {
            "title": TITLE_TR,
            "running": self.running,
            "step": self.step,
            "simulated_time": shown_time,
            "selected_scenario": self.selected_scenario,
            "scenario_name_tr": scenario_name_tr(self.selected_scenario),
            "mode": self.mode,
            "mode_label_tr": MODE_LABELS_TR[self.mode],
            "speed": self.speed,
            "model_id_short": MODEL_ID_SHORT,
            "model_id": MODEL_ID,
            "contract": CONTRACT,
            "production_policy_read_only": self.production_policy_read_only,
            "policy_indicator_tr": "Salt-okunur açık" if self.production_policy_read_only else "Salt-okunur kapalı",
            "policy_status_tr": self._policy_status_tr(),
            "policy_blocker": self._live_adapter.policy_blocker,
            "env_connected": bool(self._live_adapter.available and self.mode == CANLI_SIMULASYON_MODU),
            "env_blocker": "" if self._live_adapter.available else self._env_blocker,
            "presentation_notice_tr": PRESENTATION_NOTICE_TR,
            "map_notice_tr": MAP_COORDINATE_NOTICE_TR,
            "vehicles": visual["vehicles"],
            "orders": visual["orders"],
            "routes": visual["routes"],
            "active_route": active_route,
            "zones": visual["zones"],
            "hubs": visual["hubs"],
            "kpis": kpis,
            "ppo_controls": ppo_controls,
            "dqn_action": action,
            "scenario_stress": scenario_stress_profile(self.selected_scenario),
            "event_log": event_log,
            "action_history": action_history,
            "public_replay": self.replay,
            "actions_preview": self.actions[:48],
            "boundary": {
                "simulator_only_tr": "Simülatör içi otonom orkestrasyon; canlı şirket dağıtımı iddiası değildir.",
                "public_replay_warning_tr": PUBLIC_REPLAY_WARNING_TR,
            },
        }
        state.update(playback_payload)
        state["dashboard"] = build_dashboard_payload(state, self.scenarios, self.actions, self.replay)
        return state

    def _current_action(self) -> dict[str, Any]:
        if self.mode == SUNUM_MODU:
            return deepcopy(self._operation_step()["decoded_action"])
        if (
            self.mode == CANLI_SIMULASYON_MODU
            and self._live_adapter.last_action_id is not None
            and 0 <= self._live_adapter.last_action_id < len(self.actions)
        ):
            return deepcopy(self.actions[self._live_adapter.last_action_id])
        sequence = scenario_action_sequence(self.selected_scenario)
        action_id = sequence[self.step % len(sequence)]
        return deepcopy(self.actions[action_id])

    def _current_ppo_controls(self) -> dict[str, float]:
        if self.mode == SUNUM_MODU:
            return deepcopy(self._operation_step()["ppo_controls"])
        if self.mode == CANLI_SIMULASYON_MODU and self._live_adapter.last_continuous is not None:
            return {
                name: round(_safe_float(value), 4)
                for name, value in zip(PPO_CONTROLS, self._live_adapter.last_continuous, strict=False)
            }
        return build_ppo_controls(self.selected_scenario, self.step)

    def _policy_status_tr(self) -> str:
        if not self.production_policy_read_only:
            return "Üretim politikası kapalı; güvenli görsel denetim dizisi kullanılıyor."
        if self._live_adapter.policy_blocker:
            return f"Üretim politikası salt-okunur yüklenemedi; güvenli yedek kullanılıyor: {self._live_adapter.policy_blocker}"
        return "Üretim politikası salt-okunur devrede."

    def _append_event(self, text: str) -> None:
        self.event_log.append({"time": simulated_time(self.step), "text": text})
        self.event_log = self.event_log[-80:]


class LiveSimulationAdapter:
    def __init__(self, root: Path, scenario_id: str, *, enabled: bool = True) -> None:
        self.root = root
        self.scenario_id = scenario_id
        self.available = False
        self.blocker = ""
        self.env: Any | None = None
        self.observation: np.ndarray | None = None
        self.last_metrics: dict[str, float] = {}
        self.policy_blocker = ""
        self.last_action_id: int | None = None
        self.last_continuous: list[float] | None = None
        if not enabled:
            self.blocker = "Canl\u0131 teknik mod se\u00e7ilmedi; sunum modu deterministik iz kullan\u0131yor."
            return
        try:
            scenario_path = root / "configs" / "eval_scenarios" / f"{scenario_id}.json"
            config_path = root / "configs" / DEFAULT_TRAINING_CONFIG
            if not scenario_path.exists():
                raise FileNotFoundError(f"scenario JSON yok: {scenario_id}")
            if not config_path.exists():
                raise FileNotFoundError(f"env config yok: {config_path.name}")
            from src.eval.real_world_scenario_arena import ScenarioConfig, build_scenario_environment

            config = json.loads(config_path.read_text(encoding="utf-8"))
            scenario = ScenarioConfig.from_mapping(json.loads(scenario_path.read_text(encoding="utf-8")))
            self.env, _capability_matrix = build_scenario_environment(
                config,
                scenario,
                seed=int(scenario.seeds[0]),
                max_steps=96,
            )
            observation, info = self.env.reset(seed=int(scenario.seeds[0]))
            self.observation = np.asarray(observation, dtype=np.float32).reshape(-1)
            self.last_metrics = metrics_from_env_info(info)
            self.available = True
        except Exception as exc:
            self.blocker = f"{type(exc).__name__}: {exc}"
            self.available = False

    def step(self, production_policy_read_only: bool = False) -> dict[str, Any]:
        if not self.available or self.env is None:
            return {"ok": False, "error": self.blocker}
        try:
            action_id = scenario_action_sequence(self.scenario_id)[0]
            continuous = np.asarray([0.15, 0.50, 0.05, 0.20, 0.10], dtype=np.float32)
            self.policy_blocker = ""
            if production_policy_read_only:
                try:
                    from src.orchestration.policy_service import PolicyService

                    inference = PolicyService(device="cpu").predict_joint(
                        self.observation,
                        deterministic=True,
                        fallback_to_heuristic=False,
                    )
                    if isinstance(inference.action, dict):
                        continuous = np.asarray(inference.action["continuous"], dtype=np.float32).reshape(5)
                        action_id = int(inference.action["discrete"])
                except Exception as policy_exc:
                    self.policy_blocker = f"{type(policy_exc).__name__}: {policy_exc}"
            observation, _reward, _terminated, _truncated, info = self.env.step(
                {"continuous": continuous, "discrete": int(action_id)}
            )
            self.observation = np.asarray(observation, dtype=np.float32).reshape(-1)
            self.last_metrics = metrics_from_env_info(info)
            self.last_action_id = int(action_id)
            self.last_continuous = [round(_safe_float(value), 5) for value in continuous.tolist()]
            return {"ok": True}
        except Exception as exc:
            self.blocker = f"{type(exc).__name__}: {exc}"
            self.available = False
            return {"ok": False, "error": self.blocker}


def metrics_from_env_info(info: dict[str, Any]) -> dict[str, float]:
    snapshot = info.get("snapshot", {}) if isinstance(info, dict) else {}
    rewards = info.get("reward_components", {}) if isinstance(info, dict) else {}
    return {
        "service_level": round(float(snapshot.get("service_level", 0.0)), 3),
        "lateness": round(_safe_float(rewards.get("true_lateness_pressure")), 4),
        "dispatch_rate": round(_safe_float(rewards.get("dqn_dispatch_requested")), 3),
        "dispatch_success": round(_safe_float(rewards.get("dispatch_success_rate")), 3),
        "route_failure": round(_safe_float(rewards.get("dispatch_route_failure")), 5),
    }


def build_visual_state(scenario_id: str, step: int, action: dict[str, Any]) -> dict[str, Any]:
    hubs = [
        {"id": "H1", "x": 14, "y": 24, "label": "Kuzey Hub"},
        {"id": "H2", "x": 78, "y": 72, "label": "Güney Hub"},
    ]
    base_orders = [
        {"id": "S-201", "x": 32, "y": 18, "priority": "standart"},
        {"id": "S-204", "x": 58, "y": 29, "priority": "premium"},
        {"id": "S-309", "x": 45, "y": 82, "priority": "acil"},
        {"id": "S-411", "x": 86, "y": 47, "priority": "standart"},
        {"id": "S-512", "x": 22, "y": 74, "priority": "premium"},
        {"id": "S-618", "x": 67, "y": 88, "priority": "standart"},
    ]
    route_family = action["route_family"]
    vehicles = []
    for index, vehicle_id in enumerate(["A1", "A2", "A3", "A4"]):
        order = base_orders[(index + step) % len(base_orders)]
        hub = hubs[index % len(hubs)]
        progress = ((step * (0.10 + index * 0.025)) % 1.0)
        curve = math.sin((step + index) / 2.0) * (5 if route_family == "low_congestion" else 2)
        vehicles.append(
            {
                "id": vehicle_id,
                "x": round(hub["x"] + (order["x"] - hub["x"]) * progress + curve, 2),
                "y": round(hub["y"] + (order["y"] - hub["y"]) * progress - curve / 2, 2),
                "fleet_tr": "Birincil filo" if index in {0, 3} else "İkincil filo",
                "target_order": order["id"],
                "status_tr": "rotada" if action["dispatch"] == "dispatch" else "beklemede",
            }
        )
    orders = []
    for index, order in enumerate(base_orders):
        row = dict(order)
        delivered = step > index * 3 + 5 and action["dispatch"] == "dispatch"
        dispatched = step > index + 1 and action["dispatch"] == "dispatch"
        row["status"] = "delivered" if delivered else "dispatched" if dispatched else "pending"
        row["status_tr"] = "teslim edildi" if delivered else "sevk edildi" if dispatched else "bekliyor"
        row["pulse"] = row["status"] == "pending"
        orders.append(row)
    selected_target = vehicles[0]["target_order"]
    target = next(order for order in base_orders if order["id"] == selected_target)
    routes = [
        {
            "id": "R-short",
            "from": "H1",
            "to": selected_target,
            "family": "shortest",
            "family_tr": "En kısa rota",
            "x1": hubs[0]["x"],
            "y1": hubs[0]["y"],
            "x2": target["x"],
            "y2": target["y"],
            "selected": route_family == "shortest",
            "congestion": "orta",
        },
        {
            "id": "R-low",
            "from": "H1",
            "to": selected_target,
            "family": "low_congestion",
            "family_tr": "Düşük sıkışıklık",
            "x1": hubs[0]["x"],
            "y1": hubs[0]["y"] + 5,
            "x2": target["x"],
            "y2": target["y"] - 5,
            "selected": route_family == "low_congestion",
            "congestion": "düşük",
        },
        {
            "id": "R-res",
            "from": "H2",
            "to": selected_target,
            "family": "high_resilience",
            "family_tr": "Dayanıklı rota",
            "x1": hubs[1]["x"],
            "y1": hubs[1]["y"],
            "x2": target["x"],
            "y2": target["y"],
            "selected": route_family == "high_resilience",
            "congestion": "yüksek",
        },
    ]
    stress = scenario_stress_profile(scenario_id)
    zones = [
        {"id": "Z1", "x": 62, "y": 44, "r": 13 + stress["rota"] * 9, "label": "Sıkışıklık"},
        {"id": "Z2", "x": 38, "y": 62, "r": 9 + stress["talep"] * 7, "label": "Talep baskısı"},
    ]
    return {"hubs": hubs, "orders": orders, "vehicles": vehicles, "routes": routes, "zones": zones}


def scenario_action_sequence(scenario_id: str) -> list[int]:
    return {
        "baseline_normal": [1, 25, 41, 29, 1, 0],
        "demand_spike_volatility": [25, 41, 43, 29, 27, 25],
        "premium_sla_pressure": [41, 45, 25, 43, 29, 41],
        "route_disruption_congestion": [32, 40, 41, 32, 45, 25],
        "vehicle_scarcity_capacity_shock": [1, 25, 0, 41, 29, 1],
        "high_holding_cost": [0, 1, 25, 29, 0, 41],
        "lead_time_volatility": [1, 25, 41, 3, 43, 1],
        "mixed_stress": [24, 32, 41, 43, 25, 45],
        "low_congestion": [32, 40, 41, 32, 45, 25],
    }.get(scenario_id, [1, 25, 41, 29])


def scenario_stress_profile(scenario_id: str) -> dict[str, float]:
    base = {"talep": 0.22, "rota": 0.16, "filo": 0.18, "stok": 0.20, "SLA": 0.18}
    base.update(
        {
            "demand_spike_volatility": {"talep": 0.92, "SLA": 0.45},
            "premium_sla_pressure": {"SLA": 0.94, "talep": 0.45},
            "route_disruption_congestion": {"rota": 0.94, "talep": 0.42},
            "vehicle_scarcity_capacity_shock": {"filo": 0.88, "talep": 0.54},
            "high_holding_cost": {"stok": 0.86},
            "lead_time_volatility": {"stok": 0.66, "SLA": 0.40},
            "mixed_stress": {"talep": 0.86, "rota": 0.82, "filo": 0.76, "stok": 0.70, "SLA": 0.64},
            "low_congestion": {"rota": 0.72, "talep": 0.36},
        }.get(scenario_id, {})
    )
    return base


def build_ppo_controls(scenario_id: str, step: int) -> dict[str, float]:
    stress = scenario_stress_profile(scenario_id)
    wave = math.sin(step / 5.0)
    return {
        "reorder_fraction": round(_clamp(0.12 + stress["stok"] * 0.45 + wave * 0.03, 0, 1), 3),
        "dispatch_intensity": round(_clamp(0.28 + stress["talep"] * 0.52 + stress["SLA"] * 0.10, 0, 1), 3),
        "speed_multiplier": round(_clamp(0.92 + stress["SLA"] * 0.26 + stress["rota"] * 0.12, 0.75, 1.35), 3),
        "safety_stock_multiplier": round(_clamp(1.0 + stress["stok"] * 0.70, 1.0, 2.0), 3),
        "capacity_buffer_fraction": round(_clamp(0.04 + stress["filo"] * 0.44, 0, 0.5), 3),
    }


def build_kpis(scenario_id: str, step: int) -> dict[str, float]:
    stress = scenario_stress_profile(scenario_id)
    wave = math.sin(step / 6.0) * 0.008
    return {
        "service_level": round(_clamp(0.988 - stress["talep"] * 0.04 - stress["rota"] * 0.03 - stress["filo"] * 0.02 + wave, 0, 1), 3),
        "lateness": round(_clamp(stress["SLA"] * 0.018 + stress["rota"] * 0.014 - wave / 2, 0, 1), 4),
        "dispatch_rate": round(_clamp(0.32 + stress["talep"] * 0.45 + stress["filo"] * 0.10, 0, 1), 3),
        "dispatch_success": round(_clamp(0.998 - stress["filo"] * 0.018 - stress["rota"] * 0.012, 0, 1), 3),
        "no_work": round(_clamp(0.0002 + stress["filo"] * 0.0015, 0, 0.02), 5),
        "route_failure": round(_clamp(0.0002 + stress["rota"] * 0.0028, 0, 0.02), 5),
        "stockout_reorder_pressure": round(_clamp(stress["stok"] * 0.52, 0, 1), 3),
    }


EVIDENCE_SCORECARD_DIMENSIONS = [
    {
        "dimension": "simulator-production readiness",
        "label_tr": "Simülatör-production hazırlığı",
        "score": 4.7,
        "evidence_tr": "Üretim kopyası, aktif registry ve gate/audit kanıtları.",
        "limiter_tr": "Şirket telemetrisi yok.",
    },
    {
        "dimension": "old-production comparison",
        "label_tr": "Eski production karşılaştırması",
        "score": 4.6,
        "evidence_tr": "Eş bütçeli residual-watch karşılaştırması PASS.",
        "limiter_tr": "Karşılaştırma sentetik senaryo temelli.",
    },
    {
        "dimension": "multi-seed robustness",
        "label_tr": "Çoklu seed dayanıklılığı",
        "score": 4.2,
        "evidence_tr": "Çoklu seed benchmark kökü ve raporları.",
        "limiter_tr": "Şirket replay telemetrisi değil.",
    },
    {
        "dimension": "runtime latency",
        "label_tr": "Runtime gecikmesi",
        "score": 4.7,
        "evidence_tr": "CPU runtime smoke ve repeated latency benchmark.",
        "limiter_tr": "Donanıma bağlı.",
    },
    {
        "dimension": "rule-based baseline maturity",
        "label_tr": "Kural tabanlı baseline olgunluğu",
        "score": 4.5,
        "evidence_tr": "Sürekli-kontrol farkındalıklı baseline benchmark.",
        "limiter_tr": "Simülatör baseline'ları.",
    },
    {
        "dimension": "inventory/reorder evidence",
        "label_tr": "Stok/ikmal kanıtı",
        "score": 4.5,
        "evidence_tr": "Stok ve ikmal raporları.",
        "limiter_tr": "Şirket ikmal ekonomisi yok.",
    },
    {
        "dimension": "route/stochastic evidence",
        "label_tr": "Rota/stokastik kanıt",
        "score": 4.4,
        "evidence_tr": "Rota referansları ve public proxy analizleri.",
        "limiter_tr": "Ham stokastik alanlar eksik.",
    },
    {
        "dimension": "fleet/dispatch evidence",
        "label_tr": "Filo/dispatch kanıtı",
        "score": 4.2,
        "evidence_tr": "Fleet/dispatch public benchmark genişletmesi.",
        "limiter_tr": "Şirket sözleşme ve maliyet verisi yok.",
    },
    {
        "dimension": "public historical replay maturity",
        "label_tr": "Public replay olgunluğu",
        "score": 4.45,
        "evidence_tr": "227.891 public proxy satırı ve 48 aksiyon kapsaması.",
        "limiter_tr": "Propensity, reward ve tam trajectory yok.",
    },
    {
        "dimension": "company-data validation readiness",
        "label_tr": "Şirket verisi doğrulama hazırlığı",
        "score": 4.45,
        "evidence_tr": "Company replay bridge ve veri talep paketi.",
        "limiter_tr": "Özel şirket extract'i alınmadı.",
    },
    {
        "dimension": "research pathway maturity",
        "label_tr": "Araştırma yolu olgunluğu",
        "score": 4.45,
        "evidence_tr": "Gates, baseline, public analog ve replay kanıtları bütünleşik.",
        "limiter_tr": "Global üstünlük iddiası değildir.",
    },
    {
        "dimension": "thesis/advisor defensibility",
        "label_tr": "Tez/danışman savunulabilirliği",
        "score": 4.75,
        "evidence_tr": "Final advisor ve thesis doküman paketi.",
        "limiter_tr": "Kurumsal veri doğrulaması ayrı kalır.",
    },
    {
        "dimension": "real-world deployment readiness",
        "label_tr": "Gerçek dünya devreye alma hazırlığı",
        "score": 3.25,
        "evidence_tr": "Production handoff ve ops bundle temiz.",
        "limiter_tr": "Canlı TMS/WMS/ERP entegrasyonu yapılmadı.",
    },
]


COMPANY_DATA_TABLE_FAMILIES = [
    {
        "family": "orders",
        "label_tr": "Siparişler",
        "required_fields": ["order_id", "created_at", "promised_window_start", "promised_window_end", "priority", "sku_id", "site_id"],
        "purpose_tr": "SLA, talep baskısı ve gecikme doğrulaması.",
    },
    {
        "family": "dispatch_attempts",
        "label_tr": "Dispatch denemeleri",
        "required_fields": ["dispatch_attempt_id", "order_id", "vehicle_id", "route_id", "decision_timestamp", "action_id", "success"],
        "purpose_tr": "Aksiyon 24/32 ve dispatch başarı analizi.",
    },
    {
        "family": "deliveries",
        "label_tr": "Teslimatlar / sonuçlar",
        "required_fields": ["order_id", "pickup_timestamp", "delivery_timestamp", "delivered", "late_flag", "failure_reason"],
        "purpose_tr": "Servis, gecikme ve failure oranları.",
    },
    {
        "family": "routes",
        "label_tr": "Rotalar",
        "required_fields": ["route_id", "origin_key", "destination_key", "planned_travel_time", "actual_travel_time", "route_family"],
        "purpose_tr": "En kısa, düşük sıkışıklık ve dayanıklı rota proxy doğrulaması.",
    },
    {
        "family": "fleet_carriers",
        "label_tr": "Filo / taşıyıcı",
        "required_fields": ["vehicle_id", "carrier_id", "fleet_type", "capacity", "available_from", "cost_per_unit"],
        "purpose_tr": "Birincil/ikincil filo ekonomisi.",
    },
    {
        "family": "inventory_reorder",
        "label_tr": "Stok / ikmal",
        "required_fields": ["sku_id", "site_id", "timestamp", "on_hand", "reorder_type", "stockout_flag"],
        "purpose_tr": "İkmal modu ve stokout baskısı.",
    },
    {
        "family": "costs",
        "label_tr": "Maliyetler",
        "required_fields": ["cost_id", "order_id", "route_id", "carrier_id", "cost_type", "amount"],
        "purpose_tr": "Filo, rota ve ikmal kararlarının ekonomik yorumu.",
    },
    {
        "family": "decision_timestamps",
        "label_tr": "Karar zamanları",
        "required_fields": ["decision_event_id", "order_id", "observation_timestamp", "action_id", "continuous_vector_id"],
        "purpose_tr": "Gözlem-karar-sonuç join zinciri.",
    },
    {
        "family": "anonymized_join_keys",
        "label_tr": "Anonim join anahtarları",
        "required_fields": ["order_id", "dispatch_attempt_id", "vehicle_id", "route_id", "sku_id", "site_id"],
        "purpose_tr": "Özel veri olmadan doğrulanabilir ilişkilendirme.",
    },
]


def build_action_analytics(actions: list[dict[str, Any]], replay: dict[str, Any]) -> dict[str, Any]:
    filters = {
        "dispatch": sorted({row["dispatch"] for row in actions}),
        "route_family": sorted({row["route_family"] for row in actions}),
        "fleet_mode": sorted({row["fleet_mode"] for row in actions}),
        "reorder_mode": sorted({row["reorder_mode"] for row in actions}),
    }
    top_10 = sorted(actions, key=lambda row: int(row["global_count"]), reverse=True)[:10]
    zero_count_ids = [int(row["action_id"]) for row in replay["zero_count_actions"]]
    return {
        "action_count": len(actions),
        "filters": filters,
        "top_10_actions": deepcopy(top_10),
        "zero_count_action_ids": zero_count_ids,
        "zero_count_count": len(zero_count_ids),
        "watch_action_ids": [24, 32],
        "watch_explanation_tr": "Aksiyon 24 en kısa rota dispatch bileşenini, aksiyon 32 düşük sıkışıklık dispatch bileşenini izler.",
        "family_distribution": deepcopy(replay.get("family_distribution", {})),
        "total_public_rows": int(replay.get("global_total", 0)),
    }


def build_evidence_scorecard() -> dict[str, Any]:
    return {
        "classification": "FINAL_INTEGRATED_SCORECARD_READY_WITH_LIMITATIONS",
        "score_scale_tr": "0-5 kanıt olgunluğu ölçeği; global üstünlük skoru değildir.",
        "dimensions": deepcopy(EVIDENCE_SCORECARD_DIMENSIONS),
    }


def build_company_data_requirements() -> dict[str, Any]:
    return {
        "purpose_tr": "Gelecek doğrulama yolu; mevcut dashboard şirket verisi doğrulaması iddia etmez.",
        "table_families": deepcopy(COMPANY_DATA_TABLE_FAMILIES),
    }


def build_operation_pipeline(orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stage_map = [
        ("waiting", "Bekleyen"),
        ("assigned", "Karar verildi"),
        ("in_transit", "Sevk edildi"),
        ("delivered", "Tamamlandı"),
        ("delayed", "Gecikti / riskli"),
    ]
    normalized = []
    for order in orders:
        status = str(order.get("status", "waiting"))
        if status in {"pending"}:
            status = "waiting"
        if status in {"dispatched"}:
            status = "in_transit"
        normalized.append(status)
    counts = Counter(normalized)
    return [
        {
            "status": status,
            "label_tr": label,
            "count": int(counts.get(status, 0)),
            "order_ids": [str(order.get("id", "")) for order in orders if _normalized_order_status(order) == status],
        }
        for status, label in stage_map
    ]


def _normalized_order_status(order: dict[str, Any]) -> str:
    status = str(order.get("status", "waiting"))
    if status == "pending":
        return "waiting"
    if status == "dispatched":
        return "in_transit"
    return status


def build_decision_timeline(state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in state.get("action_history", [])[-8:]:
        rows.append(
            {
                "step": int(item.get("step", 0)),
                "action_id": int(item.get("action_id", 0)),
                "assignment_id": item.get("assignment_id", ""),
                "interpretation_tr": item.get("interpretation_tr") or item.get("label", ""),
                "watch_tr": item.get("watch_tr", ""),
            }
        )
    if not rows:
        panel = state.get("decision_panel", {})
        action = state.get("dqn_action", {})
        rows.append(
            {
                "step": int(panel.get("last_decision_step") or state.get("step", 0)),
                "action_id": int(action.get("action_id", 0)),
                "assignment_id": panel.get("assignment_id", ""),
                "interpretation_tr": action.get("interpretation_tr", ""),
                "watch_tr": action.get("watch_tr", ""),
            }
        )
    return rows


def build_kpi_timeline_payload(scenario_id: str, current_step: int) -> list[dict[str, Any]]:
    current_step = max(0, int(current_step))
    stride = max(1, current_step // 15) if current_step else 1
    sampled_steps = sorted(set(list(range(0, current_step + 1, stride)) + [current_step]))
    return [
        {
            "step": step,
            "simulated_time": simulated_time(step),
            **build_kpis(scenario_id, step),
        }
        for step in sampled_steps[-24:]
    ]


def build_scenario_comparison(scenarios: list[dict[str, Any]], selected_scenario: str, current_step: int) -> list[dict[str, Any]]:
    rows = []
    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        kpis = build_kpis(scenario_id, current_step)
        rows.append(
            {
                "scenario_id": scenario_id,
                "name_tr": scenario["name_tr"],
                "status_tr": scenario["status_tr"],
                "runnable": bool(scenario["runnable"]),
                "stress_type_tr": scenario["stress_type_tr"],
                "expected_pressure_tr": scenario["expected_pressure_tr"],
                "current": scenario_id == selected_scenario,
                "trace_available": scenario_id in PRESENTATION_TRACE_SCENARIOS,
                "kpis": kpis,
            }
        )
    return rows


def build_dashboard_payload(
    state: dict[str, Any],
    scenarios: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    replay: dict[str, Any],
) -> dict[str, Any]:
    current_step = int(state.get("step", 0))
    selected_scenario = str(state.get("selected_scenario", DEFAULT_SCENARIO))
    action = state.get("dqn_action", {})
    decision_panel = state.get("decision_panel", {})
    new_decision = state.get("current_decision_event_id") is not None or state.get("frame_type") == "decision_event"
    current_service = state.get("kpis", {}).get("service_level")
    kpi_effect = "Başlangıç" if current_step == 0 else f"Servis {_safe_float(current_service):.3f}"
    return {
        "operation_summary": {
            "model_id_short": MODEL_ID_SHORT,
            "contract": CONTRACT,
            "obs_dim": 73,
            "continuous_controls": 5,
            "discrete_actions": 48,
            "mode_label_tr": MODE_LABELS_TR.get(state.get("mode", SUNUM_MODU), "Demo Akışı"),
            "metric_context_tr": "demo trace",
            "scenario_name_tr": state.get("scenario_name_tr", ""),
            "operation_phase_tr": state.get("operation_phase_tr", ""),
            "status_tr": "Oynatılıyor" if state.get("running") else "Duraklatıldı",
        },
        "operation_pipeline": build_operation_pipeline(state.get("orders", [])),
        "action_family_flow": [
            {"label_tr": "Gözlem baskısı", "value_tr": state.get("scenario_name_tr", "")},
            {"label_tr": "PPO kontrolleri", "value_tr": "5 sürekli kontrol"},
            {"label_tr": "DQN aksiyonu", "value_tr": f"#{action.get('action_id', 0)}"},
            {"label_tr": "KPI etkisi", "value_tr": kpi_effect},
        ],
        "decision_timeline": build_decision_timeline(state),
        "decision_card": {
            "decision_event_id": state.get("current_decision_event_id") or state.get("latest_decision_event_id") or "",
            "new_decision": bool(new_decision),
            "status_message_tr": "" if new_decision else "Yeni karar yok; önceki karar uygulanıyor.",
            "assignment_id": decision_panel.get("assignment_id", ""),
            "dqn_action_id": int(action.get("action_id", 0)),
            "dispatch_tr": action.get("dispatch_tr", ""),
            "route_family_tr": action.get("route_family_tr", ""),
            "fleet_mode_tr": action.get("fleet_mode_tr", ""),
            "reorder_mode_tr": action.get("reorder_mode_tr", ""),
            "interpretation_tr": action.get("interpretation_tr", ""),
            "watch_tr": action.get("watch_tr", ""),
        },
        "kpi_timeline": build_kpi_timeline_payload(selected_scenario, current_step),
        "scenario_comparison": build_scenario_comparison(scenarios, selected_scenario, current_step),
        "action_analytics": build_action_analytics(actions, replay),
        "public_replay": deepcopy(replay),
        "evidence_scorecard": build_evidence_scorecard(),
        "company_data_requirements": build_company_data_requirements(),
        "claim_boundaries": [
            "Simülatör içi otonom orkestrasyon gösterimidir.",
            "Canlı TMS/WMS/ERP devreye alma iddiası değildir.",
            "Şirket verisi doğrulaması yapılmış sayılmaz.",
            "Kamu replay betimleyici proxy analizdir; nedensel üstünlük kanıtı değildir.",
            "Skor kartı kanıt olgunluğudur; global üstünlük skoru değildir.",
        ],
    }


def simulated_time(step: int) -> str:
    minutes = 8 * 60 + step * 6
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def export_static(root: Path, output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_static_assets(output.parent, root=root, static_preview=True)
    if output.name != "index.html":
        output.write_text(INDEX_HTML_STATIC, encoding="utf-8")


def write_static_assets(target_dir: Path, *, root: Path | None = None, static_preview: bool = False) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "index.html").write_text(INDEX_HTML_STATIC if static_preview else INDEX_HTML, encoding="utf-8")
    (target_dir / "styles.css").write_text(STYLES_CSS, encoding="utf-8")
    (target_dir / "app.js").write_text(APP_JS_V9, encoding="utf-8")
    write_golden_traces(root or repo_root(), target_dir)
    write_operation_traces(root or repo_root(), target_dir)


def find_available_port(start: int = DEFAULT_PORT) -> int:
    port = start
    while port < start + 40:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1
    raise RuntimeError("available port not found")


def make_handler(app: ControlRoomApp, static_dir: Path) -> type[BaseHTTPRequestHandler]:
    class ControlRoomHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self._handle("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._handle("POST")

        def log_message(self, format: str, *args: Any) -> None:
            return None

        def _handle(self, method: str) -> None:
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/"):
                self._handle_api(method, parsed.path)
                return
            self._serve_static(parsed.path)

        def _handle_api(self, method: str, path: str) -> None:
            try:
                body = {}
                if method == "POST":
                    length = int(self.headers.get("Content-Length", "0") or "0")
                    if length:
                        body = json.loads(self.rfile.read(length).decode("utf-8"))
                    elif "?" in self.path:
                        body = {key: values[-1] for key, values in parse_qs(urlparse(self.path).query).items()}
                payload = app.dispatch_api(method, path, body)
                self._send_json(payload)
            except Exception as exc:
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=400)

        def _serve_static(self, path: str) -> None:
            relative = "index.html" if path in {"", "/"} else path.lstrip("/")
            if relative not in {"index.html", "app.js", "styles.css"} and not relative.startswith("assets/"):
                self.send_error(404)
                return
            file_path = static_dir / relative
            if not file_path.exists():
                self.send_error(404)
                return
            data = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(str(file_path))[0] or "application/octet-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return ControlRoomHandler


def run_server(root: Path, *, port: int = DEFAULT_PORT, open_browser: bool = True) -> int:
    static_dir = root / "reports" / "demo_control_room_v9"
    write_static_assets(static_dir, root=root)
    app = ControlRoomApp(root)
    port = find_available_port(port)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app, static_dir))
    url = f"http://localhost:{port}"
    print(f"Kontrol odası hazır: {url}")
    print("Kapatmak için Ctrl+C.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Kontrol odası kapatılıyor.")
    finally:
        server.server_close()
    return 0


INDEX_HTML = """<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>5PL Dijital İkiz Karar Orkestrasyon Paneli</title>
  <link rel="stylesheet" href="/styles.css?v=20260616v9">
</head>
<body>
  <div class="app-shell">
    <header class="topbar">
      <div class="brand-block">
        <h1>5PL Dijital İkiz Karar Orkestrasyon Paneli</h1>
        <p id="dashboardContext">Simüle 5PL dijital ikizinde PPO+DQN karar orkestrasyonu.</p>
      </div>
      <div class="top-stats" aria-label="aktif durum">
        <span><b>Senaryo</b><strong id="topScenario">yükleniyor</strong></span>
        <span><b>Demo modu</b><strong id="topMode">Demo Akışı</strong></span>
        <span><b>Zaman / adım</b><strong><i id="topTime">08:00</i> · <i id="topStep">0</i></strong></span>
        <span><b>Model</b><strong id="topModel">hierarchical_v1_1m</strong></span>
        <span><b>Kontrat</b><strong id="topContract">v5 / 73 / 48</strong></span>
        <span><b>Durum</b><strong id="topStatus">Duraklatıldı</strong></span>
        <span><b>Bağlantı</b><strong id="connectionStatus">Hazır</strong></span>
      </div>
    </header>

    <aside class="sidebar">
      <section class="control-section">
        <h2>Kontroller</h2>
        <label>Senaryo<select id="scenarioSelect"></select></label>
        <label>Mod<select id="modeSelect">
          <option value="SUNUM_MODU" selected>Demo Akışı</option>
          <option value="CANLI_SIMULASYON_MODU">Teknik Simülatör</option>
          <option value="KAMU_REPLAY_MODU">Kamu Replay</option>
        </select></label>
        <label>Hız<select id="speedSelect">
          <option>0.5x</option><option selected>1x</option><option>2x</option><option>5x</option>
        </select></label>
        <div class="control-grid">
          <button id="startBtn">Oynat</button>
          <button id="pauseBtn">Duraklat</button>
          <button id="resetBtn">Baştan al</button>
          <button id="stepBtn">Tek adım</button>
        </div>
        <div class="playback-strip" aria-label="zaman çizgisi">
          <input id="timelineSlider" type="range" min="0" max="179" value="0">
          <span id="traceProgress">0 / 179</span>
        </div>
      </section>
      <section class="control-section">
        <h2>Güvenli sınırlar</h2>
        <label class="toggle"><input type="checkbox" id="policyToggle"> Üretim politikasını salt-okunur göster</label>
        <p id="presentationNotice" class="boundary">Simülatör içi otonom orkestrasyon gösterimidir; canlı şirket dağıtımı iddiası değildir.</p>
        <p class="boundary">Canlı TMS/WMS/ERP devreye alma, şirket verisi doğrulaması veya nedensel public replay üstünlüğü iddia edilmez.</p>
        <p class="boundary warning hidden" id="tabWarning">Birden fazla sekme açık olabilir; demo için tek sekme kullanın.</p>
        <p class="boundary error hidden" id="errorPanel"></p>
      </section>
    </aside>

    <main class="dashboard-grid">
      <section id="operationSummaryPanel" class="panel panel-wide">
        <div class="panel-head"><h2>Operasyon Özeti</h2><span id="operationPhase">Talep alımı</span></div>
        <div id="summaryMetrics" class="metric-row"></div>
        <h3 class="subsection-title">Operasyon Akışı</h3>
        <div id="operationPipeline" class="pipeline">
          <article class="stage"><b>Bekleyen</b><strong>Başlangıç</strong><small>demo trace</small></article>
          <article class="stage"><b>Karar verildi</b><strong>Başlangıç</strong><small>demo trace</small></article>
          <article class="stage"><b>Sevk edildi</b><strong>Başlangıç</strong><small>demo trace</small></article>
          <article class="stage"><b>Tamamlandı</b><strong>Başlangıç</strong><small>demo trace</small></article>
          <article class="stage"><b>Gecikti / riskli</b><strong>Başlangıç</strong><small>demo trace</small></article>
        </div>
        <div id="actionFamilyFlow" class="flow-row"></div>
      </section>

      <section id="ppoPanel" class="panel">
        <div class="panel-head"><h2>PPO Sürekli Kontroller</h2><span id="ppoSource">Kaynak yükleniyor</span></div>
        <p class="note">PPO çevrimiçi öğrenmez; karar anında mevcut gözleme göre beş sürekli kontrol çıktısı üretir.</p>
        <div id="ppoBars" class="control-bars"></div>
        <div id="ppoHistory" class="spark-list"></div>
      </section>

      <section id="dqnDecisionPanel" class="panel">
        <div class="panel-head"><h2>DQN Karar Kartı</h2><span id="decisionEventId">DEC</span></div>
        <p class="note">DQN, 48 taktik aksiyon içinden dispatch/rota/filo/ikmal bileşimini seçer.</p>
        <article class="decision-card">
          <strong id="dqnActionId">DQN 0</strong>
          <h3 id="dqnInterpretation">yükleniyor</h3>
          <p id="decisionStatusMessage">Yeni karar yok; önceki karar uygulanıyor.</p>
          <div id="dqnComponents" class="component-grid"></div>
          <span id="actionWatch" class="watch-badge">Normal aksiyon</span>
        </article>
      </section>

      <section id="kpiTimelinePanel" class="panel panel-wide">
        <div class="panel-head"><h2>KPI Zaman Çizgisi</h2><span id="kpiStepLabel">adım 0</span></div>
        <p class="note">KPI kartları simülatör metriği olarak okunmalıdır; başlangıç adımındaki düşük olay oranları model arızası değildir.</p>
        <div id="kpiCards" class="metric-row"></div>
        <div id="kpiTimeline" class="timeline-chart"></div>
      </section>

      <section id="scenarioComparisonPanel" class="panel panel-wide">
        <div class="panel-head"><h2>Senaryo Karşılaştırması</h2><span>senaryo kapsamı</span></div>
        <div id="scenarioComparison" class="scenario-grid"></div>
      </section>

      <section id="publicReplayPanel" class="panel">
        <div class="panel-head"><h2>Public Replay Analizi</h2><span id="replayTotal">227.891</span></div>
        <p class="warning-text">Kamu replay betimleyici proxy analizdir; nedensel OPE veya şirket verisi doğrulaması değildir.</p>
        <div id="replayTotals" class="metric-row compact"></div>
        <div id="replayDistribution" class="stack-list"></div>
      </section>

      <section id="evidenceScorecardPanel" class="panel">
        <div class="panel-head"><h2>Benchmark / Kanıt Skor Kartı</h2><span>score / 5</span></div>
        <p class="note">Bu skorlar kanıt olgunluğudur; global üstünlük skoru değildir.</p>
        <div id="scorecardGrid" class="score-list">
          <article class="score-row"><b>Simülatör-production hazırlığı</b><strong>score / 5</strong><p>Kanıt olgunluğu; şirket telemetrisi ayrı doğrulanır.</p></article>
          <article class="score-row"><b>Eski production karşılaştırması</b><strong>score / 5</strong><p>Eş bütçeli residual-watch kanıtı.</p></article>
          <article class="score-row"><b>Rota/stokastik kanıt</b><strong>score / 5</strong><p>Public rota proxy kanıtı.</p></article>
          <article class="score-row"><b>Filo/dispatch kanıtı</b><strong>score / 5</strong><p>Public dispatch ve filo proxy kanıtı.</p></article>
          <article class="score-row"><b>Public replay olgunluğu</b><strong>score / 5</strong><p>227.891 proxy satır; nedensel OPE değildir.</p></article>
          <article class="score-row"><b>Şirket verisi doğrulama hazırlığı</b><strong>score / 5</strong><p>Veri talep paketi hazır; özel extract alınmadı.</p></article>
          <article class="score-row"><b>Araştırma yolu olgunluğu</b><strong>score / 5</strong><p>Gated benchmark ve governance kanıtı.</p></article>
          <article class="score-row"><b>Gerçek dünya devreye alma hazırlığı</b><strong>score / 5</strong><p>Canlı TMS/WMS/ERP entegrasyonu iddia edilmez.</p></article>
        </div>
      </section>

      <section id="actionAnalyticsPanel" class="panel panel-wide">
        <div class="panel-head"><h2>48 Aksiyon Dağılımı</h2><span id="actionCoverage">48 aksiyon</span></div>
        <div class="filter-row">
          <input id="actionSearch" placeholder="Aksiyon ara">
          <select id="dispatchFilter"><option value="">Dispatch durumu</option></select>
          <select id="routeFilter"><option value="">Rota ailesi</option></select>
          <select id="fleetFilter"><option value="">Filo modu</option></select>
          <select id="reorderFilter"><option value="">İkmal modu</option></select>
        </div>
        <div id="actionCards" class="action-grid"></div>
        <div class="split">
          <div><h3>Top 10 public replay aksiyonu</h3><div id="topActions" class="rank-list"></div></div>
          <div><h3>Sıfır kayıtlı aksiyonlar</h3><div id="zeroActions" class="chip-row"></div></div>
        </div>
        <p id="actionWatchExplanation" class="note"></p>
      </section>

      <section id="eventFlowPanel" class="panel">
        <div class="panel-head"><h2>Olay Akışı</h2><span>karar ve hareket ayrıntıları</span></div>
        <details>
          <summary>Detayları göster</summary>
          <ol id="eventLog" class="event-list"></ol>
          <div id="decisionTimeline" class="chip-row"></div>
        </details>
      </section>

      <section id="companyDataPanel" class="panel">
        <div class="panel-head"><h2>Şirket Verisi Gereksinimleri</h2><span>gelecek doğrulama yolu</span></div>
        <div id="companyRequirements" class="requirement-list">
          <article class="requirement"><h3>Siparişler</h3><p>SLA ve talep baskısı için anonim sipariş alanları.</p></article>
          <article class="requirement"><h3>Dispatch denemeleri</h3><p>Aksiyon ve dispatch başarısı için karar kayıtları.</p></article>
          <article class="requirement"><h3>Teslimatlar</h3><p>Servis, gecikme ve başarısızlık sonucu.</p></article>
          <article class="requirement"><h3>Rotalar</h3><p>Planlanan ve gerçekleşen rota süreleri.</p></article>
          <article class="requirement"><h3>Filo / taşıyıcı</h3><p>Birincil/ikincil filo kapasitesi ve maliyeti.</p></article>
          <article class="requirement"><h3>Stok / ikmal</h3><p>İkmal modu ve stokout baskısı.</p></article>
          <article class="requirement"><h3>Maliyetler</h3><p>Kararların ekonomik yorumu.</p></article>
          <article class="requirement"><h3>Karar zamanları</h3><p>Gözlem-karar-sonuç zaman zinciri.</p></article>
          <article class="requirement"><h3>Anonim join anahtarları</h3><p>Özel veri olmadan ilişkilendirme.</p></article>
        </div>
      </section>

      <section id="claimBoundaryPanel" class="panel">
        <div class="panel-head"><h2>Sınırlar ve İddia Kontrolü</h2><span>güvenli iddia</span></div>
        <ul id="claimBoundaries" class="boundary-list"></ul>
      </section>
    </main>
  </div>
  <script src="/app.js?v=20260616v9"></script>
</body>
</html>
"""

INDEX_HTML_STATIC = INDEX_HTML.replace(
    "<body>",
    '<body class="static-preview">',
).replace(
    '<p id="dashboardContext">Simüle 5PL dijital ikizinde PPO+DQN karar orkestrasyonu.</p>',
    '<p id="dashboardContext">Bu statik önizlemedir; canlı dashboard için start_control_room.bat çalıştırın.</p>',
)


STYLES_CSS = r"""
:root{--bg:#0a0d12;--surface:#121821;--surface-2:#182130;--surface-3:#202a3a;--line:#334155;--text:#eef3f8;--muted:#9aa8ba;--accent:#22c7a9;--cyan:#54b8ff;--amber:#f5b84b;--rose:#fb7185;--green:#45d483}
*{box-sizing:border-box} html,body{max-width:100%;overflow-x:hidden} body{margin:0;background:linear-gradient(135deg,#0a0d12 0%,#111827 48%,#0f161c 100%);color:var(--text);font-family:Segoe UI,Arial,sans-serif} button,select,input{font:inherit;min-width:0}
.app-shell{min-height:100vh;display:grid;grid-template-columns:292px minmax(0,1fr);grid-template-rows:auto 1fr;gap:14px;padding:14px;min-width:0}
.topbar{grid-column:1/3;display:flex;justify-content:space-between;gap:18px;align-items:center;background:rgba(18,24,33,.96);border:1px solid var(--line);border-radius:8px;padding:15px 16px;box-shadow:0 18px 42px rgba(0,0,0,.24);min-width:0}
.brand-block h1{margin:0;font-size:25px;line-height:1.1}.brand-block p{margin:6px 0 0;color:var(--muted);font-size:13px}.static-preview .brand-block p{color:#ffe9ba;background:rgba(245,184,75,.12);border-left:3px solid var(--amber);border-radius:4px;padding:8px 10px}.static-preview .control-grid,.static-preview .playback-strip{opacity:.58}.top-stats{display:grid;grid-template-columns:repeat(4,minmax(110px,1fr));gap:8px;min-width:min(850px,62vw)}.top-stats span,.metric,.stage,.scenario-item,.requirement,.score-row,.action-card-mini{background:var(--surface-2);border:1px solid var(--line);border-radius:8px;padding:9px;min-width:0}.top-stats b,.metric b{display:block;color:var(--muted);font-size:11px;line-height:1.2}.top-stats strong,.metric strong{font-style:normal;font-size:14px;overflow-wrap:anywhere}.top-stats i{font-style:normal}.metric small{display:block;color:var(--muted);font-size:11px;margin-top:4px}
.sidebar{display:flex;flex-direction:column;gap:14px;min-width:0}.control-section{background:rgba(18,24,33,.96);border:1px solid var(--line);border-radius:8px;padding:14px;min-width:0}.control-section h2,.panel h2{margin:0;font-size:17px}.control-section label{display:flex;flex-direction:column;gap:6px;color:var(--muted);font-weight:650;margin-top:12px}.control-section select,.control-section input[type=text],.filter-row input,.filter-row select{background:#0e131a;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:9px;min-height:38px;max-width:100%}.toggle{flex-direction:row!important;align-items:center;font-weight:600!important}.control-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px}button{background:#1b2736;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:10px 11px;font-weight:750;cursor:pointer}button:hover{border-color:var(--accent);color:#fff}.control-grid button:first-child{background:linear-gradient(135deg,#0f8f7f,#22c7a9);color:#04110f}.playback-strip{display:grid;grid-template-columns:1fr auto;gap:9px;align-items:center;margin-top:12px;color:var(--muted);font-size:12px}.playback-strip input{accent-color:var(--accent);width:100%}.boundary,.note{color:#dbe6f4;background:rgba(34,199,169,.08);border-left:3px solid var(--accent);padding:9px;border-radius:4px;font-size:13px;line-height:1.4}.boundary.warning,.warning-text{background:rgba(245,184,75,.12);border-left-color:var(--amber)}.boundary.error{background:rgba(251,113,133,.12);border-left-color:var(--rose)}.hidden{display:none!important}
.dashboard-grid{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:14px;align-content:start}.panel{grid-column:span 4;background:rgba(18,24,33,.94);border:1px solid var(--line);border-radius:8px;padding:14px;min-width:0}.panel-wide{grid-column:span 8}.panel-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;margin-bottom:12px}.panel-head span{color:var(--muted);font-size:12px;text-align:right}.subsection-title{font-size:14px;margin:14px 0 8px;color:#dce7f5}.metric-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(116px,1fr));gap:8px}.metric strong{font-size:18px}.pipeline{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:8px}.stage{min-height:86px}.stage b{display:block;font-size:14px}.stage strong{font-size:28px}.stage small{color:var(--muted)}.flow-row{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:12px}.flow-step{position:relative;background:#0e131a;border:1px solid var(--line);border-radius:8px;padding:10px}.flow-step b{display:block;color:var(--accent);font-size:12px}.flow-step span{display:block;margin-top:5px;color:#d9e4ef;font-size:13px}
.control-bars,.spark-list,.stack-list,.score-list,.requirement-list,.event-list{display:flex;flex-direction:column;gap:9px}.bar-row{display:grid;grid-template-columns:146px 1fr 54px;gap:8px;align-items:center;color:var(--muted);font-size:12px}.bar-track,.spark-track,.dist-track,.timeline-track{height:10px;background:#0e131a;border:1px solid var(--line);border-radius:999px;overflow:hidden}.bar-fill,.spark-fill,.dist-fill,.timeline-fill{display:block;height:100%;background:linear-gradient(90deg,var(--cyan),var(--accent))}.sparkline{display:grid;grid-template-columns:132px 1fr 82px;gap:8px;align-items:center;color:var(--muted);font-size:12px}.stable-note{color:var(--amber);font-size:12px;margin-top:3px}.decision-card{background:linear-gradient(145deg,#122133,#152b2d);border:1px solid rgba(34,199,169,.45);border-radius:8px;padding:13px}.decision-card strong{color:var(--accent);font-size:15px}.decision-card h3{margin:8px 0;font-size:20px}.decision-card p{color:var(--muted);margin:0 0 10px}.component-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:10px 0}.component-grid span{background:#0e131a;border:1px solid var(--line);border-radius:8px;padding:8px;color:#d7e1ee;font-size:12px}.watch-badge{display:inline-flex;background:#162132;border:1px solid var(--line);border-radius:999px;padding:5px 9px;color:var(--muted);font-size:12px}.watch-badge.active{border-color:var(--amber);color:#ffe8b6;background:rgba(245,184,75,.12)}
.filter-row{display:grid;grid-template-columns:minmax(170px,1.4fr) repeat(4,minmax(126px,1fr));gap:8px}.action-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(164px,1fr));gap:8px;margin-top:10px;max-height:360px;overflow:auto;padding-right:4px}.action-card-mini b{color:var(--accent)}.action-card-mini h3{margin:5px 0;font-size:13px;line-height:1.3}.action-card-mini p{margin:0;color:var(--muted);font-size:12px}.action-card-mini.watch{border-color:var(--amber);background:rgba(245,184,75,.08)}.split{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}.split h3{font-size:14px;margin:0 0 8px}.rank-list{display:flex;flex-direction:column;gap:6px}.rank-item{display:grid;grid-template-columns:42px 1fr auto;gap:8px;align-items:center;background:#0e131a;border:1px solid var(--line);border-radius:8px;padding:7px;color:#d9e4ef;font-size:12px}.chip-row{display:flex;gap:7px;flex-wrap:wrap}.chip{background:#0e131a;border:1px solid var(--line);border-radius:999px;padding:5px 8px;color:#cbd7e5;font-size:12px}
.timeline-chart{display:flex;align-items:end;gap:6px;height:148px;background:#0e131a;border:1px solid var(--line);border-radius:8px;padding:12px;margin-top:10px}.timeline-bar{flex:1;min-width:8px;display:flex;align-items:end;gap:2px;height:100%}.timeline-bar span{display:block;flex:1;border-radius:4px 4px 0 0;min-height:4px}.timeline-service{background:var(--accent)}.timeline-late{background:var(--rose)}.timeline-dispatch{background:var(--cyan)}.timeline-route{background:var(--amber)}.scenario-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px;max-height:410px;overflow:auto}.scenario-item.current{border-color:var(--accent)}.scenario-item h3{margin:0 0 7px;font-size:14px}.scenario-item p{margin:4px 0;color:var(--muted);font-size:12px}.mini-kpi{display:grid;grid-template-columns:1fr 1fr;gap:5px;margin-top:8px}.mini-kpi span{background:#0e131a;border-radius:6px;padding:5px;font-size:11px;color:#dbe6f4}
.warning-text{border-left:3px solid var(--amber);padding:9px;border-radius:4px;color:#ffe9ba;font-size:13px}.dist-row{display:grid;grid-template-columns:122px 1fr 78px;gap:8px;align-items:center;color:var(--muted);font-size:12px}.dist-fill.alt{background:linear-gradient(90deg,var(--amber),var(--rose))}.score-row{display:grid;grid-template-columns:1fr 70px;gap:8px;align-items:start}.score-row b{font-size:13px}.score-row strong{color:var(--accent);font-size:17px}.score-row p,.requirement p{grid-column:1/3;margin:4px 0 0;color:var(--muted);font-size:12px;line-height:1.35}.event-list{padding-left:18px;margin:10px 0 0;max-height:210px;overflow:auto}.event-list li{padding:8px 0;border-bottom:1px solid var(--line);color:#dce7f5;font-size:13px}details summary{cursor:pointer;color:var(--accent);font-weight:750}.requirement h3{margin:0 0 5px;font-size:14px}.requirement code{display:inline-block;color:#dbeafe;background:#0e131a;border-radius:4px;padding:2px 4px;margin:2px}.boundary-list{margin:0;padding-left:18px}.boundary-list li{margin:8px 0;color:#dbe6f4}.compact .metric strong{font-size:15px}
@media(max-width:1180px){.app-shell{grid-template-columns:1fr;padding:10px;gap:10px}.topbar{grid-column:1;align-items:flex-start;flex-direction:column;padding:13px}.brand-block h1{font-size:23px}.top-stats{min-width:0;width:100%;grid-template-columns:1fr}.dashboard-grid{grid-template-columns:1fr;gap:10px}.panel,.panel-wide{grid-column:1}.pipeline,.flow-row,.split,.filter-row{grid-template-columns:1fr}.sidebar{grid-row:auto}.control-grid{grid-template-columns:1fr 1fr}}
"""


APP_JS_V9 = r"""
const state = { actions: [], replay: null, analytics: null, timer: null, ticking: false, tabId: Math.random().toString(36).slice(2) };
const speedMs = {"0.5x": 1600, "1x": 900, "2x": 520, "5x": 240};
async function api(path, body=null){
  try{
    setText("connectionStatus", "Bağlanıyor");
    const res = await fetch(path, {method: body ? "POST" : "GET", headers: {"Content-Type": "application/json"}, body: body ? JSON.stringify(body) : undefined});
    if(!res.ok){ throw new Error(await res.text()); }
    const payload = await res.json();
    setText("connectionStatus", "Hazır");
    showError("");
    return payload;
  }catch(err){
    setText("connectionStatus", "Hata");
    showError(`Bağlantı hatası: ${err.message || err}`);
    throw err;
  }
}
function setText(id, text){ const el=document.getElementById(id); if(el) el.textContent=text ?? ""; }
function showError(text){ const el=document.getElementById("errorPanel"); if(!el) return; el.textContent=text || ""; el.classList.toggle("hidden", !text); }
function clamp(n, lo, hi){ return Math.max(lo, Math.min(Number(n||0), hi)); }
function fmtPct(v){ return `${(Number(v||0)*100).toFixed(1)}%`; }
function fmtCount(v){ return Number(v||0).toLocaleString("tr-TR"); }
function option(label, value){ return `<option value="${value}">${label}</option>`; }
const LABELS = {
  dispatch: {hold:"Dispatch: beklet", dispatch:"Dispatch: sevk et"},
  route_family: {shortest:"En kısa rota", low_congestion:"Düşük sıkışıklık", high_resilience:"Dayanıklı rota"},
  fleet_mode: {secondary_fleet:"İkincil filo", primary_fleet:"Birincil filo"},
  reorder_mode: {none:"İkmal yok", conservative:"Temkinli ikmal", aggressive:"Agresif ikmal", emergency:"Acil ikmal"}
};
const FAMILY_LABELS = {dispatch:"Dispatch", route:"Rota ailesi", route_family:"Rota ailesi", mode:"Filo modu", fleet:"Filo modu", fleet_mode:"Filo modu", reorder:"İkmal modu", reorder_mode:"İkmal modu"};
const CONTROL_LABELS = {reorder_fraction:"İkmal oranı", dispatch_intensity:"Dispatch yoğunluğu", speed_multiplier:"Hız çarpanı", safety_stock_multiplier:"Emniyet stoku", capacity_buffer_fraction:"Kapasite tamponu"};
function labelFor(family, value){ return (LABELS[family] && LABELS[family][value]) || value; }
function metric(label, value, source=""){
  const sourceHtml = source ? `<small>${source}</small>` : "";
  return `<article class="metric"><b>${label}</b><strong>${value}</strong>${sourceHtml}</article>`;
}
function fmtContextPct(value, step=1, zeroText="0.0% · olay yok"){
  if(value === null || value === undefined || Number.isNaN(Number(value))){ return "Henüz hesaplanmadı"; }
  const n = Number(value);
  if(Number(step||0) === 0 && n <= 0.001){ return "Başlangıç"; }
  if(n === 0){ return zeroText; }
  if(n > 0 && n * 100 < 0.05){ return "<0.1% · çok düşük olay"; }
  return fmtPct(n);
}
function bar(label, value, max=1){
  const pct = Math.round((clamp(value, 0, max) / max) * 100);
  return `<div class="bar-row"><span>${label}</span><span class="bar-track"><i class="bar-fill" style="width:${pct}%"></i></span><b>${pct}%</b></div>`;
}
function renderSummary(s){
  const d=s.dashboard || {};
  const summary=d.operation_summary || {};
  setText("operationPhase", summary.operation_phase_tr || s.operation_phase_tr || "");
  document.getElementById("summaryMetrics").innerHTML = [
    metric("Model", summary.model_id_short || s.model_id_short),
    metric("Gözlem", `${summary.obs_dim || 73} boyut`),
    metric("PPO", `${summary.continuous_controls || 5} kontrol`),
    metric("DQN", `${summary.discrete_actions || 48} aksiyon`),
    metric("Servis", fmtContextPct((s.kpis||{}).service_level, s.step), "simülatör metriği"),
    metric("Gecikme", fmtContextPct((s.kpis||{}).lateness, s.step), "simülatör metriği"),
    metric("Dispatch", fmtContextPct((s.kpis||{}).dispatch_rate, s.step), "simülatör metriği"),
    metric("Rota hatası", fmtContextPct((s.kpis||{}).route_failure, s.step), "simülatör metriği")
  ].join("");
  document.getElementById("operationPipeline").innerHTML = (d.operation_pipeline||[]).map(stage =>
    `<article class="stage"><b>${stage.label_tr}</b><strong>${stage.count}</strong><small>${(stage.order_ids||[]).join(", ") || "boş"}</small></article>`
  ).join("");
  document.getElementById("actionFamilyFlow").innerHTML = (d.action_family_flow||[]).map(item =>
    `<article class="flow-step"><b>${item.label_tr}</b><span>${item.value_tr}</span></article>`
  ).join("");
}
function renderPpo(s){
  const controls = s.ppo_controls || {};
  setText("ppoSource", `${s.ppo_source_label_tr || s.ppo_source || "Kaynak yok"} · ${s.production_policy_read_only ? "politika salt-okunur" : "demo trace"}`);
  const maxByKey = {speed_multiplier:1.35, safety_stock_multiplier:2, capacity_buffer_fraction:.5};
  document.getElementById("ppoBars").innerHTML = Object.entries(controls).map(([name,value]) => bar(CONTROL_LABELS[name] || name, value, maxByKey[name] || 1)).join("");
  const summary = s.ppo_variance_summary || {};
  document.getElementById("ppoHistory").innerHTML = Object.entries(summary).map(([name,row]) => {
    const min=Number(row.min||0), max=Number(row.max||1), current=Number(row.current||0);
    const pct = max > min ? Math.round(((current-min)/(max-min))*100) : 50;
    return `<div><div class="sparkline"><span>${CONTROL_LABELS[name] || name}</span><span class="spark-track"><i class="spark-fill" style="width:${clamp(pct,2,100)}%"></i></span><b>${min.toFixed(2)} / ${max.toFixed(2)} / ${current.toFixed(2)}</b></div>${row.message_tr ? `<p class="stable-note">${row.message_tr}</p>` : ""}</div>`;
  }).join("");
}
function renderDqn(s){
  const card=(s.dashboard||{}).decision_card || {};
  setText("decisionEventId", card.decision_event_id || "DEC");
  setText("dqnActionId", `DQN ${card.dqn_action_id ?? (s.dqn_action||{}).action_id ?? 0}`);
  setText("dqnInterpretation", card.interpretation_tr || (s.dqn_action||{}).interpretation_tr || "");
  setText("decisionStatusMessage", card.status_message_tr || "Yeni karar olayı işlendi.");
  document.getElementById("dqnComponents").innerHTML = [
    ["Dispatch", card.dispatch_tr],["Rota", card.route_family_tr],["Filo", card.fleet_mode_tr],["İkmal", card.reorder_mode_tr],["Atama", card.assignment_id],["Karar", card.decision_event_id]
  ].filter(([,v])=>v).map(([k,v])=>`<span><b>${k}</b><br>${v}</span>`).join("");
  const watch=document.getElementById("actionWatch");
  const active=Boolean(card.watch_tr);
  watch.textContent = active ? `${card.watch_tr}: 24/32 izlemesi` : "Normal aksiyon";
  watch.classList.toggle("active", active);
}
function renderKpis(s){
  const k=s.kpis || {};
  setText("kpiStepLabel", `adım ${s.step}`);
  document.getElementById("kpiCards").innerHTML = [
    metric("Servis", fmtContextPct(k.service_level, s.step), "simülatör metriği"),
    metric("Gecikme", fmtContextPct(k.lateness, s.step), "simülatör metriği"),
    metric("Dispatch oranı", fmtContextPct(k.dispatch_rate, s.step), "simülatör metriği"),
    metric("Dispatch başarısı", fmtContextPct(k.dispatch_success, s.step), "simülatör metriği"),
    metric("Faydasız iş", fmtContextPct(k.no_work, s.step), "simülatör metriği"),
    metric("Rota hatası", fmtContextPct(k.route_failure, s.step), "simülatör metriği"),
    metric("Stokout / ikmal baskısı", fmtContextPct(k.stockout_reorder_pressure, s.step), "simülatör metriği")
  ].join("");
  const rows=((s.dashboard||{}).kpi_timeline || []);
  document.getElementById("kpiTimeline").innerHTML = rows.map(row => {
    const service = clamp(row.service_level,0,1)*100;
    const late = clamp(row.lateness*8,0,1)*100;
    const dispatch = clamp(row.dispatch_rate,0,1)*100;
    const route = clamp(row.route_failure*80,0,1)*100;
    return `<div class="timeline-bar" title="${row.simulated_time} step ${row.step}"><span class="timeline-service" style="height:${service}%"></span><span class="timeline-late" style="height:${late}%"></span><span class="timeline-dispatch" style="height:${dispatch}%"></span><span class="timeline-route" style="height:${route}%"></span></div>`;
  }).join("");
}
function fillFilter(id, values){
  const el=document.getElementById(id);
  const first=el.options[0]?.outerHTML || option("Tümü","");
  const family = id.replace("Filter", "");
  const familyKey = family === "route" ? "route_family" : family === "fleet" ? "fleet_mode" : family === "reorder" ? "reorder_mode" : family;
  el.innerHTML = first + (values||[]).map(v=>option(labelFor(familyKey, v), v)).join("");
}
function renderActions(){
  const q=(document.getElementById("actionSearch")?.value||"").toLowerCase();
  const dispatch=document.getElementById("dispatchFilter")?.value || "";
  const route=document.getElementById("routeFilter")?.value || "";
  const fleet=document.getElementById("fleetFilter")?.value || "";
  const reorder=document.getElementById("reorderFilter")?.value || "";
  const rows=state.actions.filter(a => {
    const searchable = [a.action_id, a.interpretation_tr, a.dispatch_tr, a.route_family_tr, a.fleet_mode_tr, a.reorder_mode_tr].join(" ").toLowerCase();
    return (!q || searchable.includes(q)) &&
      (!dispatch || a.dispatch===dispatch) &&
      (!route || a.route_family===route) &&
      (!fleet || a.fleet_mode===fleet) &&
      (!reorder || a.reorder_mode===reorder);
  });
  document.getElementById("actionCoverage").textContent = `${rows.length} / 48 görünür`;
  document.getElementById("actionCards").innerHTML = rows.map(a =>
    `<article class="action-card-mini ${a.watch_tr ? "watch" : ""}"><b>#${a.action_id}</b><h3>${a.interpretation_tr}</h3><p>${fmtCount(a.global_count)} public replay metriği · ${fmtPct(a.global_rate)}</p></article>`
  ).join("");
  document.getElementById("topActions").innerHTML = (state.analytics?.top_10_actions||[]).map(a =>
    `<div class="rank-item"><b>#${a.action_id}</b><span>${a.interpretation_tr || [labelFor("dispatch", a.dispatch), labelFor("route_family", a.route_family), labelFor("fleet_mode", a.fleet_mode), labelFor("reorder_mode", a.reorder_mode)].join(" / ")}</span><strong>${fmtCount(a.global_count)}</strong></div>`
  ).join("");
  document.getElementById("zeroActions").innerHTML = (state.analytics?.zero_count_action_ids||[]).map(id=>`<span class="chip">#${id}</span>`).join("");
  setText("actionWatchExplanation", state.analytics?.watch_explanation_tr || "");
}
function renderReplay(){
  const r=state.replay; if(!r) return;
  setText("replayTotal", fmtCount(r.global_total));
  document.getElementById("replayTotals").innerHTML = [
    metric("Toplam", fmtCount(r.global_total), "public replay metriği"),
    metric("LaDe", fmtCount(r.dataset_totals.LaDe), "public replay metriği"),
    metric("NYC HVFHS", fmtCount(r.dataset_totals["NYC HVFHS"]), "public replay metriği"),
    metric("Olist", fmtCount(r.dataset_totals.Olist), "public replay metriği")
  ].join("");
  const families = r.family_distribution || {};
  const rows = [];
  for(const [family, values] of Object.entries(families)){
    for(const [label, count] of Object.entries(values)){
      rows.push({family,label,count});
    }
  }
  document.getElementById("replayDistribution").innerHTML = rows.map(row => {
    const pct = r.global_total ? Math.round((row.count/r.global_total)*100) : 0;
    const familyKey = row.family === "route" ? "route_family" : row.family === "mode" ? "fleet_mode" : row.family === "reorder" ? "reorder_mode" : row.family;
    return `<div class="dist-row"><span>${FAMILY_LABELS[row.family] || row.family}: ${labelFor(familyKey, row.label)}</span><span class="dist-track"><i class="dist-fill ${pct<5?'alt':''}" style="width:${Math.max(1,pct)}%"></i></span><b>${fmtCount(row.count)}</b></div>`;
  }).join("");
}
function renderScenarios(s){
  const rows=(s.dashboard||{}).scenario_comparison || [];
  document.getElementById("scenarioComparison").innerHTML = rows.map(row =>
    `<article class="scenario-item ${row.current ? "current" : ""}"><h3>${row.name_tr}</h3><p>${row.runnable ? "çalıştırılabilir" : "dokümante" } · ${row.stress_type_tr}</p><p>${row.expected_pressure_tr}</p><div class="mini-kpi"><span>Servis ${fmtPct(row.kpis.service_level)}</span><span>Gecikme ${fmtPct(row.kpis.lateness)}</span><span>Dispatch ${fmtPct(row.kpis.dispatch_rate)}</span><span>Başarı ${fmtPct(row.kpis.dispatch_success)}</span></div></article>`
  ).join("");
}
function renderScorecard(s){
  const card=(s.dashboard||{}).evidence_scorecard || {};
  document.getElementById("scorecardGrid").innerHTML = (card.dimensions||[]).map(row =>
    `<article class="score-row"><b>${row.label_tr}</b><strong>${Number(row.score).toFixed(2)} / 5</strong><p>${row.evidence_tr}<br><small>Limit: ${row.limiter_tr}</small></p></article>`
  ).join("");
}
function renderCompanyRequirements(s){
  const req=(s.dashboard||{}).company_data_requirements || {};
  document.getElementById("companyRequirements").innerHTML = (req.table_families||[]).map(row =>
    `<article class="requirement"><h3>${row.label_tr}</h3><p>${row.purpose_tr}</p><p>${(row.required_fields||[]).map(f=>`<code>${f}</code>`).join(" ")}</p></article>`
  ).join("");
}
function renderEvents(s){
  document.getElementById("eventLog").innerHTML = (s.event_log||[]).map(e=>`<li><b>${e.time}</b> · ${e.text}</li>`).join("");
  const rows=(s.dashboard||{}).decision_timeline || [];
  document.getElementById("decisionTimeline").innerHTML = rows.map(item=>`<span class="chip">step ${item.step} · DQN ${item.action_id}${item.watch_tr ? " · izlenen" : ""}</span>`).join("");
  document.getElementById("claimBoundaries").innerHTML = ((s.dashboard||{}).claim_boundaries||[]).map(item=>`<li>${item}</li>`).join("");
}
function syncControls(s){
  setText("topScenario", s.scenario_name_tr);
  setText("topMode", s.mode_label_tr);
  setText("topStatus", s.running ? "Oynatılıyor" : "Duraklatıldı");
  setText("topTime", s.simulated_time);
  setText("topStep", s.step);
  setText("topModel", s.model_id_short);
  const contractEl=document.getElementById("topContract");
  if(contractEl){
    contractEl.textContent = "v5 route candidate visibility · obs 73 · action 48";
    contractEl.title = s.contract || "";
  }
  const slider=document.getElementById("timelineSlider");
  const maxStep=Math.max(0, Number(s.trace_step_count||1)-1);
  slider.max=String(maxStep);
  slider.value=String(clamp(s.step,0,maxStep));
  setText("traceProgress", `${s.step} / ${maxStep}`);
  const scenarioSelect=document.getElementById("scenarioSelect"); if(scenarioSelect && s.selected_scenario && scenarioSelect.value!==s.selected_scenario){ scenarioSelect.value=s.selected_scenario; }
  const modeSelect=document.getElementById("modeSelect"); if(modeSelect && [...modeSelect.options].some(o=>o.value===s.mode)){ modeSelect.value=s.mode; }
  const speedSelect=document.getElementById("speedSelect"); if(speedSelect && s.speed && speedMs[s.speed]){ speedSelect.value=s.speed; }
  const policyToggle=document.getElementById("policyToggle"); if(policyToggle){ policyToggle.checked=!!s.production_policy_read_only; }
}
function renderState(s){
  syncControls(s);
  renderSummary(s);
  renderPpo(s);
  renderDqn(s);
  renderKpis(s);
  renderScenarios(s);
  renderScorecard(s);
  renderCompanyRequirements(s);
  renderEvents(s);
}
async function refresh(){ const payload=await api("/api/state"); renderState(payload.state); if(payload.state.running) schedule(); }
async function tick(){ if(state.ticking) return; state.ticking=true; try{ const payload=await api("/api/control/tick", {}); renderState(payload.state); if(payload.state.running) schedule(); } finally { state.ticking=false; } }
function stopTimer(){ if(state.timer){ clearTimeout(state.timer); state.timer=null; } }
function schedule(){ stopTimer(); state.timer=setTimeout(tick, speedMs[document.getElementById("speedSelect").value] || 900); }
async function post(path, body={}){ const payload=await api(path, body); renderState(payload.state); if(payload.state.running) schedule(); else stopTimer(); }
async function loadScenarios(){
  const payload=await api("/api/scenarios");
  document.getElementById("scenarioSelect").innerHTML=(payload.scenarios||[]).map(s=>option(`${s.name_tr} · ${s.status_tr}`, s.scenario_id)).join("");
}
async function loadReplay(){
  const payload=await api("/api/replay");
  state.actions=payload.actions || [];
  state.replay=payload.replay;
  state.analytics=payload.analytics;
  fillFilter("dispatchFilter", state.analytics?.filters?.dispatch || []);
  fillFilter("routeFilter", state.analytics?.filters?.route_family || []);
  fillFilter("fleetFilter", state.analytics?.filters?.fleet_mode || []);
  fillFilter("reorderFilter", state.analytics?.filters?.reorder_mode || []);
  renderActions();
  renderReplay();
}
function setupTabWarning(){
  try{
    const bc = new BroadcastChannel("control-room-v9-tabs");
    bc.onmessage = ev => { if(ev.data && ev.data.tabId !== state.tabId){ document.getElementById("tabWarning")?.classList.remove("hidden"); } };
    bc.postMessage({tabId:state.tabId, type:"hello"});
  }catch(_err){
    const key="control-room-v9-last-tab";
    const previous=localStorage.getItem(key);
    localStorage.setItem(key, `${state.tabId}:${Date.now()}`);
    if(previous && !previous.startsWith(state.tabId)){ document.getElementById("tabWarning")?.classList.remove("hidden"); }
  }
}
function setupStaticPreview(){
  if(!document.body.classList.contains("static-preview")) return;
  for(const el of document.querySelectorAll("#startBtn,#pauseBtn,#resetBtn,#stepBtn,#timelineSlider,#speedSelect,#scenarioSelect,#modeSelect,#policyToggle")){
    el.setAttribute("disabled", "disabled");
    el.setAttribute("aria-disabled", "true");
  }
  setText("connectionStatus", "Statik önizleme");
}
document.getElementById("startBtn").onclick=()=>post("/api/control/start");
document.getElementById("pauseBtn").onclick=()=>post("/api/control/pause");
document.getElementById("resetBtn").onclick=()=>post("/api/control/reset");
document.getElementById("stepBtn").onclick=()=>post("/api/control/step");
document.getElementById("timelineSlider").oninput=e=>post("/api/playback/seek", {step:Number(e.target.value)});
document.getElementById("scenarioSelect").onchange=e=>post("/api/control/set_scenario", {scenario_id:e.target.value});
document.getElementById("speedSelect").onchange=e=>post("/api/control/set_speed", {speed:e.target.value});
document.getElementById("modeSelect").onchange=e=>post("/api/control/set_mode", {mode:e.target.value});
document.getElementById("policyToggle").onchange=e=>post("/api/control/set_policy", {enabled:e.target.checked});
for(const id of ["actionSearch","dispatchFilter","routeFilter","fleetFilter","reorderFilter"]){ document.getElementById(id).oninput=renderActions; document.getElementById(id).onchange=renderActions; }
(async function init(){ setupStaticPreview(); setupTabWarning(); if(document.body.classList.contains("static-preview")) return; await loadScenarios(); await loadReplay(); await refresh(); })();
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Türkçe canlı dijital ikiz kontrol odası.")
    parser.add_argument("--root", type=Path, default=repo_root())
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-open", action="store_true", help="Tarayıcıyı otomatik açma.")
    parser.add_argument("--export-static", type=Path, help="Statik önizleme dosyasını üret.")
    args = parser.parse_args(argv)
    if args.export_static:
        export_static(args.root, args.export_static)
        print(json.dumps({"exported": str(args.export_static), "classification": "NO_MAP_5PL_DASHBOARD_READY_WITH_SAFE_FALLBACK"}, ensure_ascii=False, indent=2))
        return 0
    return run_server(args.root, port=args.port, open_browser=not args.no_open)


if __name__ == "__main__":
    raise SystemExit(main())
