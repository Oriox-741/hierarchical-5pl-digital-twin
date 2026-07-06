from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover - exercised only when Streamlit is absent.
    st = None

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas is optional for HTML export.
    pd = None

try:
    import plotly.graph_objects as go  # type: ignore
except Exception:  # pragma: no cover - Plotly fallback is covered through payload tests.
    go = None

try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
except Exception:  # pragma: no cover - optional dependency.
    st_autorefresh = None

from src.act.discrete_action_mapper import DiscreteActionMapper


TITLE = "Multi-Layered Digital Twin Framework for Autonomous Supply Chain Orchestration"
MODEL_ID = "joint_torch_v5_prod_hierarchical_v1_1m_20260611"
CONTRACT = "physical_reality_v5_route_candidate_visibility"
OBSERVATION_DIM = 73
PPO_CONTROLS = [
    "reorder_fraction",
    "dispatch_intensity",
    "speed_multiplier",
    "safety_stock_multiplier",
    "capacity_buffer_fraction",
]
DQN_ACTION_COUNT = 48
ARCHITECTURE = "hierarchical_v1"
MAX_ACTION_HISTORY = 12
DEFAULT_TRAINING_CONFIG = "training_joint_curriculum_v5_prod_hierarchical_v1_1m_20260611.json"

CANLI_SIMULASYON_MODU = "CANLI_SIMULASYON_MODU"
GUVENLI_SENTETIK_MOD = "GÜVENLİ_SENTETİK_MOD"
PUBLIC_REPLAY_MODU = "PUBLIC_REPLAY_MODU"

MODE_LABELS_TR = {
    CANLI_SIMULASYON_MODU: "Canlı Simülasyon",
    GUVENLI_SENTETIK_MOD: "Güvenli Sentetik Mod",
    PUBLIC_REPLAY_MODU: "Kamu Replay Modu",
}

PAGE_TABS = [
    "Kontrol Odası",
    "Canlı Senaryo",
    "Dijital İkiz Haritası",
    "PPO + DQN Karar Akışı",
    "48 Aksiyon Sözlüğü",
    "Kamu Replay Analizi",
    "KPI ve Zaman Çizgisi",
    "Kanıtlar ve Sınırlar",
]

DIGITAL_TWIN_STATEMENT = (
    "Dijital ikiz, 5PL operasyonlarının simülasyon ortamıdır; PPO+DQN bu ortamın içinde çalışan kontrol politikasıdır."
)
SIMULATION_SCOPE_STATEMENT = (
    "Sistem simülasyon içinde otonomdur; canlı TMS/WMS/ERP entegrasyonu veya şirket verisi doğrulaması iddia edilmez."
)
PUBLIC_REPLAY_BOUNDARY = (
    "Kamu replay betimleyici proxy analizdir; nedensel OPE veya şirket verisi doğrulaması değildir."
)
LIVE_TRAJECTORY_LABEL = "canlı env_5pl reset/step döngüsü"
STATIC_TRAJECTORY_LABEL = "güvenli sentetik görsel demo trajektorisi"

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

FORBIDDEN_DASHBOARD_PHRASES = [
    "decision-" + "support system",
    "decision " + "support " + "system",
    "live company " + "deployment",
    "company-data validation " + "is complete",
    "public replay " + "proves",
    "causal public replay " + "superiority",
    "global state-" + "of-the-art",
]

TURKISH_DISPATCH = {"hold": "beklet", "dispatch": "sevk et"}
TURKISH_ROUTE = {
    "shortest": "en kısa rota",
    "low_congestion": "düşük sıkışıklık rotası",
    "high_resilience": "dayanıklılığı yüksek rota",
}
TURKISH_FLEET = {
    "secondary_fleet": "ikincil filo",
    "primary_fleet": "birincil filo",
}
TURKISH_REORDER = {
    "none": "ikmal yok",
    "conservative": "temkinli ikmal",
    "aggressive": "agresif ikmal",
    "emergency": "acil ikmal",
}


def repo_root() -> Path:
    return ROOT


def discover_scenarios(scenario_dir: Path) -> list[dict[str, Any]]:
    scenarios: dict[str, dict[str, Any]] = {}
    if scenario_dir.exists():
        for path in sorted(scenario_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            scenario_id = str(payload.get("scenario_id") or path.stem)
            overrides = payload.get("environment_overrides", {})
            scenarios[scenario_id] = {
                "scenario_id": scenario_id,
                "name": str(payload.get("name", scenario_id.replace("_", " ").title())),
                "name_tr": scenario_name_tr(scenario_id, str(payload.get("name", scenario_id))),
                "description": str(payload.get("description", "")),
                "description_tr": scenario_description_tr(scenario_id, str(payload.get("description", ""))),
                "status": "runnable",
                "status_tr": "çalıştırılabilir",
                "source": str(path),
                "stress_type": infer_stress_type(scenario_id, overrides),
                "stress_type_tr": infer_stress_type_tr(scenario_id, overrides),
                "expected_operational_pressure": infer_expected_pressure(
                    scenario_id,
                    str(payload.get("expected_behavior", "")),
                    overrides,
                ),
                "expected_operational_pressure_tr": infer_expected_pressure_tr(scenario_id, overrides),
                "episode_count": int(payload.get("episode_count", 0) or 0),
                "seeds": payload.get("seeds", []),
                "runnable": True,
            }
    for alias in DOCUMENTED_SCENARIO_ALIASES:
        scenarios.setdefault(
            alias,
            {
                "scenario_id": alias,
                "name": alias.replace("_", " ").title(),
                "name_tr": scenario_name_tr(alias, alias),
                "description": "Documented scenario alias from benchmark reports; no runnable JSON config is present in this workspace.",
                "description_tr": "Benchmark raporlarında geçen senaryo adı; bu çalışma alanında çalıştırılabilir JSON konfigürasyonu yok.",
                "status": "documented / not runnable in current workspace",
                "status_tr": "dokümante / bu çalışma alanında çalıştırılamaz",
                "source": "",
                "stress_type": infer_stress_type(alias, {}),
                "stress_type_tr": infer_stress_type_tr(alias, {}),
                "expected_operational_pressure": infer_expected_pressure(alias, "", {}),
                "expected_operational_pressure_tr": infer_expected_pressure_tr(alias, {}),
                "episode_count": 0,
                "seeds": [],
                "runnable": False,
            },
        )
    return [scenarios[key] for key in sorted(scenarios)]


def scenario_name_tr(scenario_id: str, fallback: str) -> str:
    labels = {
        "baseline_normal": "Normal Operasyon Baz Senaryosu",
        "demand_spike_volatility": "Talep Sıçraması ve Talep Oynaklığı",
        "premium_sla_pressure": "Premium SLA Baskısı",
        "route_disruption_congestion": "Rota Kesintisi ve Sıkışıklık",
        "vehicle_scarcity_capacity_shock": "Araç Kıtlığı ve Kapasite Şoku",
        "high_holding_cost": "Yüksek Stok Tutma Maliyeti",
        "lead_time_volatility": "Tedarik Süresi Oynaklığı",
        "mixed_stress": "Karma Stres Senaryosu",
        "low_congestion": "Düşük Sıkışıklık Rota Baskısı",
    }
    return labels.get(scenario_id, fallback.replace("_", " ").title())


def scenario_description_tr(scenario_id: str, fallback: str) -> str:
    descriptions = {
        "baseline_normal": "Normal 5PL koşullarında servis, sevk ve gecikme davranışını izler.",
        "demand_spike_volatility": "Daha yoğun ve oynak sipariş gelişlerine karşı sevk ve stok dengesini görselleştirir.",
        "premium_sla_pressure": "Dar teslim penceresi ve premium sipariş baskısı altında taktik seçimi gösterir.",
        "route_disruption_congestion": "Rota kesintisi ve sıkışıklık arttığında rota ailesi seçimini görünür yapar.",
        "vehicle_scarcity_capacity_shock": "Araç ve kapasite kıtlığında sevk seçiciliği ile kapasite tamponunu izler.",
        "high_holding_cost": "Stok tutma maliyeti yükseldiğinde ikmal ve güvenlik stoğu baskısını gösterir.",
        "lead_time_volatility": "Tedarik süresi oynaklığında stok riski ve servis dengesini görselleştirir.",
        "mixed_stress": "Talep, rota, araç ve stok baskılarını birlikte gösteren karma stres görünümüdür.",
        "low_congestion": "Düşük sıkışıklık rota bileşeninin dokümante edilen proxy bağlamını gösterir.",
    }
    return descriptions.get(scenario_id, fallback)


def infer_stress_type(scenario_id: str, overrides: dict[str, Any]) -> str:
    text = f"{scenario_id} {' '.join(overrides)}".lower()
    if "route" in text or "congestion" in text:
        return "route disruption / congestion"
    if "vehicle" in text or "fleet" in text or "capacity" in text:
        return "vehicle scarcity / capacity"
    if "holding" in text or "inventory" in text:
        return "inventory holding-cost pressure"
    if "lead_time" in text or "lead" in text:
        return "lead-time volatility"
    if "premium" in text or "sla" in text:
        return "premium SLA pressure"
    if "demand" in text:
        return "demand volatility"
    if "mixed" in text:
        return "mixed stress"
    if "baseline" in text:
        return "baseline operations"
    return "documented route/control stress"


def infer_stress_type_tr(scenario_id: str, overrides: dict[str, Any]) -> str:
    stress = infer_stress_type(scenario_id, overrides)
    labels = {
        "route disruption / congestion": "rota kesintisi / sıkışıklık",
        "vehicle scarcity / capacity": "araç kıtlığı / kapasite",
        "inventory holding-cost pressure": "stok tutma maliyeti baskısı",
        "lead-time volatility": "tedarik süresi oynaklığı",
        "premium SLA pressure": "premium SLA baskısı",
        "demand volatility": "talep oynaklığı",
        "mixed stress": "karma stres",
        "baseline operations": "normal operasyon",
        "documented route/control stress": "dokümante rota/kontrol baskısı",
    }
    return labels.get(stress, stress)


def infer_expected_pressure(scenario_id: str, expected_behavior: str, overrides: dict[str, Any]) -> str:
    if expected_behavior:
        return expected_behavior
    hints = {
        "low_congestion": "Route-family preference should remain visible when congestion pressure is present.",
        "route_disruption_congestion": "Route candidates and disruption flags pressure the controller away from brittle paths.",
        "vehicle_scarcity_capacity_shock": "Fleet availability and dispatch feasibility pressure hold/dispatch balance.",
        "premium_sla_pressure": "Premium service pressure tests dispatch success and lateness control.",
        "high_holding_cost": "Inventory pressure tests reorder mode while preserving delivery service.",
        "lead_time_volatility": "Lead-time uncertainty tests safety stock and reorder timing.",
        "mixed_stress": "Combined order, route, fleet, and inventory stress tests the full 5PL action surface.",
        "demand_spike_volatility": "Demand spikes pressure dispatch rate, route choice, and backlog control.",
        "baseline_normal": "Normal operations should keep service high with low lateness and no hard blockers.",
    }
    return hints.get(scenario_id, f"Configured overrides: {json.dumps(overrides, sort_keys=True)}")


def infer_expected_pressure_tr(scenario_id: str, overrides: dict[str, Any]) -> str:
    hints = {
        "low_congestion": "Sıkışıklık baskısı varken rota ailesi tercihi görünür kalmalıdır.",
        "route_disruption_congestion": "Rota adayları ve kesinti bayrakları kırılgan kısa yollardan kaçınma baskısı yaratır.",
        "vehicle_scarcity_capacity_shock": "Araç bulunurluğu ve kapasite sevk/beklet dengesini baskılar.",
        "premium_sla_pressure": "Premium servis baskısı sevk başarısını ve gecikme kontrolünü sınar.",
        "high_holding_cost": "Stok maliyeti, servis korunurken ikmal modunu baskılar.",
        "lead_time_volatility": "Tedarik süresi belirsizliği güvenlik stoğu ve ikmal zamanlamasını sınar.",
        "mixed_stress": "Sipariş, rota, filo ve stok baskısı 5PL aksiyon yüzeyini birlikte sınar.",
        "demand_spike_volatility": "Talep sıçraması sevk oranı, rota seçimi ve bekleyen iş kontrolünü baskılar.",
        "baseline_normal": "Normal koşullarda servis yüksek, gecikme düşük ve sert blokajlar sıfır kalmalıdır.",
    }
    return hints.get(scenario_id, f"Konfigürasyon etkileri: {json.dumps(overrides, sort_keys=True, ensure_ascii=False)}")


def build_action_decoder_rows() -> list[dict[str, Any]]:
    mapper = DiscreteActionMapper()
    rows: list[dict[str, Any]] = []
    for action_id in range(mapper.action_count):
        decoded = mapper.map(action_id).as_dict()
        route_family = decoded["route"]
        fleet_mode = decoded["mode"]
        reorder_mode = decoded["reorder"]
        dispatch_mode = decoded["dispatch"]
        watch = "watched action" if action_id in {24, 32} else ""
        watch_tr = "izlenen aksiyon" if action_id in {24, 32} else ""
        interpretation = (
            f"{dispatch_mode}; {route_family.replace('_', ' ')} route; "
            f"{fleet_mode.replace('_', ' ')}; {reorder_mode} reorder"
        )
        interpretation_tr = (
            f"{TURKISH_DISPATCH[dispatch_mode]}; {TURKISH_ROUTE[route_family]}; "
            f"{TURKISH_FLEET[fleet_mode]}; {TURKISH_REORDER[reorder_mode]}"
        )
        if watch:
            interpretation += "; residual-watch component only"
            interpretation_tr += "; yalnızca izlenen bileşen, model davranışının tamamı değildir"
        rows.append(
            {
                "action_id": action_id,
                "dispatch": dispatch_mode,
                "dispatch_tr": TURKISH_DISPATCH[dispatch_mode],
                "route_family": route_family,
                "route_family_tr": TURKISH_ROUTE[route_family],
                "fleet_mode": fleet_mode,
                "fleet_mode_tr": TURKISH_FLEET[fleet_mode],
                "reorder_mode": reorder_mode,
                "reorder_mode_tr": TURKISH_REORDER[reorder_mode],
                "operational_interpretation": interpretation,
                "operational_interpretation_tr": interpretation_tr,
                "watch": watch,
                "watch_tr": watch_tr,
            }
        )
    return rows


def load_public_replay_distribution(action_appendix: Path, family_appendix: Path | None = None) -> dict[str, Any]:
    text = action_appendix.read_text(encoding="utf-8") if action_appendix.exists() else ""
    global_match = re.search(r"Global public replay rows:\s*`?([\d,]+)`?", text)
    dataset_match = re.search(
        r"Dataset public replay rows:\s*LaDe\s*`?([\d,]+)`?;\s*NYC HVFHS\s*`?([\d,]+)`?;\s*Olist\s*`?([\d,]+)`?",
        text,
    )
    global_total = _to_int(global_match.group(1)) if global_match else 0
    dataset_totals = {
        "LaDe": _to_int(dataset_match.group(1)) if dataset_match else 0,
        "NYC HVFHS": _to_int(dataset_match.group(2)) if dataset_match else 0,
        "Olist": _to_int(dataset_match.group(3)) if dataset_match else 0,
    }
    actions: list[dict[str, Any]] = []
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 14 or not cells[0].isdigit():
            continue
        action_id = int(cells[0])
        dispatch_mode = cells[1]
        route_family = cells[2]
        fleet_mode = cells[3]
        reorder_mode = cells[4]
        actions.append(
            {
                "action_id": action_id,
                "dispatch": dispatch_mode,
                "dispatch_tr": TURKISH_DISPATCH.get(dispatch_mode, dispatch_mode),
                "route_family": route_family,
                "route_family_tr": TURKISH_ROUTE.get(route_family, route_family),
                "fleet_mode": fleet_mode,
                "fleet_mode_tr": TURKISH_FLEET.get(fleet_mode, fleet_mode),
                "reorder_mode": reorder_mode,
                "reorder_mode_tr": TURKISH_REORDER.get(reorder_mode, reorder_mode),
                "lade_count": _to_int(cells[5]),
                "lade_rate": float(cells[6]),
                "nyc_count": _to_int(cells[7]),
                "nyc_rate": float(cells[8]),
                "olist_count": _to_int(cells[9]),
                "olist_rate": float(cells[10]),
                "global_count": _to_int(cells[11]),
                "global_rate": float(cells[12]),
                "interpretation": cells[13],
                "interpretation_tr": (
                    f"{TURKISH_DISPATCH.get(dispatch_mode, dispatch_mode)}; "
                    f"{TURKISH_ROUTE.get(route_family, route_family)}; "
                    f"{TURKISH_FLEET.get(fleet_mode, fleet_mode)}; "
                    f"{TURKISH_REORDER.get(reorder_mode, reorder_mode)}"
                ),
                "watch": "watched action" if action_id in {24, 32} else "",
                "watch_tr": "izlenen aksiyon" if action_id in {24, 32} else "",
            }
        )
    family_distribution = _family_counts(actions)
    if family_appendix and family_appendix.exists():
        family_distribution["source"] = str(family_appendix)
    return {
        "global_total": global_total,
        "dataset_totals": dataset_totals,
        "actions": actions,
        "top_10_actions": sorted(actions, key=lambda row: row["global_count"], reverse=True)[:10],
        "zero_count_actions": [row for row in actions if row["global_count"] == 0],
        "family_distribution": family_distribution,
        "boundary_statement": PUBLIC_REPLAY_BOUNDARY,
        "dataset_selector_note": "Veri kümesi seçici: LaDe, NYC HVFHS, Olist veya tüm kamu replay satırları.",
    }


def _to_int(value: str) -> int:
    return int(value.replace(",", "").strip())


def _family_counts(actions: list[dict[str, Any]]) -> dict[str, Any]:
    distributions: dict[str, Counter[str]] = {
        "dispatch": Counter(),
        "route_family": Counter(),
        "fleet_mode": Counter(),
        "reorder_mode": Counter(),
    }
    for row in actions:
        count = int(row["global_count"])
        for key, counter in distributions.items():
            counter[str(row[key])] += count
    return {key: dict(counter) for key, counter in distributions.items()}


def merge_action_replay_counts(
    actions: list[dict[str, Any]],
    replay_actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {row["action_id"]: row for row in replay_actions}
    merged: list[dict[str, Any]] = []
    for action in actions:
        row = dict(action)
        replay = by_id.get(row["action_id"], {})
        row["public_replay_global_count"] = int(replay.get("global_count", 0))
        row["public_replay_global_rate"] = float(replay.get("global_rate", 0.0))
        row["public_replay_lade_count"] = int(replay.get("lade_count", 0))
        row["public_replay_nyc_count"] = int(replay.get("nyc_count", 0))
        row["public_replay_olist_count"] = int(replay.get("olist_count", 0))
        merged.append(row)
    return merged


def build_synthetic_zone_view() -> dict[str, Any]:
    return {
        "mode": "synthetic_zone_grid",
        "note_tr": "Şirket lokasyonu kullanılmaz. Koordinatlar tez demosu için soyut bir 0-100 bölge ızgarasıdır.",
        "coordinate_system": "synthetic_grid",
        "no_real_company_coordinates": True,
        "hubs": [
            {"id": "H1", "x": 18.0, "y": 25.0, "label_tr": "Kuzey hub"},
            {"id": "H2", "x": 78.0, "y": 72.0, "label_tr": "Güney hub"},
        ],
        "orders": [
            {"id": "S-101", "x": 34.0, "y": 18.0, "priority_tr": "standart", "zone": "A", "status_tr": "bekliyor"},
            {"id": "S-204", "x": 62.0, "y": 28.0, "priority_tr": "premium", "zone": "B", "status_tr": "bekliyor"},
            {"id": "S-309", "x": 44.0, "y": 82.0, "priority_tr": "acil", "zone": "C", "status_tr": "bekliyor"},
            {"id": "S-411", "x": 83.0, "y": 46.0, "priority_tr": "standart", "zone": "D", "status_tr": "bekliyor"},
            {"id": "S-512", "x": 25.0, "y": 74.0, "priority_tr": "premium", "zone": "C", "status_tr": "bekliyor"},
        ],
        "vehicles": [
            {"id": "A1", "x": 24.0, "y": 33.0, "fleet": "primary_fleet", "fleet_tr": "birincil filo", "target_order": "S-204"},
            {"id": "A2", "x": 73.0, "y": 65.0, "fleet": "secondary_fleet", "fleet_tr": "ikincil filo", "target_order": "S-309"},
            {"id": "A3", "x": 56.0, "y": 55.0, "fleet": "secondary_fleet", "fleet_tr": "ikincil filo", "target_order": "S-411"},
        ],
        "route_candidates": [
            {"from": "H1", "to": "S-204", "family": "shortest", "family_tr": "en kısa rota", "congestion_tr": "orta", "disruption": False},
            {"from": "H1", "to": "S-204", "family": "low_congestion", "family_tr": "düşük sıkışıklık", "congestion_tr": "düşük", "disruption": False},
            {"from": "H2", "to": "S-309", "family": "high_resilience", "family_tr": "dayanıklı rota", "congestion_tr": "orta", "disruption": True},
            {"from": "H2", "to": "S-411", "family": "shortest", "family_tr": "en kısa rota", "congestion_tr": "yüksek", "disruption": True},
        ],
        "delivery_zones": ["A", "B", "C", "D"],
        "legend_tr": {
            "hub": "Hub/depo",
            "vehicle": "Araç",
            "pending_order": "Bekleyen sipariş",
            "dispatched_order": "Sevk edilen sipariş",
            "highlighted_route": "Seçilen aksiyonun rota etkisi",
        },
    }


def initialize_simulation_state(scenario_id: str = "baseline_normal") -> dict[str, Any]:
    zone = build_synthetic_zone_view()
    action_id = scenario_action_sequence(scenario_id)[0]
    decoded = build_action_decoder_rows()[action_id]
    return {
        "scenario_id": scenario_id,
        "step_index": 0,
        "simulated_time": "08:00",
        "pending_orders": len(zone["orders"]),
        "dispatched_orders": 0,
        "vehicles": deepcopy(zone["vehicles"]),
        "orders": deepcopy(zone["orders"]),
        "hubs": deepcopy(zone["hubs"]),
        "route_candidates": deepcopy(zone["route_candidates"]),
        "selected_action_id": action_id,
        "decoded_action": decoded,
        "ppo_controls": build_synthetic_ppo_controls(scenario_id, 0),
        "kpi_values": build_step_kpis(scenario_id, 0),
        "recent_action_history": [
            {
                "step": 0,
                "action_id": action_id,
                "route_family_tr": decoded["route_family_tr"],
                "interpretation_tr": decoded["operational_interpretation_tr"],
            }
        ],
        "observation_summary": build_observation_summary(scenario_id, 0),
        "trajectory_label": STATIC_TRAJECTORY_LABEL,
        "running": False,
    }


def advance_demo_simulation(
    state: dict[str, Any],
    *,
    scenario_id: str | None = None,
    steps: int = 1,
) -> dict[str, Any]:
    next_state = deepcopy(state)
    active_scenario = scenario_id or str(next_state.get("scenario_id", "baseline_normal"))
    for _ in range(max(1, int(steps))):
        step = int(next_state["step_index"]) + 1
        action_id = scenario_action_sequence(active_scenario)[step % len(scenario_action_sequence(active_scenario))]
        decoded = build_action_decoder_rows()[action_id]
        route_family = decoded["route_family"]
        vehicles = move_vehicles(next_state["vehicles"], next_state["orders"], step, route_family)
        pending_orders = max(0, int(next_state["pending_orders"]) - (1 if decoded["dispatch"] == "dispatch" and step % 2 == 1 else 0))
        dispatched_orders = int(next_state["dispatched_orders"]) + (1 if decoded["dispatch"] == "dispatch" else 0)
        history = list(next_state.get("recent_action_history", []))
        history.append(
            {
                "step": step,
                "action_id": action_id,
                "route_family_tr": decoded["route_family_tr"],
                "interpretation_tr": decoded["operational_interpretation_tr"],
            }
        )
        next_state.update(
            {
                "scenario_id": active_scenario,
                "step_index": step,
                "simulated_time": simulated_time_for_step(step),
                "pending_orders": pending_orders,
                "dispatched_orders": dispatched_orders,
                "vehicles": vehicles,
                "selected_action_id": action_id,
                "decoded_action": decoded,
                "ppo_controls": build_synthetic_ppo_controls(active_scenario, step),
                "kpi_values": build_step_kpis(active_scenario, step),
                "observation_summary": build_observation_summary(active_scenario, step),
                "recent_action_history": history[-MAX_ACTION_HISTORY:],
                "trajectory_label": STATIC_TRAJECTORY_LABEL,
            }
        )
    return next_state


def scenario_action_sequence(scenario_id: str) -> list[int]:
    sequences = {
        "baseline_normal": [1, 0, 25, 41, 1, 29],
        "demand_spike_volatility": [25, 41, 43, 1, 29, 27],
        "premium_sla_pressure": [41, 45, 25, 43, 29, 1],
        "route_disruption_congestion": [32, 41, 40, 25, 32, 45],
        "vehicle_scarcity_capacity_shock": [1, 25, 0, 41, 29, 1],
        "high_holding_cost": [0, 1, 25, 29, 0, 41],
        "lead_time_volatility": [1, 25, 41, 3, 43, 1],
        "mixed_stress": [24, 32, 41, 43, 25, 45],
        "low_congestion": [32, 40, 41, 32, 45, 25],
    }
    return sequences.get(scenario_id, sequences["baseline_normal"])


def move_vehicles(
    vehicles: list[dict[str, Any]],
    orders: list[dict[str, Any]],
    step: int,
    route_family: str,
) -> list[dict[str, Any]]:
    order_by_id = {order["id"]: order for order in orders}
    route_multiplier = {"shortest": 0.18, "low_congestion": 0.12, "high_resilience": 0.09}.get(route_family, 0.11)
    moved: list[dict[str, Any]] = []
    for index, vehicle in enumerate(vehicles):
        row = dict(vehicle)
        target = order_by_id.get(str(row.get("target_order"))) or orders[index % len(orders)]
        dx = float(target["x"]) - float(row["x"])
        dy = float(target["y"]) - float(row["y"])
        wobble = math.sin((step + index) * 0.7) * 0.45
        row["x"] = round(float(row["x"]) + (dx * route_multiplier) + wobble, 3)
        row["y"] = round(float(row["y"]) + (dy * route_multiplier) - wobble, 3)
        row["status_tr"] = "rotada" if route_family != "low_congestion" else "düşük sıkışıklık rotasında"
        moved.append(row)
    return moved


def simulated_time_for_step(step: int) -> str:
    minutes = 8 * 60 + step * 6
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def build_synthetic_ppo_controls(scenario_id: str, step: int) -> dict[str, float]:
    stress = scenario_stress_profile(scenario_id)
    wave = math.sin(step / 3.0)
    controls = {
        "reorder_fraction": 0.18 + 0.20 * stress["inventory"] + 0.03 * wave,
        "dispatch_intensity": 0.42 + 0.34 * stress["demand"] + 0.02 * math.cos(step / 4.0),
        "speed_multiplier": 1.00 + 0.12 * stress["sla"] + 0.04 * stress["route"],
        "safety_stock_multiplier": 1.05 + 0.32 * stress["lead_time"] + 0.08 * stress["inventory"],
        "capacity_buffer_fraction": 0.06 + 0.25 * stress["capacity"] + 0.02 * max(wave, 0.0),
    }
    return {
        "reorder_fraction": round(_clamp(controls["reorder_fraction"], 0.0, 1.0), 3),
        "dispatch_intensity": round(_clamp(controls["dispatch_intensity"], 0.0, 1.0), 3),
        "speed_multiplier": round(_clamp(controls["speed_multiplier"], 0.75, 1.35), 3),
        "safety_stock_multiplier": round(_clamp(controls["safety_stock_multiplier"], 1.0, 2.0), 3),
        "capacity_buffer_fraction": round(_clamp(controls["capacity_buffer_fraction"], 0.0, 0.5), 3),
    }


def build_observation_summary(scenario_id: str, step: int) -> dict[str, Any]:
    stress = scenario_stress_profile(scenario_id)
    return {
        "boyut": OBSERVATION_DIM,
        "servis_baskısı": round(0.24 + stress["sla"] * 0.44 + 0.02 * math.sin(step / 2.0), 3),
        "stok_baskısı": round(0.18 + stress["inventory"] * 0.48 + stress["lead_time"] * 0.22, 3),
        "rota_sıkışıklık_baskısı": round(0.16 + stress["route"] * 0.66, 3),
        "filo_bulunurluk_baskısı": round(0.12 + stress["capacity"] * 0.62, 3),
        "faydalı_sevk_fırsatı": round(0.38 + stress["demand"] * 0.42 - stress["capacity"] * 0.08, 3),
        "özet": "73-D gözlem ham liste olarak değil, operasyonel baskı grupları halinde gösterilir.",
    }


def build_step_kpis(scenario_id: str, step: int) -> dict[str, float]:
    stress = scenario_stress_profile(scenario_id)
    wave = math.sin(step / 5.0)
    return {
        "service_level": round(_clamp(0.985 - stress["demand"] * 0.055 - stress["route"] * 0.035 - stress["capacity"] * 0.025 + wave * 0.006, 0.0, 1.0), 3),
        "lateness": round(_clamp(stress["sla"] * 0.018 + stress["route"] * 0.016 + max(0.0, -wave) * 0.004, 0.0, 1.0), 3),
        "dispatch_rate": round(_clamp(0.36 + stress["demand"] * 0.34 + stress["capacity"] * 0.16, 0.0, 1.0), 3),
        "dispatch_success": round(_clamp(0.998 - stress["capacity"] * 0.018 - stress["route"] * 0.01, 0.0, 1.0), 3),
        "no_work": round(_clamp(0.0002 + stress["capacity"] * 0.0012, 0.0, 0.02), 5),
        "route_failure": round(_clamp(0.0003 + stress["route"] * 0.0024, 0.0, 0.02), 5),
        "stockout_reorder_pressure": round(_clamp(stress["inventory"] * 0.38 + stress["lead_time"] * 0.44, 0.0, 1.0), 3),
        "action_24_rate": round(_clamp(0.02 + (0.38 if scenario_id == "mixed_stress" else 0.0), 0.0, 1.0), 3),
        "action_32_rate": round(_clamp(0.03 + (0.42 if scenario_id in {"route_disruption_congestion", "low_congestion"} else 0.0), 0.0, 1.0), 3),
    }


def build_synthetic_kpi_trajectory(scenario_id: str, steps: int = 24) -> list[dict[str, Any]]:
    rows = []
    for step in range(steps):
        row = {"step": step, "simulated_time": simulated_time_for_step(step), "trajectory_label": STATIC_TRAJECTORY_LABEL}
        row.update(build_step_kpis(scenario_id, step))
        row["selected_action_id"] = scenario_action_sequence(scenario_id)[step % len(scenario_action_sequence(scenario_id))]
        rows.append(row)
    return rows


def scenario_stress_profile(scenario_id: str) -> dict[str, float]:
    base = {"demand": 0.20, "route": 0.10, "capacity": 0.15, "inventory": 0.16, "lead_time": 0.10, "sla": 0.18}
    overrides = {
        "demand_spike_volatility": {"demand": 0.88, "sla": 0.38, "inventory": 0.34},
        "premium_sla_pressure": {"sla": 0.92, "demand": 0.42},
        "route_disruption_congestion": {"route": 0.92, "demand": 0.36},
        "vehicle_scarcity_capacity_shock": {"capacity": 0.88, "demand": 0.54},
        "high_holding_cost": {"inventory": 0.82, "lead_time": 0.20},
        "lead_time_volatility": {"lead_time": 0.88, "inventory": 0.58},
        "mixed_stress": {"demand": 0.84, "route": 0.78, "capacity": 0.72, "inventory": 0.68, "lead_time": 0.52, "sla": 0.62},
        "low_congestion": {"route": 0.72, "demand": 0.32},
    }
    profile = dict(base)
    profile.update(overrides.get(scenario_id, {}))
    return profile


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_map_figure_payload(sim_state: dict[str, Any]) -> dict[str, Any]:
    selected_route = str(sim_state["decoded_action"]["route_family"])
    return {
        "coordinate_system": "synthetic_grid",
        "no_real_company_coordinates": True,
        "hubs": deepcopy(sim_state.get("hubs", [])),
        "vehicles": deepcopy(sim_state.get("vehicles", [])),
        "orders": deepcopy(sim_state.get("orders", [])),
        "route_candidates": [
            {
                **route,
                "highlighted": str(route.get("family")) == selected_route,
            }
            for route in sim_state.get("route_candidates", [])
        ],
        "selected_action_id": int(sim_state.get("selected_action_id", 0)),
        "selected_route_family": selected_route,
        "note_tr": "sentetik dijital ikiz haritası: koordinatlar soyut simülasyon ızgarasıdır; gerçek şirket koordinatı içermez.",
    }


def build_plotly_map_figure(payload: dict[str, Any]) -> Any | None:
    if go is None:
        return None
    fig = go.Figure()
    order_lookup = {order["id"]: order for order in payload["orders"]}
    hub_lookup = {hub["id"]: hub for hub in payload["hubs"]}
    for route in payload["route_candidates"]:
        source = hub_lookup.get(route["from"])
        target = order_lookup.get(route["to"])
        if not source or not target:
            continue
        fig.add_trace(
            go.Scatter(
                x=[source["x"], target["x"]],
                y=[source["y"], target["y"]],
                mode="lines",
                line={
                    "width": 4 if route["highlighted"] else 1.5,
                    "dash": "solid" if route["highlighted"] else "dot",
                    "color": "#0f766e" if route["highlighted"] else "#94a3b8",
                },
                name=f"Rota adayı: {route['family_tr']}",
                hovertemplate="Aday rota<br>%{x:.1f}, %{y:.1f}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[hub["x"] for hub in payload["hubs"]],
            y=[hub["y"] for hub in payload["hubs"]],
            mode="markers+text",
            marker={"size": 24, "color": "#0f766e", "symbol": "square"},
            text=[hub["id"] for hub in payload["hubs"]],
            textposition="middle center",
            name="Hub/depo",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[order["x"] for order in payload["orders"]],
            y=[order["y"] for order in payload["orders"]],
            mode="markers+text",
            marker={"size": 18, "color": "#1d4ed8", "symbol": "circle"},
            text=[order["zone"] for order in payload["orders"]],
            textposition="top center",
            name="Sipariş",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[vehicle["x"] for vehicle in payload["vehicles"]],
            y=[vehicle["y"] for vehicle in payload["vehicles"]],
            mode="markers+text",
            marker={"size": 20, "color": "#a16207", "symbol": "triangle-up"},
            text=[vehicle["id"] for vehicle in payload["vehicles"]],
            textposition="bottom center",
            name="Araç",
        )
    )
    fig.update_layout(
        height=520,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis={"range": [0, 100], "title": "Sentetik X bölgesi"},
        yaxis={"range": [0, 100], "title": "Sentetik Y bölgesi", "scaleanchor": "x", "scaleratio": 1},
        legend={"orientation": "h"},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
    )
    return fig


def build_decision_trace_sample(action_id: int = 32, scenario_id: str = "route_disruption_congestion") -> dict[str, Any]:
    sim_state = advance_demo_simulation(initialize_simulation_state(scenario_id), scenario_id=scenario_id)
    if sim_state["selected_action_id"] != action_id:
        decoded = build_action_decoder_rows()[action_id]
        sim_state["selected_action_id"] = action_id
        sim_state["decoded_action"] = decoded
    return {
        "observation_summary": sim_state["observation_summary"],
        "ppo_continuous_output": sim_state["ppo_controls"],
        "dqn_action": sim_state["decoded_action"],
        "hierarchical_note_tr": "Hiyerarşik DQN içeride faktörize başlıklar kullanır; dış sözleşme 0..47 aksiyon olarak kalır.",
        "watch_note_tr": "Aksiyon 24 ve aksiyon 32 yalnızca izlenen aksiyonlardır; model davranışının tamamı olarak yorumlanmaz.",
    }


def initial_dashboard_session_state(scenario_id: str = "baseline_normal") -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "mode": CANLI_SIMULASYON_MODU,
        "running": False,
        "speed": "1x",
        "use_production_policy": False,
    }


def autorun_interval_ms(speed: str) -> int:
    intervals = {
        "0.25x": 3200,
        "0.5x": 2000,
        "1x": 1000,
        "2x": 500,
        "5x": 220,
    }
    return intervals.get(speed, intervals["1x"])


def should_auto_advance(running: bool, mode: str) -> bool:
    return bool(running and mode in {CANLI_SIMULASYON_MODU, GUVENLI_SENTETIK_MOD})


def create_dashboard_backend(
    root: Path | None,
    *,
    scenario_id: str,
    mode: str = CANLI_SIMULASYON_MODU,
    use_production_policy: bool = False,
) -> "DashboardBackendRuntime":
    return DashboardBackendRuntime(
        root=root or repo_root(),
        scenario_id=scenario_id,
        mode=mode,
        use_production_policy=use_production_policy,
    )


class DashboardBackendRuntime:
    """Small bounded runtime used by Streamlit; never writes model or eval artifacts."""

    def __init__(
        self,
        *,
        root: Path,
        scenario_id: str,
        mode: str = CANLI_SIMULASYON_MODU,
        use_production_policy: bool = False,
        max_steps: int = 96,
    ) -> None:
        self.root = Path(root)
        self.requested_mode = mode
        self.scenario_id = scenario_id
        self.use_production_policy = bool(use_production_policy)
        self.max_steps = int(max_steps)
        self.env: Any | None = None
        self.observation: np.ndarray | None = None
        self.info: dict[str, Any] = {}
        self.capability_matrix: list[dict[str, str]] = []
        self.active_mode = mode
        self.backend_source = ""
        self.fallback_reason = ""
        self.policy_source = ""
        self.policy_error = ""
        self.policy_model_path: Any = None
        self.policy_algorithm = "deterministik_demo_policy"
        self._loaded_policy_service: Any | None = None
        self._last_raw_continuous = np.zeros(5, dtype=np.float32)
        self._last_action_id = scenario_action_sequence(scenario_id)[0]
        self._state: dict[str, Any] = {}
        self._reset()

    def current_state(self) -> dict[str, Any]:
        return deepcopy(self._state)

    def reset(self, *, scenario_id: str | None = None, mode: str | None = None, use_production_policy: bool | None = None) -> dict[str, Any]:
        if scenario_id is not None:
            self.scenario_id = scenario_id
        if mode is not None:
            self.requested_mode = mode
        if use_production_policy is not None:
            self.use_production_policy = bool(use_production_policy)
        self._reset()
        return self.current_state()

    def step(self) -> dict[str, Any]:
        if self.active_mode == PUBLIC_REPLAY_MODU:
            return self.current_state()
        if self.active_mode != CANLI_SIMULASYON_MODU or self.env is None or self.observation is None:
            self._state = advance_demo_simulation(self._state, scenario_id=self.scenario_id)
            self._state.update(self._common_runtime_fields())
            return self.current_state()

        continuous, action_id = self._select_action()
        self._last_raw_continuous = continuous
        self._last_action_id = action_id
        try:
            observation, _reward, terminated, truncated, info = self.env.step(
                {"continuous": continuous.astype(np.float32), "discrete": int(action_id)}
            )
            self.observation = np.asarray(observation, dtype=np.float32).reshape(-1)
            self.info = dict(info)
            self._state = self._state_from_live(self.observation, self.info)
            if terminated or truncated:
                self._reset_live_env()
        except Exception as exc:  # pragma: no cover - exercised by integration failures.
            self._activate_synthetic_fallback(f"Canlı env adımı çalışmadı; güvenli sentetik moda geçildi: {type(exc).__name__}: {exc}")
            self._state = advance_demo_simulation(self._state, scenario_id=self.scenario_id)
            self._state.update(self._common_runtime_fields())
        return self.current_state()

    def close(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass

    def _reset(self) -> None:
        self.close()
        self.env = None
        self.observation = None
        self.info = {}
        self.capability_matrix = []
        self.policy_error = ""
        self.policy_model_path = None
        if self.requested_mode == PUBLIC_REPLAY_MODU:
            self.active_mode = PUBLIC_REPLAY_MODU
            self.backend_source = "kamu replay betimleyici proxy görünümü"
            self.fallback_reason = ""
            self._state = initialize_simulation_state(self.scenario_id)
            self._state.update(self._common_runtime_fields())
            return
        if self.requested_mode == GUVENLI_SENTETIK_MOD:
            self._activate_synthetic_fallback("Güvenli sentetik mod kullanıcı tarafından seçildi; üretim policy çıktısı gibi sunulmaz.")
            return
        self._reset_live_env()

    def _reset_live_env(self) -> None:
        try:
            scenario_path = self.root / "configs" / "eval_scenarios" / f"{self.scenario_id}.json"
            config_path = self.root / "configs" / DEFAULT_TRAINING_CONFIG
            if not scenario_path.exists():
                raise FileNotFoundError(f"senaryo JSON bulunamadı: {scenario_path}")
            if not config_path.exists():
                raise FileNotFoundError(f"canlı env konfigürasyonu bulunamadı: {config_path}")

            from src.eval.real_world_scenario_arena import ScenarioConfig, build_scenario_environment

            config = json.loads(config_path.read_text(encoding="utf-8"))
            scenario = ScenarioConfig.from_mapping(json.loads(scenario_path.read_text(encoding="utf-8")))
            env, capability_matrix = build_scenario_environment(
                config,
                scenario,
                seed=int(scenario.seeds[0]),
                max_steps=self.max_steps,
            )
            observation, info = env.reset(seed=int(scenario.seeds[0]))
            self.env = env
            self.observation = np.asarray(observation, dtype=np.float32).reshape(-1)
            self.info = dict(info)
            self.capability_matrix = list(capability_matrix)
            self.active_mode = CANLI_SIMULASYON_MODU
            self.backend_source = "env_5pl + real_world_scenario_arena reset/step döngüsü"
            self.fallback_reason = ""
            self.policy_source = (
                "üretim policy read-only opsiyonel; kapalıyken deterministik demo policy kullanılır"
            )
            self._state = self._state_from_live(self.observation, self.info)
        except Exception as exc:
            self._activate_synthetic_fallback(
                f"Canlı env bağlanamadı; güvenli sentetik moda geçildi: {type(exc).__name__}: {exc}"
            )

    def _activate_synthetic_fallback(self, reason: str) -> None:
        self.active_mode = GUVENLI_SENTETIK_MOD
        self.backend_source = "güvenli sentetik demo backend"
        self.fallback_reason = reason
        self.policy_source = "deterministik sentetik demo policy; üretim policy çıktısı değildir"
        self._state = initialize_simulation_state(self.scenario_id)
        self._state.update(self._common_runtime_fields())

    def _select_action(self) -> tuple[np.ndarray, int]:
        if self.use_production_policy:
            try:
                if self._loaded_policy_service is None:
                    from src.orchestration.policy_service import PolicyService

                    self._loaded_policy_service = PolicyService(device="cpu")
                inference = self._loaded_policy_service.predict_joint(
                    self.observation,
                    deterministic=True,
                    fallback_to_heuristic=False,
                )
                action = inference.action
                if not isinstance(action, dict):
                    raise TypeError("PolicyService joint action must be a dict.")
                continuous = np.asarray(action["continuous"], dtype=np.float32).reshape(5)
                action_id = int(action["discrete"])
                self.policy_source = "aktif üretim policy read-only"
                self.policy_error = ""
                self.policy_model_path = inference.model_path
                self.policy_algorithm = str(inference.algorithm)
                return continuous, action_id
            except Exception as exc:
                self.policy_error = f"{type(exc).__name__}: {exc}"
                self.policy_source = "üretim policy yüklenemedi; deterministik demo policy fallback"
                self.policy_model_path = None
                self.policy_algorithm = "deterministik_demo_policy"
        step = int(self._state.get("step_index", 0)) + 1
        return demo_raw_continuous_action(self.scenario_id, step), scenario_action_sequence(self.scenario_id)[
            step % len(scenario_action_sequence(self.scenario_id))
        ]

    def _common_runtime_fields(self) -> dict[str, Any]:
        return {
            "mode": self.active_mode,
            "mode_label_tr": MODE_LABELS_TR.get(self.active_mode, self.active_mode),
            "requested_mode": self.requested_mode,
            "backend_source": self.backend_source,
            "safe_fallback_active": self.active_mode == GUVENLI_SENTETIK_MOD,
            "fallback_reason": self.fallback_reason,
            "policy_source": self.policy_source,
            "policy_algorithm": self.policy_algorithm,
            "policy_error": self.policy_error,
            "policy_model_path": self.policy_model_path,
            "production_policy_read_only_enabled": self.use_production_policy,
        }

    def _state_from_live(self, observation: np.ndarray, info: dict[str, Any]) -> dict[str, Any]:
        step = int(info.get("step", 0) or 0)
        base = initialize_simulation_state(self.scenario_id)
        if step > 0:
            base = advance_demo_simulation(base, scenario_id=self.scenario_id, steps=step)

        snapshot = info.get("snapshot", {})
        reward_components = info.get("reward_components", {})
        projected = info.get("projected_action", {})
        projected_continuous = projected.get("continuous", {}) if isinstance(projected, dict) else {}
        action_id = int(self._last_action_id)
        decoded = build_action_decoder_rows()[action_id]
        orders = snapshot.get("orders", {}) if isinstance(snapshot, dict) else {}
        vehicles = snapshot.get("vehicles", {}) if isinstance(snapshot, dict) else {}
        inventory = snapshot.get("inventory", {}) if isinstance(snapshot, dict) else {}
        pending = _count_orders_not_delivered(orders)
        delivered = _count_orders_with_status(orders, "delivered")
        dispatched = _count_assigned_orders(orders)

        base.update(
            {
                "scenario_id": self.scenario_id,
                "step_index": step,
                "simulated_time": _format_env_time(snapshot.get("time", 0.0) if isinstance(snapshot, dict) else 0.0),
                "pending_orders": pending,
                "dispatched_orders": dispatched,
                "delivered_orders": delivered,
                "vehicle_count": len(vehicles) if isinstance(vehicles, dict) else 0,
                "inventory_record_count": len(inventory) if isinstance(inventory, dict) else 0,
                "selected_action_id": action_id,
                "decoded_action": decoded,
                "ppo_controls": _display_projected_controls(projected_continuous, self._last_raw_continuous),
                "ppo_raw_output": [round(float(value), 4) for value in self._last_raw_continuous.reshape(-1)],
                "kpi_values": _live_kpis_from_info(snapshot, reward_components),
                "observation_summary": grouped_observation_summary(
                    observation=observation,
                    scenario_id=self.scenario_id,
                    snapshot=snapshot if isinstance(snapshot, dict) else {},
                    reward_components=reward_components if isinstance(reward_components, dict) else {},
                ),
                "capability_matrix": self.capability_matrix,
                "trajectory_label": LIVE_TRAJECTORY_LABEL,
                "live_snapshot_keys": sorted(snapshot.keys()) if isinstance(snapshot, dict) else [],
                "recent_action_history": _append_history(
                    base.get("recent_action_history", []),
                    step=step,
                    action_id=action_id,
                    decoded=decoded,
                    source=self.policy_algorithm,
                ),
            }
        )
        base.update(self._common_runtime_fields())
        return base


def demo_raw_continuous_action(scenario_id: str, step: int) -> np.ndarray:
    stress = scenario_stress_profile(scenario_id)
    wave = math.sin(step / 4.0)
    return np.asarray(
        [
            _clamp(-0.35 + stress["inventory"] * 0.70 + wave * 0.05, -1.0, 1.0),
            _clamp(-0.10 + stress["demand"] * 0.85 - stress["capacity"] * 0.20, -1.0, 1.0),
            _clamp(-0.05 + stress["sla"] * 0.35 + stress["route"] * 0.15, -1.0, 1.0),
            _clamp(-0.15 + stress["lead_time"] * 0.80 + stress["inventory"] * 0.25, -1.0, 1.0),
            _clamp(-0.50 + stress["capacity"] * 0.90 + max(wave, 0.0) * 0.10, -1.0, 1.0),
        ],
        dtype=np.float32,
    )


def grouped_observation_summary(
    *,
    observation: np.ndarray,
    scenario_id: str,
    snapshot: dict[str, Any],
    reward_components: dict[str, Any],
) -> dict[str, Any]:
    obs = np.asarray(observation, dtype=np.float32).reshape(-1)
    orders = snapshot.get("orders", {}) if isinstance(snapshot, dict) else {}
    vehicles = snapshot.get("vehicles", {}) if isinstance(snapshot, dict) else {}
    inventory = snapshot.get("inventory", {}) if isinstance(snapshot, dict) else {}
    stress = scenario_stress_profile(scenario_id)
    return {
        "talep/SLA": {
            "gözlem_aralığı": "0..14",
            "aktif_sipariş": len(orders) if isinstance(orders, dict) else 0,
            "bekleyen_sipariş": _count_orders_not_delivered(orders),
            "servis_seviyesi": round(float(snapshot.get("service_level", 0.0)), 4),
            "gecikme_baskısı": round(_safe_float(reward_components.get("true_lateness_pressure")), 4),
        },
        "filo/kapasite": {
            "gözlem_aralığı": "15..29",
            "araç_sayısı": len(vehicles) if isinstance(vehicles, dict) else 0,
            "kapasite_baskısı": round(_safe_float(reward_components.get("capacity_pressure")), 4),
            "araç_yok_sayacı": int(_safe_float(reward_components.get("dispatch_no_vehicle_available"))),
        },
        "rota/sıkışıklık": {
            "gözlem_aralığı": "30..44",
            "disruption_score": round(float(snapshot.get("disruption_score", 0.0)), 4),
            "rota_hatası": int(_safe_float(reward_components.get("dispatch_route_failure"))),
            "en_iyi_aday_skoru": round(_safe_float(reward_components.get("best_route_candidate_score")), 4),
        },
        "stok/ikmal": {
            "gözlem_aralığı": "45..59",
            "stok_kayıt_sayısı": len(inventory) if isinstance(inventory, dict) else 0,
            "stok_baskısı": round(_safe_float(reward_components.get("inventory_shortfall_penalty")), 4),
            "ikmal_maliyeti": round(_safe_float(reward_components.get("dqn_replenishment_override_cost")), 4),
        },
        "senaryo stresi": {
            "gözlem_aralığı": "60..72",
            "senaryo": scenario_id,
            "stress_profile": stress,
            "gözlem_min": round(float(obs.min()) if obs.size else 0.0, 4),
            "gözlem_max": round(float(obs.max()) if obs.size else 0.0, 4),
            "özet": "73-D gözlem canlı env çıktısından gruplanarak gösterilir.",
        },
    }


def _display_projected_controls(projected_continuous: Any, raw_continuous: np.ndarray) -> dict[str, float]:
    if isinstance(projected_continuous, dict) and projected_continuous:
        return {
            name: round(float(projected_continuous.get(name, 0.0)), 4)
            for name in PPO_CONTROLS
        }
    raw = np.asarray(raw_continuous, dtype=np.float32).reshape(-1)
    return {name: round(float(raw[index]) if index < raw.size else 0.0, 4) for index, name in enumerate(PPO_CONTROLS)}


def _live_kpis_from_info(snapshot: Any, reward_components: Any) -> dict[str, float]:
    snap = snapshot if isinstance(snapshot, dict) else {}
    rewards = reward_components if isinstance(reward_components, dict) else {}
    return {
        "service_level": round(float(snap.get("service_level", 0.0)), 4),
        "lateness": round(_safe_float(rewards.get("true_lateness_pressure")), 5),
        "dispatch_rate": round(_safe_float(rewards.get("dqn_dispatch_requested")), 5),
        "dispatch_success": round(_safe_float(rewards.get("dispatch_success_rate")), 5),
        "no_work": round(
            _safe_float(rewards.get("dispatch_no_unassigned_orders"))
            + _safe_float(rewards.get("dqn_delivery_credit_blocked_no_current_dispatch_work")),
            5,
        ),
        "route_failure": round(_safe_float(rewards.get("dispatch_route_failure")), 5),
        "no_vehicle": round(_safe_float(rewards.get("dispatch_no_vehicle_available")), 5),
        "already_assigned": round(_safe_float(rewards.get("dispatch_already_assigned_count")), 5),
        "stockout_reorder_pressure": round(_safe_float(rewards.get("inventory_shortfall_penalty")), 5),
    }


def _append_history(history: list[dict[str, Any]], *, step: int, action_id: int, decoded: dict[str, Any], source: str) -> list[dict[str, Any]]:
    rows = list(history)
    rows.append(
        {
            "step": step,
            "action_id": action_id,
            "route_family_tr": decoded["route_family_tr"],
            "interpretation_tr": decoded["operational_interpretation_tr"],
            "source": source,
        }
    )
    return rows[-MAX_ACTION_HISTORY:]


def _count_orders_with_status(orders: Any, status: str) -> int:
    if not isinstance(orders, dict):
        return 0
    return sum(1 for row in orders.values() if isinstance(row, dict) and str(row.get("status")) == status)


def _count_orders_not_delivered(orders: Any) -> int:
    if not isinstance(orders, dict):
        return 0
    return sum(1 for row in orders.values() if isinstance(row, dict) and str(row.get("status")) != "delivered")


def _count_assigned_orders(orders: Any) -> int:
    if not isinstance(orders, dict):
        return 0
    return sum(1 for row in orders.values() if isinstance(row, dict) and row.get("assigned_vehicle_id"))


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return number


def _format_env_time(seconds: Any) -> str:
    total_minutes = int(_safe_float(seconds) // 60)
    return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"


def build_kpi_timeline() -> list[dict[str, Any]]:
    scenarios = [
        ("baseline_normal", 0.990, 0.000, 0.372, 1.000, 0.000000, 0.000174, 0.000521, 0.193, 0.0, 0.0),
        ("demand_spike_volatility", 0.859, 0.027, 0.793, 1.000, 0.000000, 0.541, 0.002257, 0.144618, 0.0, 0.0),
        ("high_holding_cost", 0.966, 0.000, 0.449, 0.999, 0.000347, 0.240, 0.001389, 0.000000, 0.0, 0.0),
        ("lead_time_volatility", 1.000, 0.000, 0.360, 0.999, 0.000174, 0.080, 0.000347, 0.000347, 0.0, 0.0),
        ("mixed_stress", 0.942, 0.000, 0.987, 0.999, 0.001389, 0.659, 0.001736, 0.050868, 0.58, 0.000009),
        ("premium_sla_pressure", 1.000, 0.000, 0.357, 0.999, 0.000174, 0.078, 0.000347, 0.000347, 0.0, 0.0),
        ("route_disruption_congestion", 0.901, 0.008, 0.499, 0.999, 0.000347, 0.260, 0.001389, 0.000000, 0.001097, 0.47),
        ("vehicle_scarcity_capacity_shock", 0.949, 0.001, 0.806, 1.000, 0.000000, 0.447, 0.001389, 0.000347, 0.0, 0.0),
    ]
    return [
        {
            "scenario_id": name,
            "scenario_tr": scenario_name_tr(name, name),
            "service_level": service,
            "lateness": lateness,
            "dispatch_rate": dispatch,
            "dispatch_success": success,
            "no_work_per_step": no_work,
            "top_action_rate": top_action,
            "route_failure_per_step": route_fail,
            "no_vehicle_per_step": no_vehicle,
            "no_current_per_step": "statik güvence tablosunda yok",
            "no_unassigned_per_step": "statik güvence tablosunda yok",
            "action_24_rate_watch": action24,
            "action_32_rate_watch": action32,
            "trajectory_label": STATIC_TRAJECTORY_LABEL,
        }
        for name, service, lateness, dispatch, success, no_work, top_action, route_fail, no_vehicle, action24, action32 in scenarios
    ]


def build_dashboard_state(root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    scenarios = discover_scenarios(root / "configs" / "eval_scenarios")
    replay = load_public_replay_distribution(
        root / "Tez" / "FINAL_EKLER" / "EK_A_48_ACTION_MAPPING_AND_PUBLIC_REPLAY_COUNTS.md",
        root / "Tez" / "FINAL_EKLER" / "EK_A2_ACTION_FAMILY_DISTRIBUTION_SUMMARY.md",
    )
    actions = merge_action_replay_counts(build_action_decoder_rows(), replay["actions"])
    backend = create_dashboard_backend(
        root,
        scenario_id="baseline_normal",
        mode=CANLI_SIMULASYON_MODU,
        use_production_policy=False,
    )
    try:
        initial_simulation = backend.current_state()
    finally:
        backend.close()
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "page_tabs": PAGE_TABS,
        "overview": {
            "title": TITLE,
            "model_identity": MODEL_ID,
            "contract": CONTRACT,
            "observation_dimension": OBSERVATION_DIM,
            "ppo_continuous_controls": PPO_CONTROLS,
            "dqn_discrete_actions": DQN_ACTION_COUNT,
            "architecture": ARCHITECTURE,
            "statement": DIGITAL_TWIN_STATEMENT,
            "scope": SIMULATION_SCOPE_STATEMENT,
            "terms_tr": [
                "PPO sürekli kontroller: beş fiziksel kontrol sinyali",
                "DQN ayrık aksiyon: 0..47 arası taktik karar",
                "Dijital ikiz ortamı: 5PL operasyonlarının simülasyonu",
                "Canlı ürün demosu: güvenli ve bounded env_5pl reset/step döngüsü",
                "Kamu replay = betimleyici proxy analiz",
            ],
        },
        "scenarios": scenarios,
        "zone_view": build_synthetic_zone_view(),
        "simulation": initial_simulation,
        "map_payload": build_map_figure_payload(initial_simulation),
        "decision_trace": build_decision_trace_sample(),
        "actions": actions,
        "public_replay": replay,
        "kpi_timeline": build_kpi_timeline(),
        "synthetic_trajectories": {
            scenario["scenario_id"]: build_synthetic_kpi_trajectory(scenario["scenario_id"])
            for scenario in scenarios
        },
        "evidence_limitations": [
            "İç simülatör kapıları ve eş bütçeli karşılaştırma, simülatör sınırları içinde davranışı destekler.",
            "Kamu proxy kanıtı rota, filo/sevk ve stok/ikmal eğilimlerini betimleyici olarak gösterir.",
            "Şirket verisi; özel maliyetler, operasyonel kalibrasyon ve canlıya alma doğrulaması için hâlâ gereklidir.",
            "Bu çalışma canlı şirket devreye alımı, global üstünlük veya kamu replay üzerinden karşı-olgusal üstünlük iddia etmez.",
            PUBLIC_REPLAY_BOUNDARY,
        ],
        "modes": {
            "available": [CANLI_SIMULASYON_MODU, GUVENLI_SENTETIK_MOD, PUBLIC_REPLAY_MODU],
            "labels_tr": MODE_LABELS_TR,
            "default_mode": CANLI_SIMULASYON_MODU,
            "live_backend_source": initial_simulation["backend_source"],
            "safe_fallback_active": initial_simulation["safe_fallback_active"],
            "fallback_reason": initial_simulation["fallback_reason"],
            "policy_read_only_default": False,
            "html_fallback": "Bu statik önizlemedir. Canlı ürün demosu için Streamlit çalıştırın.",
        },
    }


def dashboard_text_corpus() -> str:
    state = build_dashboard_state(repo_root())
    return json.dumps(state, ensure_ascii=False)


def export_html_dashboard(state: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_text = render_html_dashboard(state)
    output_path.write_text(html_text, encoding="utf-8")


def render_html_dashboard(state: dict[str, Any]) -> str:
    overview = state["overview"]
    return f"""<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(overview['title'])} - Canlı Türkçe Dijital İkiz Kontrol Odası V3</title>
  <style>
    :root {{ color-scheme: light; --ink:#17202a; --muted:#5f6c7b; --line:#d7dde5; --panel:#f7f9fc; --accent:#0f766e; --warn:#a16207; --risk:#b42318; --blue:#1d4ed8; --live:#16a34a; }}
    body {{ margin:0; font-family:Segoe UI, Arial, sans-serif; color:var(--ink); background:#f5f7fb; }}
    header {{ min-height:62vh; padding:38px 34px 28px; background:linear-gradient(135deg,#0f172a,#115e59); color:white; display:flex; flex-direction:column; justify-content:center; }}
    header h1 {{ margin:0 0 12px; font-size:clamp(30px,4vw,54px); line-height:1.08; max-width:1160px; }}
    header p {{ max-width:1040px; margin:7px 0; color:#dbeafe; font-size:17px; }}
    main {{ padding:24px 34px 40px; max-width:1420px; margin:auto; }}
    nav {{ position:sticky; top:0; background:#ffffff; border-bottom:1px solid var(--line); padding:10px 34px; z-index:2; display:flex; gap:10px; overflow:auto; }}
    nav a {{ color:#0f172a; text-decoration:none; border:1px solid var(--line); padding:7px 10px; border-radius:6px; white-space:nowrap; }}
    section {{ border-top:1px solid var(--line); padding:26px 0; }}
    h2 {{ margin:0 0 14px; font-size:24px; }}
    h3 {{ margin:18px 0 10px; font-size:17px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; }}
    .metric {{ background:#ffffff; border:1px solid var(--line); border-radius:8px; padding:14px; box-shadow:0 1px 2px rgba(15,23,42,.05); }}
    .metric strong {{ display:block; font-size:20px; margin-top:4px; }}
    .note {{ background:#eef6f5; border-left:4px solid var(--accent); padding:12px 14px; margin:12px 0; }}
    .warn {{ background:#fff8e6; border-left:4px solid var(--warn); padding:12px 14px; margin:12px 0; }}
    .live {{ background:#ecfdf3; border-left:4px solid var(--live); padding:12px 14px; margin:12px 0; }}
    .control-strip {{ display:flex; flex-wrap:wrap; gap:8px; margin:12px 0; }}
    .control-strip button {{ border:1px solid var(--line); background:#ffffff; border-radius:6px; padding:9px 12px; font-weight:600; }}
    table {{ border-collapse:collapse; width:100%; font-size:13px; }}
    th,td {{ border:1px solid var(--line); padding:7px 8px; text-align:left; vertical-align:top; }}
    th {{ background:#eef2f7; }}
    .scroll {{ overflow:auto; max-height:500px; border:1px solid var(--line); border-radius:8px; }}
    .map {{ width:100%; max-width:850px; border:1px solid var(--line); border-radius:8px; background:#fbfdff; }}
    .bar {{ height:10px; background:#dbeafe; border-radius:999px; overflow:hidden; display:block; }}
    .bar span {{ display:block; height:10px; background:var(--accent); }}
    .legend-dot {{ display:inline-block; width:12px; height:12px; border-radius:50%; margin-right:6px; vertical-align:middle; }}
    code {{ background:#eef2f7; padding:1px 4px; border-radius:4px; }}
    select,input {{ max-width:100%; padding:8px; border:1px solid var(--line); border-radius:6px; background:white; }}
    .pill {{ display:inline-block; padding:3px 7px; border-radius:999px; background:#e0f2fe; margin:2px; }}
  </style>
  <script>
    function showScenario(id) {{
      document.querySelectorAll('[data-scenario-detail]').forEach(function(el) {{
        el.style.display = el.getAttribute('data-scenario-detail') === id ? 'block' : 'none';
      }});
    }}
    function filterActions() {{
      var route = document.getElementById('route-filter').value;
      var dispatch = document.getElementById('dispatch-filter').value;
      var fleet = document.getElementById('fleet-filter').value;
      var reorder = document.getElementById('reorder-filter').value;
      var q = document.getElementById('action-search').value.toLowerCase();
      document.querySelectorAll('#action-table tbody tr').forEach(function(row) {{
        var routeOk = !route || row.getAttribute('data-route') === route;
        var dispatchOk = !dispatch || row.getAttribute('data-dispatch') === dispatch;
        var fleetOk = !fleet || row.getAttribute('data-fleet') === fleet;
        var reorderOk = !reorder || row.getAttribute('data-reorder') === reorder;
        var qOk = !q || row.innerText.toLowerCase().includes(q);
        row.style.display = routeOk && dispatchOk && fleetOk && reorderOk && qOk ? '' : 'none';
      }});
    }}
  </script>
</head>
<body>
  <header>
    <h1>{html.escape(overview['title'])}</h1>
    <p>{html.escape(overview['statement'])}</p>
    <p>{html.escape(overview['scope'])}</p>
    <p><strong>V3:</strong> Bu statik önizlemedir. Canlı ürün demosu için Streamlit çalıştırın.</p>
  </header>
  <nav>{''.join(f"<a href='#{section_id(tab)}'>{html.escape(tab)}</a>" for tab in PAGE_TABS)}</nav>
  <main>
    {render_overview_html(state)}
    {render_scenarios_html(state)}
    {render_zone_html(state)}
    {render_decision_trace_html(state)}
    {render_actions_html(state)}
    {render_replay_html(state)}
    {render_kpi_html(state)}
    {render_evidence_html(state)}
  </main>
</body>
</html>
"""


def section_id(label: str) -> str:
    return (
        label.lower()
        .replace(" ", "-")
        .replace("+", "plus")
        .replace("ı", "i")
        .replace("ğ", "g")
        .replace("ü", "u")
        .replace("ş", "s")
        .replace("ö", "o")
        .replace("ç", "c")
    )


def render_overview_html(state: dict[str, Any]) -> str:
    overview = state["overview"]
    sim = state["simulation"]
    metrics = [
        ("Model kimliği", overview["model_identity"]),
        ("Çalışma modu", sim["mode"]),
        ("Backend", sim["backend_source"]),
        ("Adım", str(sim["step_index"])),
        ("Servis", str(sim["kpi_values"].get("service_level"))),
        ("DQN aksiyon", str(sim["selected_action_id"])),
        ("Sözleşme", overview["contract"]),
        ("Gözlem boyutu", str(overview["observation_dimension"])),
    ]
    cards = "\n".join(f"<div class='metric'>{html.escape(label)}<strong>{html.escape(value)}</strong></div>" for label, value in metrics)
    terms = "".join(f"<span class='pill'>{html.escape(term)}</span>" for term in overview["terms_tr"])
    return (
        f"<section id='{section_id('Kontrol Odası')}'><h2>Kontrol Odası</h2>"
        f"<p class='live'>Canlı Streamlit modunda bu bölüm otomatik ilerleyen env_5pl kontrol odasıdır; bu HTML dosyası yalnızca statik önizlemedir.</p>"
        f"<div class='control-strip'><button>Başlat</button><button>Durdur</button><button>Sıfırla</button><button>Tek Adım</button><button>Hız 1x</button></div>"
        f"<div class='grid'>{cards}</div><p class='note'>{html.escape(DIGITAL_TWIN_STATEMENT)}</p>"
        f"<p>{terms}</p>"
        f"<p class='warn'>{html.escape(state['modes']['html_fallback'])}</p></section>"
    )


def render_scenarios_html(state: dict[str, Any]) -> str:
    rows = [
        [
            scenario["scenario_id"],
            scenario["name_tr"],
            scenario["status_tr"],
            scenario["stress_type_tr"],
            scenario["expected_operational_pressure_tr"],
        ]
        for scenario in state["scenarios"]
    ]
    options = "".join(
        f"<option value='{html.escape(scenario['scenario_id'])}'>{html.escape(scenario['name_tr'])}</option>"
        for scenario in state["scenarios"]
    )
    detail_cards = "".join(
        "<div class='metric' data-scenario-detail='{}' style='display:{}'><strong>{}</strong><p>{}</p><p><b>Durum:</b> {}<br><b>Stres tipi:</b> {}<br><b>Operasyonel baskı:</b> {}</p></div>".format(
            html.escape(scenario["scenario_id"]),
            "block" if index == 0 else "none",
            html.escape(scenario["name_tr"]),
            html.escape(scenario["description_tr"]),
            html.escape(scenario["status_tr"]),
            html.escape(scenario["stress_type_tr"]),
            html.escape(scenario["expected_operational_pressure_tr"]),
        )
        for index, scenario in enumerate(state["scenarios"])
    )
    return (
        f"<section id='{section_id('Canlı Senaryo')}'><h2>Canlı Senaryo</h2>"
        "<p>Senaryolar <code>configs/eval_scenarios</code> klasöründen dinamik bulunur; çalışan JSON senaryoları canlı env reset/step döngüsüne bağlanır.</p>"
        "<label for='scenario-select'><strong>Senaryo seçimi</strong></label><br>"
        f"<select id='scenario-select' onchange='showScenario(this.value)'>{options}</select>"
        f"<p class='note'>Varsayılan mod: <code>{CANLI_SIMULASYON_MODU}</code>. Eğer env bağlanamazsa neden gösterilerek <code>{GUVENLI_SENTETIK_MOD}</code> kullanılır.</p>"
        f"<div class='grid' style='margin-top:12px'>{detail_cards}</div>"
        + html_table(["Senaryo", "Ad", "Durum", "Stres tipi", "Beklenen operasyonel baskı"], rows)
        + "</section>"
    )


def render_zone_html(state: dict[str, Any]) -> str:
    sim_state = state["simulation"]
    payload = build_map_figure_payload(sim_state)
    svg = render_zone_svg(payload)
    legend_rows = [
        ["Hub/depo", "Yeşil kare", "Sentetik çıkış veya aktarma noktası"],
        ["Araç", "Turuncu üçgen", "Sentetik filodaki hareketli araç"],
        ["Bekleyen sipariş", "Mavi/gri/kırmızı nokta", "Öncelik durumuna göre sipariş"],
        ["Sıkışıklık/kesinti", "Rota adayı tablosu", "Her aday rotada sıkışıklık ve kesinti bayrağı"],
        ["Seçilen aksiyonun rota etkisi", "Kalın yeşil çizgi", "DQN aksiyonundaki rota ailesinin vurgusu"],
    ]
    route_rows = [
        [
            row["from"],
            row["to"],
            row["family_tr"],
            row["congestion_tr"],
            "evet" if row["disruption"] else "hayır",
            "seçili" if row["highlighted"] else "",
        ]
        for row in payload["route_candidates"]
    ]
    zones = ", ".join(state["zone_view"]["delivery_zones"])
    return (
        f"<section id='{section_id('Dijital İkiz Haritası')}'><h2>Dijital İkiz Haritası</h2>"
        f"<p class='note'>{html.escape(payload['note_tr'])}</p>"
        f"{svg}<p><strong>Teslimat bölgeleri:</strong> {html.escape(zones)}</p>"
        "<h3>Sentetik döngü durumu</h3>"
        + html_table(
            ["Adım", "Simüle zaman", "Bekleyen sipariş", "Sevk edilen", "Seçilen DQN aksiyon"],
            [[sim_state["step_index"], sim_state["simulated_time"], sim_state["pending_orders"], sim_state["dispatched_orders"], sim_state["selected_action_id"]]],
        )
        + "<h3>Lejant</h3>"
        + html_table(["Öğe", "Görsel işaret", "Anlam"], legend_rows)
        + "<h3>Rota adayları</h3>"
        + html_table(["Kaynak", "Hedef", "Rota ailesi", "Sıkışıklık", "Kesinti", "Etki"], route_rows)
        + "</section>"
    )


def render_zone_svg(payload: dict[str, Any]) -> str:
    order_lookup = {order["id"]: order for order in payload["orders"]}
    hub_lookup = {hub["id"]: hub for hub in payload["hubs"]}
    parts = [
        "<svg class='map' viewBox='0 0 100 100' role='img' aria-label='Sentetik dijital ikiz bölge haritası'>",
        "<defs><pattern id='grid' width='10' height='10' patternUnits='userSpaceOnUse'><path d='M 10 0 L 0 0 0 10' fill='none' stroke='#d7dde5' stroke-width='0.4'/></pattern></defs>",
        "<rect width='100' height='100' fill='url(#grid)'/>",
    ]
    for route in payload["route_candidates"]:
        source = hub_lookup.get(route["from"])
        target = order_lookup.get(route["to"])
        if not source or not target:
            continue
        color = "#0f766e" if route["highlighted"] else "#94a3b8"
        width = "2.4" if route["highlighted"] else "1.1"
        dash = "" if route["highlighted"] else " stroke-dasharray='3 2'"
        parts.append(
            f"<line x1='{source['x']}' y1='{source['y']}' x2='{target['x']}' y2='{target['y']}' stroke='{color}' stroke-width='{width}'{dash}/>"
        )
    for hub in payload["hubs"]:
        parts.append(
            f"<rect x='{float(hub['x']) - 3}' y='{float(hub['y']) - 3}' width='6' height='6' rx='1.2' fill='#0f766e'/>"
            f"<text x='{hub['x']}' y='{float(hub['y']) + 7}' text-anchor='middle' fill='#0f172a' font-size='4'>{html.escape(hub['id'])}</text>"
        )
    for order in payload["orders"]:
        fill = "#b42318" if order["priority_tr"] == "acil" else "#1d4ed8" if order["priority_tr"] == "premium" else "#64748b"
        parts.append(
            f"<circle cx='{order['x']}' cy='{order['y']}' r='2.5' fill='{fill}'/>"
            f"<text x='{order['x']}' y='{float(order['y']) - 3.5}' text-anchor='middle' fill='#0f172a' font-size='3.5'>{html.escape(order['zone'])}</text>"
        )
    for vehicle in payload["vehicles"]:
        parts.append(
            f"<polygon points='{float(vehicle['x'])},{float(vehicle['y']) - 3} {float(vehicle['x']) + 2.6},{float(vehicle['y']) + 2.6} {float(vehicle['x']) - 2.6},{float(vehicle['y']) + 2.6}' fill='#a16207'/>"
            f"<text x='{vehicle['x']}' y='{float(vehicle['y']) + 7}' text-anchor='middle' fill='#0f172a' font-size='3.5'>{html.escape(vehicle['id'])}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def render_decision_trace_html(state: dict[str, Any]) -> str:
    trace = state["simulation"]
    obs = [[key, json.dumps(value, ensure_ascii=False)] for key, value in trace["observation_summary"].items()]
    controls = [[key, value] for key, value in trace["ppo_controls"].items()]
    action = trace["decoded_action"]
    action_rows = [[key, value] for key, value in action.items()]
    kpi_rows = [[key, value] for key, value in trace["kpi_values"].items()]
    return (
        f"<section id='{section_id('PPO + DQN Karar Akışı')}'><h2>PPO + DQN Karar Akışı</h2>"
        f"<p>Backend: <code>{html.escape(trace['backend_source'])}</code>. Policy kaynağı: <code>{html.escape(trace['policy_source'])}</code>.</p>"
        + (f"<p class='warn'>Policy yükleme hatası: {html.escape(trace['policy_error'])}</p>" if trace.get("policy_error") else "")
        + "<div class='grid'><div>"
        + html_table(["73-D gözlem özeti", "Değer"], obs)
        + "</div><div>"
        + html_table(["PPO sürekli kontrol", "Değer"], controls)
        + "</div><div>"
        + html_table(["DQN alanı", "Değer"], action_rows)
        + "</div><div><h3>KPI değerleri</h3>"
        + html_table(["KPI", "Değer"], kpi_rows)
        + "</div></div><p class='note'>Hiyerarşik DQN içeride faktörize başlıklar kullanır; dış sözleşme 0..47 aksiyon olarak kalır.</p>"
        + "<p class='warn'>Aksiyon 24 ve aksiyon 32 yalnızca izlenen aksiyonlardır; model davranışının tamamı olarak yorumlanmaz.</p></section>"
    )


def render_actions_html(state: dict[str, Any]) -> str:
    headers = [
        "action_id",
        "sevk/beklet",
        "rota ailesi",
        "filo modu",
        "ikmal modu",
        "kamu replay sayı/oran",
        "izleme",
        "operasyonel yorum",
    ]
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = "\n".join(
        "<tr data-route='{}' data-dispatch='{}' data-fleet='{}' data-reorder='{}'><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{:,} / {:.6f}</td><td>{}</td><td>{}</td></tr>".format(
            html.escape(row["route_family"]),
            html.escape(row["dispatch"]),
            html.escape(row["fleet_mode"]),
            html.escape(row["reorder_mode"]),
            row["action_id"],
            html.escape(row["dispatch_tr"]),
            html.escape(row["route_family_tr"]),
            html.escape(row["fleet_mode_tr"]),
            html.escape(row["reorder_mode_tr"]),
            int(row["public_replay_global_count"]),
            float(row["public_replay_global_rate"]),
            html.escape(row["watch_tr"]),
            html.escape(row["operational_interpretation_tr"]),
        )
        for row in state["actions"]
    )
    table = f"<table id='action-table'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    filters = (
        "<div class='grid'>"
        "<label>Arama<br><input id='action-search' oninput='filterActions()' placeholder='aksiyon, rota, filo...'></label>"
        "<label>Rota ailesi<br><select id='route-filter' onchange='filterActions()'><option value=''>tümü</option><option value='shortest'>en kısa</option><option value='low_congestion'>düşük sıkışıklık</option><option value='high_resilience'>dayanıklı</option></select></label>"
        "<label>Sevk/beklet<br><select id='dispatch-filter' onchange='filterActions()'><option value=''>tümü</option><option value='hold'>beklet</option><option value='dispatch'>sevk et</option></select></label>"
        "<label>Filo modu<br><select id='fleet-filter' onchange='filterActions()'><option value=''>tümü</option><option value='secondary_fleet'>ikincil filo</option><option value='primary_fleet'>birincil filo</option></select></label>"
        "<label>İkmal modu<br><select id='reorder-filter' onchange='filterActions()'><option value=''>tümü</option><option value='none'>yok</option><option value='conservative'>temkinli</option><option value='aggressive'>agresif</option><option value='emergency'>acil</option></select></label>"
        "</div>"
    )
    return (
        f"<section id='{section_id('48 Aksiyon Sözlüğü')}'><h2>48 Aksiyon Sözlüğü</h2>"
        "<p>Aksiyon 24 ve aksiyon 32 yalnızca izlenen aksiyonlardır; model davranışının tamamı değildir.</p>"
        + filters
        + "<div class='scroll'>"
        + table
        + "</div></section>"
    )


def render_replay_html(state: dict[str, Any]) -> str:
    replay = state["public_replay"]
    total_cards = "".join(
        f"<div class='metric'>{html.escape(name)} satır<strong>{value:,}</strong></div>"
        for name, value in replay["dataset_totals"].items()
    )
    total_cards += f"<div class='metric'>Toplam satır<strong>{replay['global_total']:,}</strong></div>"
    top_rows = [[row["action_id"], row["global_count"], row["global_rate"], row["interpretation_tr"]] for row in replay["top_10_actions"]]
    action_rows = [
        [
            row["action_id"],
            row["dispatch_tr"],
            row["route_family_tr"],
            row["fleet_mode_tr"],
            row["reorder_mode_tr"],
            row["lade_count"],
            row["nyc_count"],
            row["olist_count"],
            row["global_count"],
            f"{row['global_rate']:.6f}",
            row["watch_tr"],
        ]
        for row in replay["actions"]
    ]
    zero_ids = ", ".join(str(row["action_id"]) for row in replay["zero_count_actions"])
    family_rows = []
    for family, counts in replay["family_distribution"].items():
        if family == "source":
            continue
        for value, count in counts.items():
            label = {
                "dispatch": "sevk/beklet",
                "route_family": "rota ailesi",
                "fleet_mode": "filo modu",
                "reorder_mode": "ikmal modu",
            }.get(family, family)
            translated = (
                TURKISH_DISPATCH.get(value)
                or TURKISH_ROUTE.get(value)
                or TURKISH_FLEET.get(value)
                or TURKISH_REORDER.get(value)
                or value
            )
            family_rows.append([label, translated, count, f"{count / max(replay['global_total'], 1):.6f}"])
    bars = "".join(
        f"<p><strong>{row['action_id']}</strong> {html.escape(row['interpretation_tr'])}<br><span class='bar'><span style='width:{min(row['global_rate'] * 100, 100):.3f}%'></span></span> {row['global_count']:,}</p>"
        for row in replay["top_10_actions"]
    )
    return (
        f"<section id='{section_id('Kamu Replay Analizi')}'><h2>Kamu Replay Analizi</h2>"
        f"<p class='warn'>{html.escape(PUBLIC_REPLAY_BOUNDARY)}</p>"
        f"<p>{html.escape(replay['dataset_selector_note'])}</p>"
        "<label for='dataset-filter'><strong>Veri kümesi seçici</strong></label><br>"
        "<select id='dataset-filter'><option>Tümü</option><option>LaDe</option><option>NYC HVFHS</option><option>Olist</option></select>"
        f"<div class='grid'>{total_cards}</div>"
        "<h3>Tüm Aksiyon Kamu Replay Dağılımı 0..47</h3>"
        "<div class='scroll'>"
        + html_table(["Aksiyon", "Sevk", "Rota", "Filo", "İkmal", "LaDe", "NYC HVFHS", "Olist", "Global sayı", "Global oran", "İzleme"], action_rows)
        + "</div><h3>Top 10 aksiyon</h3>"
        + bars
        + html_table(["Aksiyon", "Sayı", "Oran", "Çözümleme"], top_rows)
        + f"<h3>Sıfır sayımlı aksiyonlar</h3><p>{html.escape(zero_ids)}</p><h3>Aile dağılımı</h3>"
        + html_table(["Aile", "Değer", "Global sayı", "Oran"], family_rows)
        + "</section>"
    )


def render_kpi_html(state: dict[str, Any]) -> str:
    rows = [
        [
            row["scenario_id"],
            row["scenario_tr"],
            row["service_level"],
            row["lateness"],
            row["dispatch_rate"],
            row["dispatch_success"],
            row["no_work_per_step"],
            row["route_failure_per_step"],
            row["no_vehicle_per_step"],
            row["no_current_per_step"],
            row["no_unassigned_per_step"],
            row["top_action_rate"],
            row["action_24_rate_watch"],
            row["action_32_rate_watch"],
            row["trajectory_label"],
        ]
        for row in state["kpi_timeline"]
    ]
    trajectory = state["synthetic_trajectories"]["route_disruption_congestion"][:8]
    trajectory_rows = [
        [row["step"], row["simulated_time"], row["service_level"], row["dispatch_rate"], row["route_failure"], row["selected_action_id"], row["trajectory_label"]]
        for row in trajectory
    ]
    distribution_rows = [
        [row["action_id"], row["count"], f"{row['rate']:.3f}", row["route_family_tr"], row["watch_tr"]]
        for row in summarize_action_distribution_over_time(state["synthetic_trajectories"]["route_disruption_congestion"])
    ]
    return (
        f"<section id='{section_id('KPI ve Zaman Çizgisi')}'><h2>KPI ve Zaman Çizgisi</h2>"
        f"<p class='note'>Streamlit modunda KPI'lar canlı env adımlarıyla güncellenir; statik önizlemede güvenli zaman çizgisi gösterilir. Offline eval çalıştırılmaz.</p>"
        + html_table(
            ["Senaryo", "TR ad", "Service", "Lateness", "Dispatch rate", "Dispatch success", "No-work/step", "Route failure/step", "No-vehicle/step", "No-current/step", "No-unassigned/step", "Top action rate", "Action 24 watch", "Action 32 watch", "Etiket"],
            rows,
        )
        + "<h3>Örnek sentetik zaman çizgisi</h3>"
        + html_table(["Adım", "Zaman", "Servis", "Sevk oranı", "Rota hatası", "Aksiyon", "Etiket"], trajectory_rows)
        + "<h3>Aksiyon dağılımı zaman özeti</h3>"
        + html_table(["Aksiyon", "Sayı", "Oran", "Rota ailesi", "İzleme"], distribution_rows)
        + "</section>"
    )


def summarize_action_distribution_over_time(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(int(row["selected_action_id"]) for row in trajectory)
    total = sum(counts.values()) or 1
    decoder = {row["action_id"]: row for row in build_action_decoder_rows()}
    return [
        {
            "action_id": action_id,
            "count": count,
            "rate": count / total,
            "dispatch_tr": decoder[action_id]["dispatch_tr"],
            "route_family_tr": decoder[action_id]["route_family_tr"],
            "fleet_mode_tr": decoder[action_id]["fleet_mode_tr"],
            "reorder_mode_tr": decoder[action_id]["reorder_mode_tr"],
            "watch_tr": decoder[action_id]["watch_tr"],
        }
        for action_id, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def render_evidence_html(state: dict[str, Any]) -> str:
    items = "".join(f"<li>{html.escape(item)}</li>" for item in state["evidence_limitations"])
    return f"<section id='{section_id('Kanıtlar ve Sınırlar')}'><h2>Kanıtlar ve Sınırlar</h2><ul>{items}</ul></section>"


def html_table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_streamlit_dashboard(state: dict[str, Any]) -> None:
    if st is None:
        raise RuntimeError("Streamlit kurulu değil. Bağımsız yedek için --export-html kullanın.")
    st.set_page_config(page_title="Canlı Türkçe Dijital İkiz Kontrol Odası V3", layout="wide")
    st.markdown(streamlit_chrome_cleanup_css(), unsafe_allow_html=True)
    st.title("Canlı Türkçe Dijital İkiz Kontrol Odası V3")
    st.caption(TITLE)
    st.info(DIGITAL_TWIN_STATEMENT)
    st.warning(SIMULATION_SCOPE_STATEMENT)

    scenario_ids = [scenario["scenario_id"] for scenario in state["scenarios"]]
    defaults = initial_dashboard_session_state("baseline_normal")
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if "dashboard_backend" not in st.session_state:
        st.session_state.dashboard_backend = create_dashboard_backend(
            repo_root(),
            scenario_id=st.session_state.scenario_id,
            mode=st.session_state.mode,
            use_production_policy=st.session_state.use_production_policy,
        )

    with st.sidebar:
        st.header("Kontrol Paneli")
        selected_scenario = st.selectbox(
            "Senaryo seç",
            scenario_ids,
            index=scenario_ids.index(st.session_state.scenario_id) if st.session_state.scenario_id in scenario_ids else 0,
        )
        selected_mode = st.selectbox(
            "Mod seç",
            [CANLI_SIMULASYON_MODU, GUVENLI_SENTETIK_MOD, PUBLIC_REPLAY_MODU],
            format_func=lambda value: MODE_LABELS_TR.get(value, value),
            index=[CANLI_SIMULASYON_MODU, GUVENLI_SENTETIK_MOD, PUBLIC_REPLAY_MODU].index(st.session_state.mode),
        )
        use_policy = st.toggle("Production policy read-only kullan", value=bool(st.session_state.use_production_policy))
        speed_label = st.select_slider("Hız", options=["0.25x", "0.5x", "1x", "2x", "5x"], value=st.session_state.speed)

        changed = (
            selected_scenario != st.session_state.scenario_id
            or selected_mode != st.session_state.mode
            or use_policy != st.session_state.use_production_policy
        )
        if changed:
            st.session_state.scenario_id = selected_scenario
            st.session_state.mode = selected_mode
            st.session_state.use_production_policy = use_policy
            st.session_state.running = False
            st.session_state.dashboard_backend = create_dashboard_backend(
                repo_root(),
                scenario_id=selected_scenario,
                mode=selected_mode,
                use_production_policy=use_policy,
            )
        st.session_state.speed = speed_label

        c1, c2 = st.columns(2)
        if c1.button("Başlat", use_container_width=True):
            st.session_state.running = True
        if c2.button("Durdur", use_container_width=True):
            st.session_state.running = False
        c3, c4 = st.columns(2)
        if c3.button("Sıfırla", use_container_width=True):
            st.session_state.dashboard_backend.reset(
                scenario_id=st.session_state.scenario_id,
                mode=st.session_state.mode,
                use_production_policy=st.session_state.use_production_policy,
            )
            st.session_state.running = False
        if c4.button("Tek Adım", use_container_width=True):
            st.session_state.dashboard_backend.step()
        st.caption("Streamlit Cloud hesabı gerekmez; yerel Streamlit oturumu yeterlidir.")

    if should_auto_advance(bool(st.session_state.running), str(st.session_state.mode)):
        if st_autorefresh is not None:
            st_autorefresh(interval=autorun_interval_ms(str(st.session_state.speed)), key="dashboard_autorefresh")
        st.session_state.dashboard_backend.step()

    live_state = st.session_state.dashboard_backend.current_state()
    if live_state["safe_fallback_active"]:
        st.warning(live_state["fallback_reason"])
    else:
        st.success(f"Canlı backend bağlı: {live_state['backend_source']}")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Adım", live_state["step_index"])
    k2.metric("Servis", live_state["kpi_values"].get("service_level"))
    k3.metric("Bekleyen", live_state["pending_orders"])
    k4.metric("Seçili DQN", live_state["selected_action_id"])
    k5.metric("Mod", live_state["mode_label_tr"])

    tabs = st.tabs(PAGE_TABS)
    with tabs[0]:
        _streamlit_overview(state, live_state)
    with tabs[1]:
        _streamlit_scenario_runner(state, live_state)
    with tabs[2]:
        _streamlit_map(live_state)
    with tabs[3]:
        _streamlit_decision_trace(live_state)
    with tabs[4]:
        _streamlit_actions(state)
    with tabs[5]:
        _streamlit_replay(state)
    with tabs[6]:
        _streamlit_kpi(live_state)
    with tabs[7]:
        _streamlit_evidence(state)


def _streamlit_df(rows: list[dict[str, Any]] | list[list[Any]]) -> Any:
    return pd.DataFrame(rows) if pd is not None else rows


def _streamlit_overview(state: dict[str, Any], live_state: dict[str, Any]) -> None:
    overview = state["overview"]
    st.subheader("Kontrol Odası")
    st.caption("Canlı Streamlit modunda senaryo, env adımı, PPO+DQN aksiyonu ve KPI kartları birlikte akar.")
    cols = st.columns(3)
    metrics = [
        ("Model kimliği", overview["model_identity"]),
        ("Backend", live_state["backend_source"]),
        ("Policy", live_state["policy_source"]),
        ("Sözleşme", overview["contract"]),
        ("Gözlem boyutu", overview["observation_dimension"]),
        ("Mimari", overview["architecture"]),
    ]
    for col, (label, value) in zip(cols * 2, metrics):
        col.metric(label, value)
    if live_state["policy_error"]:
        st.warning(f"Üretim policy yükleme hatası: {live_state['policy_error']}")
    if live_state["safe_fallback_active"]:
        st.error(live_state["fallback_reason"])
    st.write("PPO sürekli kontroller:", ", ".join(overview["ppo_continuous_controls"]))
    for term in overview["terms_tr"]:
        st.markdown(f"- {term}")


def _streamlit_scenario_runner(state: dict[str, Any], live_state: dict[str, Any]) -> None:
    st.subheader("Canlı Senaryo")
    st.markdown(f"**Seçili senaryo:** `{live_state['scenario_id']}`")
    st.markdown(f"**Çalışma modu:** `{live_state['mode']}`")
    st.markdown(f"**Backend:** {live_state['backend_source']}")
    scenarios = state["scenarios"]
    selected = st.selectbox("Senaryo detay seçimi", [row["scenario_id"] for row in scenarios], key="scenario_detail_select")
    scenario = next(row for row in scenarios if row["scenario_id"] == selected)
    st.markdown(f"**{scenario['name_tr']}**")
    st.write(scenario["description_tr"])
    st.dataframe(_streamlit_df(scenarios), use_container_width=True)


def _streamlit_map(sim_state: dict[str, Any]) -> None:
    st.subheader("Dijital İkiz Haritası")
    payload = build_map_figure_payload(sim_state)
    st.caption(payload["note_tr"])
    fig = build_plotly_map_figure(payload)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown(render_zone_svg(payload), unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Adım", sim_state["step_index"])
    c2.metric("Simüle zaman", sim_state["simulated_time"])
    c3.metric("Bekleyen sipariş", sim_state["pending_orders"])
    c4.metric("Sevk edilen", sim_state["dispatched_orders"])
    st.markdown("**Lejant**")
    st.markdown(
        """
- Hub/depo: yeşil kare.
- Araç: turuncu üçgen.
- Bekleyen sipariş: mavi/gri/kırmızı nokta.
- Sıkışıklık/kesinti: rota adayı tablosundaki bayraklar.
- Seçilen aksiyonun rota etkisi: kalın yeşil çizgi.
"""
    )
    st.markdown("**Rota adayları ve sıkışıklık/kesinti bayrakları**")
    st.dataframe(_streamlit_df(payload["route_candidates"]), use_container_width=True)


def _streamlit_decision_trace(sim_state: dict[str, Any]) -> None:
    st.subheader("PPO + DQN Karar Akışı")
    st.caption("73-D gözlem ham satırlar yerine canlı env çıktısından operasyonel baskı grupları olarak gösterilir.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**73-D gözlem özeti**")
        st.json(sim_state["observation_summary"])
    with c2:
        st.markdown("**PPO sürekli kontroller**")
        st.json(sim_state["ppo_controls"])
    with c3:
        st.markdown("**DQN ayrık aksiyon**")
        st.json(sim_state["decoded_action"])
    st.markdown("**KPI değerleri**")
    st.json(sim_state["kpi_values"])
    st.warning("Aksiyon 24 ve aksiyon 32 izlenen aksiyonlardır; modelin tamamını temsil etmez.")
    st.markdown("**Son aksiyon geçmişi**")
    st.dataframe(_streamlit_df(sim_state["recent_action_history"]), use_container_width=True)


def _streamlit_actions(state: dict[str, Any]) -> None:
    rows = state["actions"]
    st.subheader("48 Aksiyon Sözlüğü")
    search = st.text_input("Arama", key="action_search")
    c1, c2, c3, c4 = st.columns(4)
    route = c1.multiselect("Rota ailesi", sorted({row["route_family"] for row in rows}))
    dispatch = c2.multiselect("Sevk/beklet", sorted({row["dispatch"] for row in rows}))
    fleet = c3.multiselect("Filo modu", sorted({row["fleet_mode"] for row in rows}))
    reorder = c4.multiselect("İkmal modu", sorted({row["reorder_mode"] for row in rows}))
    filtered = []
    for row in rows:
        if route and row["route_family"] not in route:
            continue
        if dispatch and row["dispatch"] not in dispatch:
            continue
        if fleet and row["fleet_mode"] not in fleet:
            continue
        if reorder and row["reorder_mode"] not in reorder:
            continue
        if search and search.lower() not in json.dumps(row, ensure_ascii=False).lower():
            continue
        filtered.append(row)
    st.caption(f"{len(filtered)} / 48 aksiyon listeleniyor. Aksiyon 24 ve Aksiyon 32 izlenen residual aksiyonlardır.")
    st.dataframe(_streamlit_df(filtered), use_container_width=True)


def _streamlit_replay(state: dict[str, Any]) -> None:
    replay = state["public_replay"]
    st.subheader("Kamu Replay Analizi")
    st.warning(PUBLIC_REPLAY_BOUNDARY)
    dataset = st.selectbox("Veri kümesi seçici", ["Tümü", "LaDe", "NYC HVFHS", "Olist"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LaDe", f"{replay['dataset_totals']['LaDe']:,}")
    c2.metric("NYC HVFHS", f"{replay['dataset_totals']['NYC HVFHS']:,}")
    c3.metric("Olist", f"{replay['dataset_totals']['Olist']:,}")
    c4.metric("Toplam", f"{replay['global_total']:,}")
    st.caption(f"Seçili görünüm: {dataset}")
    st.dataframe(_streamlit_df(replay["actions"]), use_container_width=True)
    st.markdown("**Top 10 aksiyon**")
    st.dataframe(_streamlit_df(replay["top_10_actions"]), use_container_width=True)
    st.markdown("**Aile dağılımı**")
    st.json(replay["family_distribution"])


def _streamlit_kpi(sim_state: dict[str, Any]) -> None:
    st.subheader("KPI ve Zaman Çizgisi")
    st.caption("Canlı modda KPI kartları env adımlarıyla güncellenir; bu sayfa offline eval veya gate çalıştırmaz.")
    rows = build_synthetic_kpi_trajectory(str(sim_state["scenario_id"]), steps=36)
    if sim_state.get("kpi_values"):
        rows = [dict(step=sim_state["step_index"], simulated_time=sim_state["simulated_time"], trajectory_label=sim_state["trajectory_label"], selected_action_id=sim_state["selected_action_id"], **sim_state["kpi_values"])] + rows
    st.dataframe(_streamlit_df(rows), use_container_width=True)
    if pd is not None:
        df = pd.DataFrame(rows).set_index("step")
        st.line_chart(df[["service_level", "dispatch_rate", "dispatch_success", "lateness", "route_failure"]])
    st.markdown("**Aksiyon dağılımı zaman özeti**")
    st.dataframe(_streamlit_df(summarize_action_distribution_over_time(rows)), use_container_width=True)


def _streamlit_evidence(state: dict[str, Any]) -> None:
    st.subheader("Kanıtlar ve Sınırlar")
    for item in state["evidence_limitations"]:
        st.write(f"- {item}")
    st.info("Yeni 3M/5M/10M koşusu, canlı şirket entegrasyonu veya özel şirket verisi içe aktarımı bu dashboard'un parçası değildir.")


def validate_dashboard_text() -> list[str]:
    corpus = dashboard_text_corpus().lower()
    return [phrase for phrase in FORBIDDEN_DASHBOARD_PHRASES if phrase in corpus]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Türkçe interaktif dijital ikiz dashboard'u.")
    parser.add_argument("--export-html", type=Path, help="Bağımsız Türkçe HTML yedeği yaz ve çık.")
    parser.add_argument("--root", type=Path, default=repo_root(), help="Repository root.")
    args = parser.parse_args(argv)
    state = build_dashboard_state(args.root)
    forbidden = validate_dashboard_text()
    if forbidden:
        raise SystemExit(f"Forbidden dashboard wording found: {forbidden}")
    if args.export_html:
        export_html_dashboard(state, args.export_html)
        print(
            json.dumps(
                {
                    "exported": str(args.export_html),
                    "classification": "LIVE_TURKISH_DIGITAL_TWIN_DASHBOARD_READY_WITH_SAFE_FALLBACK",
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    if st is None:
        fallback = args.root / "reports" / "demo_dashboard_v3" / "index.html"
        export_html_dashboard(state, fallback)
        print(json.dumps({"streamlit_available": False, "fallback": str(fallback)}, indent=2, ensure_ascii=False))
        return 0
    render_streamlit_dashboard(state)
    return 0


def streamlit_chrome_cleanup_css() -> str:
    return """
<style>
#MainMenu, footer, header [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
  visibility: hidden;
  display: none;
}
[data-testid="stDeployButton"] {
  visibility: hidden;
  display: none;
}
</style>
"""


if __name__ == "__main__":
    raise SystemExit(main())
