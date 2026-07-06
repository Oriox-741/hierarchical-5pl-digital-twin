"""Public historical replay adapter for CODEX-like proxy states.

The adapter maps public logistics rows into the fixed 73-dimensional CODEX
observation contract with explicit feature provenance. It is a replay
feasibility layer, not a causal counterfactual evaluator: public datasets that
lack logged behavior-policy propensities, alternatives, and reward trajectories
can support action-tendency/proxy analysis only.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np

from src.act.discrete_action_mapper import DISCRETE_ACTION_COUNT, DiscreteActionMapper
from src.act.observation_builder import OBSERVATION_DIM, OBSERVATION_FEATURES
from src.orchestration.torch_joint_runtime import load_torch_joint_policy


ADAPTER_VERSION = "public_historical_replay_adapter_v1"
DEFAULT_PRODUCTION_CHECKPOINT = (
    Path("models")
    / "production"
    / "joint_torch_v5_prod_hierarchical_v1_1m_20260611"
    / "joint_torch_latest.pt"
)
PROTECTED_OUTPUT_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("models", "registry"),
    ("models", "production"),
    ("models", "baselines"),
    ("models", "checkpoints"),
    ("models", "eval"),
    ("db",),
)
FEATURE_STATUS_VALUES = frozenset({"observed", "imputed", "neutral", "unavailable"})
WATCHED_ACTIONS = (24, 32)


@dataclass(frozen=True, slots=True)
class ReplayInputRow:
    source_dataset: str
    source_row_id: str
    event_time: str | None = None
    service_time_seconds: float | None = None
    wait_time_seconds: float | None = None
    route_distance_km: float | None = None
    pickup_location_id: str | None = None
    dropoff_location_id: str | None = None
    demand_pressure: float | None = None
    dispatch_pressure: float | None = None
    route_congestion_proxy: float | None = None
    route_disruption_proxy: float | None = None
    fleet_pressure: float | None = None
    lateness_risk: float | None = None
    cost_proxy: float | None = None
    service_success_proxy: float | None = None
    actual_logged_action: str | int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "ReplayInputRow":
        return cls(
            source_dataset=str(row.get("source_dataset", "public")),
            source_row_id=str(row.get("source_row_id", row.get("id", ""))),
            event_time=_optional_text(row.get("event_time")),
            service_time_seconds=_finite_float(row.get("service_time_seconds")),
            wait_time_seconds=_finite_float(row.get("wait_time_seconds")),
            route_distance_km=_finite_float(row.get("route_distance_km")),
            pickup_location_id=_optional_text(row.get("pickup_location_id")),
            dropoff_location_id=_optional_text(row.get("dropoff_location_id")),
            demand_pressure=_finite_float(row.get("demand_pressure")),
            dispatch_pressure=_finite_float(row.get("dispatch_pressure")),
            route_congestion_proxy=_finite_float(row.get("route_congestion_proxy")),
            route_disruption_proxy=_finite_float(row.get("route_disruption_proxy")),
            fleet_pressure=_finite_float(row.get("fleet_pressure")),
            lateness_risk=_finite_float(row.get("lateness_risk")),
            cost_proxy=_finite_float(row.get("cost_proxy")),
            service_success_proxy=_finite_float(row.get("service_success_proxy")),
            actual_logged_action=row.get("actual_logged_action"),
            extra=row.get("extra") if isinstance(row.get("extra"), Mapping) else {},
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PublicObservation:
    vector: np.ndarray
    status_by_feature: dict[str, str]
    missingness_mask: list[int]
    missingness_rate: float
    confidence: str
    provenance: dict[str, str]


def build_public_observation(row: ReplayInputRow | Mapping[str, Any]) -> PublicObservation:
    normalized = row if isinstance(row, ReplayInputRow) else ReplayInputRow.from_mapping(row)
    vector = np.zeros(OBSERVATION_DIM, dtype=np.float32)
    status_by_feature = {feature: "unavailable" for feature in OBSERVATION_FEATURES}
    provenance: dict[str, str] = {}
    index_by_feature = {feature: index for index, feature in enumerate(OBSERVATION_FEATURES)}

    def set_feature(name: str, value: float, status: str, source: str) -> None:
        if status not in FEATURE_STATUS_VALUES:
            raise ValueError(f"unsupported feature status: {status}")
        if name not in index_by_feature:
            return
        vector[index_by_feature[name]] = _clamp01(value)
        status_by_feature[name] = status
        provenance[name] = source

    set_feature("bias", 1.0, "neutral", "constant_bias")

    time_ratio = _event_time_ratio(normalized.event_time)
    if time_ratio is not None:
        set_feature("time_ratio", time_ratio, "observed", "event_time")

    demand_pressure = _clamp01_or_none(normalized.demand_pressure)
    dispatch_pressure = _clamp01_or_none(normalized.dispatch_pressure)
    fleet_pressure = _clamp01_or_none(normalized.fleet_pressure)
    lateness_risk = _clamp01_or_none(normalized.lateness_risk)
    congestion = _clamp01_or_none(normalized.route_congestion_proxy)
    disruption = _clamp01_or_none(normalized.route_disruption_proxy)
    success = _clamp01_or_none(normalized.service_success_proxy)
    service_time_pressure = _normalize_positive(normalized.service_time_seconds, scale=86_400.0)
    wait_pressure = _normalize_positive(normalized.wait_time_seconds, scale=3_600.0)
    distance_pressure = _normalize_positive(normalized.route_distance_km, scale=100.0)
    cost_pressure = _normalize_positive(normalized.cost_proxy, scale=250.0)
    zone_change = 1.0 if _has_text(normalized.pickup_location_id) and _has_text(normalized.dropoff_location_id) else None

    if demand_pressure is not None:
        set_feature("pending_order_ratio", demand_pressure, "observed", "demand_pressure")
        set_feature("demand_mean_pressure", demand_pressure, "observed", "demand_pressure")
        set_feature("normalized_dispatchable_order_count", demand_pressure, "observed", "demand_pressure")
        set_feature("urgent_order_ratio", max(demand_pressure, wait_pressure or 0.0), "imputed", "demand_pressure/wait")
    if success is not None:
        set_feature("service_level", success, "observed", "service_success_proxy")
        set_feature("delivered_order_ratio", success, "imputed", "service_success_proxy")
    elif lateness_risk is not None:
        set_feature("service_level", 1.0 - lateness_risk, "imputed", "lateness_risk")
    if lateness_risk is not None:
        set_feature("late_delivery_ratio", lateness_risk, "observed", "lateness_risk")
        set_feature("backlog_age_pressure", lateness_risk, "imputed", "lateness_risk")
        set_feature("premium_sla_pressure", lateness_risk, "imputed", "lateness_risk")
    if service_time_pressure is not None:
        set_feature("lead_time_mean_pressure", service_time_pressure, "observed", "service_time_seconds")
        set_feature("scenario_lead_time_volatility_pressure", service_time_pressure, "imputed", "service_time_seconds")
        set_feature("supplier_delay_pressure", service_time_pressure, "imputed", "service_time_seconds")
    if wait_pressure is not None:
        set_feature("lead_time_variance_pressure", wait_pressure, "observed", "wait_time_seconds")
        set_feature("feasible_dispatch_ratio", 1.0 - wait_pressure, "imputed", "wait_time_seconds")
        set_feature("vehicle_availability_pressure", wait_pressure, "imputed", "wait_time_seconds")
    if dispatch_pressure is not None:
        set_feature("feasible_dispatch_opportunity", 1.0 if dispatch_pressure > 0.0 else 0.0, "observed", "dispatch_pressure")
        set_feature("useful_dispatch_opportunity", dispatch_pressure, "imputed", "dispatch_pressure")
        set_feature("normalized_available_vehicle_count", 1.0 - dispatch_pressure, "imputed", "dispatch_pressure")
        set_feature("capacity_pressure", dispatch_pressure, "observed", "dispatch_pressure")
    if fleet_pressure is not None:
        set_feature("capacity_slack", 1.0 - fleet_pressure, "observed", "fleet_pressure")
        set_feature("capacity_shock_pressure", fleet_pressure, "observed", "fleet_pressure")
        set_feature("active_vehicle_ratio", 1.0 - fleet_pressure, "imputed", "fleet_pressure")
        set_feature("secondary_fleet_feasible_dispatch_ratio", 1.0 - fleet_pressure, "imputed", "fleet_pressure")
        set_feature("secondary_fleet_vehicle_availability_pressure", fleet_pressure, "observed", "fleet_pressure")
        set_feature("premium_fleet_exposure", fleet_pressure, "imputed", "fleet_pressure")
    if congestion is not None:
        set_feature("db_congestion_score", congestion, "observed", "route_congestion_proxy")
        set_feature("low_congestion_candidate_score", 1.0 - congestion, "imputed", "route_congestion_proxy")
        set_feature("low_congestion_candidate_score_gap", congestion, "imputed", "route_congestion_proxy")
        set_feature("route_pressure_congestion_share", congestion, "observed", "route_congestion_proxy")
        set_feature("route_pressure_balance", 1.0 - abs(congestion - (disruption or congestion)), "imputed", "route_congestion_proxy")
    if disruption is not None:
        set_feature("disruption_score", disruption, "observed", "route_disruption_proxy")
        set_feature("route_disruption_pressure", disruption, "observed", "route_disruption_proxy")
        set_feature("scenario_route_disruption_pressure", disruption, "observed", "route_disruption_proxy")
        set_feature("high_resilience_candidate_score", disruption, "imputed", "route_disruption_proxy")
        set_feature("high_resilience_candidate_score_gap", 1.0 - disruption, "imputed", "route_disruption_proxy")
        set_feature("route_pressure_reliability_share", disruption, "observed", "route_disruption_proxy")
    if distance_pressure is not None:
        set_feature("route_cost_pressure", distance_pressure, "observed", "route_distance_km")
        set_feature("shortest_candidate_score", 1.0 - distance_pressure, "imputed", "route_distance_km")
        set_feature("shortest_candidate_score_gap", distance_pressure, "imputed", "route_distance_km")
        set_feature("shortest_candidate_near_best", 1.0 if distance_pressure <= 0.35 else 0.0, "imputed", "route_distance_km")
    if cost_pressure is not None:
        set_feature("transport_cost_ratio", cost_pressure, "observed", "cost_proxy")
        set_feature("speed_cost_exposure", cost_pressure, "imputed", "cost_proxy")
    if zone_change is not None:
        set_feature("hub_count_ratio", _stable_hash_ratio(normalized.pickup_location_id), "observed", "pickup_location_id")
        set_feature("customer_count_ratio", _stable_hash_ratio(normalized.dropoff_location_id), "observed", "dropoff_location_id")
        set_feature("route_arc_count_ratio", zone_change, "imputed", "pickup/dropoff_location_id")
        set_feature("network_safety_potential", 1.0 - max(disruption or 0.0, congestion or 0.0), "imputed", "route_proxy_fields")

    if dispatch_pressure is not None and fleet_pressure is not None and distance_pressure is not None:
        safe_context = (1.0 - fleet_pressure) * (1.0 - max(congestion or 0.0, disruption or 0.0))
        set_feature("shortest_secondary_safe_context", safe_context, "imputed", "public_proxy_composite")
        set_feature("shortest_secondary_brittle_risk", 1.0 - safe_context, "imputed", "public_proxy_composite")
        set_feature("route_alt_pressure_imbalance", abs((congestion or 0.0) - (disruption or 0.0)), "imputed", "public_proxy_composite")

    unavailable_count = sum(1 for status in status_by_feature.values() if status == "unavailable")
    missingness_rate = unavailable_count / OBSERVATION_DIM
    if missingness_rate <= 0.45:
        confidence = "high"
    elif missingness_rate <= 0.75:
        confidence = "medium"
    else:
        confidence = "low"
    missingness_mask = [1 if status_by_feature[feature] == "unavailable" else 0 for feature in OBSERVATION_FEATURES]
    return PublicObservation(
        vector=vector,
        status_by_feature=status_by_feature,
        missingness_mask=missingness_mask,
        missingness_rate=missingness_rate,
        confidence=confidence,
        provenance=provenance,
    )


def predict_public_row(policy: Any, row: ReplayInputRow | Mapping[str, Any]) -> dict[str, Any]:
    observation = build_public_observation(row)
    result = policy.predict_joint(observation.vector, deterministic=True)
    continuous = np.asarray(getattr(result, "continuous"), dtype=np.float32).reshape(-1)
    if continuous.shape != (5,):
        raise ValueError(f"continuous prediction must have shape (5,), got {continuous.shape}.")
    if not np.all(np.isfinite(continuous)):
        raise ValueError("continuous prediction contains non-finite values.")
    if np.any(continuous < -1.0) or np.any(continuous > 1.0):
        raise ValueError("continuous prediction must be bounded within [-1, 1].")
    discrete = int(getattr(result, "discrete"))
    if not 0 <= discrete < DISCRETE_ACTION_COUNT:
        raise ValueError(f"discrete prediction must be in [0, {DISCRETE_ACTION_COUNT - 1}], got {discrete}.")
    return {
        "continuous": [float(value) for value in continuous.tolist()],
        "discrete": discrete,
        "decoded_action": decode_discrete_action(discrete),
    }


def decode_discrete_action(action_id: int) -> dict[str, str]:
    return DiscreteActionMapper().map(int(action_id)).as_dict()


def build_replay_report(
    rows: Sequence[ReplayInputRow | Mapping[str, Any]],
    *,
    policy: Any | None,
    source_name: str,
    include_rows: bool = True,
) -> dict[str, Any]:
    normalized_rows = [row if isinstance(row, ReplayInputRow) else ReplayInputRow.from_mapping(row) for row in rows]
    record_rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    missingness_values: list[float] = []
    confidence_counts: Counter[str] = Counter()
    status_totals: Counter[str] = Counter()
    service_proxy_values: list[float] = []
    action24_indicator: list[float] = []
    action32_indicator: list[float] = []
    dispatch_indicator: list[float] = []

    for row in normalized_rows:
        observation = build_public_observation(row)
        missingness_values.append(observation.missingness_rate)
        confidence_counts[observation.confidence] += 1
        status_totals.update(observation.status_by_feature.values())
        service_proxy = _service_proxy(row)
        if service_proxy is not None:
            service_proxy_values.append(service_proxy)

        prediction = predict_public_row(policy, row) if policy is not None else None
        if prediction is not None:
            predictions.append(prediction)
            decoded = prediction["decoded_action"]
            action = int(prediction["discrete"])
            action24_indicator.append(1.0 if action == 24 else 0.0)
            action32_indicator.append(1.0 if action == 32 else 0.0)
            dispatch_indicator.append(1.0 if decoded["dispatch"] == "dispatch" else 0.0)

        if include_rows:
            record_rows.append(
                {
                    "source_dataset": row.source_dataset,
                    "source_row_id": row.source_row_id,
                    "confidence": observation.confidence,
                    "missingness_rate": observation.missingness_rate,
                    "feature_status_counts": dict(Counter(observation.status_by_feature.values())),
                    "service_proxy": service_proxy,
                    "actual_logged_action": row.actual_logged_action,
                    "prediction": prediction,
                }
            )

    action_counts = Counter(str(prediction["discrete"]) for prediction in predictions)
    route_counts = Counter(prediction["decoded_action"]["route"] for prediction in predictions)
    mode_counts = Counter(prediction["decoded_action"]["mode"] for prediction in predictions)
    reorder_counts = Counter(prediction["decoded_action"]["reorder"] for prediction in predictions)
    dispatch_counts = Counter(prediction["decoded_action"]["dispatch"] for prediction in predictions)
    prediction_count = len(predictions)
    return {
        "report_metadata": {
            "adapter_version": ADAPTER_VERSION,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "source_name": source_name,
            "policy_loaded": policy is not None,
            "row_count": len(normalized_rows),
            "causal_counterfactual_claim": False,
            "public_proxy_replay_only": True,
        },
        "summary": {
            "row_count": len(normalized_rows),
            "prediction_count": prediction_count,
            "mean_missingness_rate": mean(missingness_values) if missingness_values else None,
            "confidence_distribution": dict(confidence_counts),
            "aggregate_feature_status_counts": dict(status_totals),
            "action_distribution": dict(action_counts),
            "action_24_rate": action_counts.get("24", 0) / prediction_count if prediction_count else None,
            "action_32_rate": action_counts.get("32", 0) / prediction_count if prediction_count else None,
            "dispatch_rate": dispatch_counts.get("dispatch", 0) / prediction_count if prediction_count else None,
            "hold_rate": dispatch_counts.get("hold", 0) / prediction_count if prediction_count else None,
            "route_distribution": dict(route_counts),
            "mode_distribution": dict(mode_counts),
            "reorder_distribution": dict(reorder_counts),
            "service_proxy": _numeric_summary(service_proxy_values),
            "service_proxy_correlation": {
                "action_24": _pearson(service_proxy_values, action24_indicator),
                "action_32": _pearson(service_proxy_values, action32_indicator),
                "dispatch": _pearson(service_proxy_values, dispatch_indicator),
            }
            if len(service_proxy_values) == prediction_count
            else {"action_24": None, "action_32": None, "dispatch": None},
        },
        "interpretation_limits": {
            "logged_public_actions_available": any(row.actual_logged_action is not None for row in normalized_rows),
            "logged_propensities_available": False,
            "counterfactual_reward_available": False,
            "safe_claim": "Action tendencies and proxy correlations only; no causal policy-value or optimality claim.",
        },
        "rows": record_rows if include_rows else [],
    }


def load_jsonl_rows(path: Path, *, limit: int | None = None) -> list[ReplayInputRow]:
    rows: list[ReplayInputRow] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, Mapping):
                raise ValueError("each JSONL replay row must be an object")
            rows.append(ReplayInputRow.from_mapping(payload))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_report_fresh(path: Path | str, report: Mapping[str, Any]) -> None:
    output = Path(path)
    _reject_protected_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing replay report: {output}")
    output.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    rows = load_jsonl_rows(args.input_jsonl, limit=args.limit)
    policy = None
    if args.load_policy:
        policy = load_torch_joint_policy(args.checkpoint, device=args.device)
    report = build_replay_report(rows, policy=policy, source_name=args.source_name, include_rows=not args.no_rows)
    write_report_fresh(args.output, report)
    print("PUBLIC_HISTORICAL_REPLAY_REPORT_READY")
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a public historical replay proxy report.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-name", default="public_replay")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--load-policy", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_PRODUCTION_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-rows", action="store_true", help="Omit per-row records from the output report.")
    return parser


def _service_proxy(row: ReplayInputRow) -> float | None:
    if row.service_success_proxy is not None:
        return _clamp01(row.service_success_proxy)
    if row.lateness_risk is not None:
        return 1.0 - _clamp01(row.lateness_risk)
    if row.wait_time_seconds is not None:
        return 1.0 - _clamp01(max(float(row.wait_time_seconds), 0.0) / 3_600.0)
    if row.service_time_seconds is not None:
        return 1.0 - _clamp01(max(float(row.service_time_seconds), 0.0) / 86_400.0)
    return None


def _numeric_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {"count": len(finite), "mean": mean(finite), "min": min(finite), "max": max(finite)}


def _pearson(x_values: Sequence[float], y_values: Sequence[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _event_time_ratio(value: str | None) -> float | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    for candidate in (text, text.replace("Z", "+00:00"), text.replace("/", "-")):
        try:
            parsed = datetime.fromisoformat(candidate)
            return ((parsed.hour * 3600.0) + (parsed.minute * 60.0) + parsed.second) / 86_400.0
        except ValueError:
            pass
    return None


def _normalize_positive(value: Any, *, scale: float) -> float | None:
    number = _finite_float(value)
    if number is None:
        return None
    return _clamp01(max(0.0, number) / max(scale, 1e-9))


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _has_text(value: Any) -> bool:
    return _optional_text(value) is not None


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _clamp01_or_none(value: Any) -> float | None:
    number = _finite_float(value)
    if number is None:
        return None
    return _clamp01(number)


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _stable_hash_ratio(value: str | None) -> float:
    if value is None:
        return 0.0
    total = sum((index + 1) * ord(char) for index, char in enumerate(str(value)))
    return (total % 10_000) / 10_000.0


def _reject_protected_path(path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    candidate = Path(path).resolve(strict=False)
    for prefix in PROTECTED_OUTPUT_PREFIXES:
        protected = repo_root.joinpath(*prefix).resolve(strict=False)
        if candidate == protected or protected in candidate.parents:
            raise ValueError(f"refusing to write under protected path: {path}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    return value


if __name__ == "__main__":
    raise SystemExit(main())
