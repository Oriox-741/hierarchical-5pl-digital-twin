# Public Historical Replay Adapter Design - 2026-06-14

## Classification

`PUBLIC_HISTORICAL_REPLAY_ADAPTER_SPEC_READY`

## Ama?

Public historical logistics sat?rlar?n? CODEX production policy i?in g?venli bir proxy replay format?na d?n??t?rmek. Adapter, `physical_reality_v5_route_candidate_visibility` s?zle?mesindeki 73 boyutlu observation y?zeyini korur fakat public veride eksik olan simulator/company-state alanlar?n? a??k?a `unavailable` veya `imputed` olarak i?aretler.

## Input: ReplayInputRow

Zorunlu alanlar:

- `source_dataset`
- `source_row_id`

Opsiyonel proxy alanlar?:

- timestamp: `event_time`
- timing: `service_time_seconds`, `wait_time_seconds`
- route/location: `route_distance_km`, `pickup_location_id`, `dropoff_location_id`, `route_congestion_proxy`, `route_disruption_proxy`
- operational pressure: `demand_pressure`, `dispatch_pressure`, `fleet_pressure`, `lateness_risk`
- outcome/cost: `cost_proxy`, `service_success_proxy`
- logged behavior: `actual_logged_action` varsa sadece descriptive comparison i?in kullan?l?r.

## 73-Dim Fill Strategy

Her feature bir status ta??r:

- `observed`: public alandan do?rudan normalize edildi.
- `imputed`: public proxy'den t?retildi; ger?ek simulator state de?ildir.
- `neutral`: kontrat i?in sabit/n?tr de?er.
- `unavailable`: public sat?r bu feature'? desteklemiyor, de?er 0.0.

Confidence:

- high: missingness <= 0.45
- medium: missingness <= 0.75
- low: missingness > 0.75

## Action Interpretation

Model y?klenirse her sat?r i?in:

- continuous len 5
- discrete 0..47
- decoded dispatch/route/mode/reorder
- action 24/32 rates
- dispatch/hold rates
- route/fleet/reorder distributions

Action 24: `dispatch + shortest + secondary_fleet + none`.
Action 32: `dispatch + low_congestion + secondary_fleet + none`.

## G?venlik

- Production checkpoint sadece `--load-policy` verilirse read-only y?klenir.
- Default CLI no-policy dry-run ?retir.
- Protected output path'leri reddedilir.
- Existing output overwrite reddedilir.
- Adapter model/eval/registry/DB/checkpoint yazmaz.

## Causal Limitation

Bu adapter public veride OPE yapmaz. Propensity, unchosen alternatives, reward/cost trajectory ve behavior policy yoksa sonu? yaln?zca descriptive replay/proxy evidence olarak yorumlan?r.

Makine-okunur spec:

`reports/benchmarks/historical_replay_20260614/public_replay_adapter/public_replay_adapter_spec.json`
