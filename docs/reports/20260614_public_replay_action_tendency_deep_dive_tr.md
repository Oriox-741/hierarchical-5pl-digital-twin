# Public Replay Action Tendency Deep Dive - 2026-06-14

## Classification

`PUBLIC_REPLAY_ACTION_TENDENCY_DEEP_DIVE_READY`

## Kapsam

Bu analiz final large public replay raporlar?n? a??klar. Large replay JSON'lar?nda per-row veri tutulmad??? i?in segment oranlar? ayn? public veri e?lemeleri ve production checkpoint'in read-only tahmini ile yeniden aggregate olarak hesapland?. Per-row dump yaz?lmad?.

## Overall Action Tendencies

| Dataset | Rows | Dispatch | Hold | Action 24 | Action 32 | Top actions | Confidence | Missingness |
|---|---:|---:|---:|---:|---:|---|---|---:|
| LaDe | 31415 | 0.004106 | 0.995894 | 0.000000 | 0.000000 | [('1', 24687), ('0', 6589), ('41', 82), ('45', 47), ('2', 10)] | {'high': 31415} | 0.305118 |
| NYC_HVFHS | 100000 | 0.040920 | 0.959080 | 0.000510 | 0.000020 | [('1', 79220), ('0', 13929), ('2', 2754), ('41', 2510), ('25', 1007)] | {'high': 100000} | 0.301370 |
| Olist | 96476 | 0.036434 | 0.963566 | 0.002063 | 0.000000 | [('1', 52201), ('0', 38699), ('43', 1661), ('3', 1190), ('2', 871)] | {'medium': 96476} | 0.547951 |

## Ana Yorum

D???k dispatch, action 24 ve action 32 oran? ?ncelikle beklenen bir public replay sonucudur. Public sat?rlar tam simulator decision-time state de?ildir; current unassigned order, feasible dispatch alternative, secondary fleet availability, inventory state, route alternatives ve reward context eksiktir veya proxy/imputed olarak ta??n?r.

- LaDe ve NYC adapter confidence a??s?ndan high g?r?n?r, ancak yine de public proxy state'tir.
- Olist medium confidence ve y?ksek missingness ta??r; route/fleet alanlar? do?al olarak eksiktir.
- Dispatch/action 24/32 d???kl??? actual env rollout ba?ar?s?zl??? de?ildir. Actual env performans? internal scenario/gate/equal-budget eval ile ?l??l?r.
- Bu sonu? ?irket verisiyle replay/OPE ihtiyac?n? g??lendirir; causal superiority veya company-data validation iddias? ?retmez.

### LaDe Segmentleri
#### confidence
| Segment | Rows | Dispatch | Hold | Action 24 | Action 32 | Top actions |
|---|---:|---:|---:|---:|---:|---|
| high | 31415 | 0.004106 | 0.995894 | 0.000000 | 0.000000 | [('1', 24687), ('0', 6589), ('41', 82), ('45', 47), ('2', 10)] |
#### missingness
| Segment | Rows | Dispatch | Hold | Action 24 | Action 32 | Top actions |
|---|---:|---:|---:|---:|---:|---|
| low | 31415 | 0.004106 | 0.995894 | 0.000000 | 0.000000 | [('1', 24687), ('0', 6589), ('41', 82), ('45', 47), ('2', 10)] |
#### dispatch_pressure
| Segment | Rows | Dispatch | Hold | Action 24 | Action 32 | Top actions |
|---|---:|---:|---:|---:|---:|---|
| high | 1333 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | [('1', 1245), ('0', 88)] |
| low | 30082 | 0.004288 | 0.995712 | 0.000000 | 0.000000 | [('1', 23442), ('0', 6501), ('41', 82), ('45', 47), ('2', 10)] |
#### demand_pressure
| Segment | Rows | Dispatch | Hold | Action 24 | Action 32 | Top actions |
|---|---:|---:|---:|---:|---:|---|
| high | 1333 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | [('1', 1245), ('0', 88)] |
| low | 30082 | 0.004288 | 0.995712 | 0.000000 | 0.000000 | [('1', 23442), ('0', 6501), ('41', 82), ('45', 47), ('2', 10)] |
#### fleet_pressure
| Segment | Rows | Dispatch | Hold | Action 24 | Action 32 | Top actions |
|---|---:|---:|---:|---:|---:|---|
| high | 11984 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | [('1', 8564), ('0', 3420)] |
| low | 19431 | 0.006639 | 0.993361 | 0.000000 | 0.000000 | [('1', 16123), ('0', 3169), ('41', 82), ('45', 47), ('2', 10)] |
#### lateness_risk
| Segment | Rows | Dispatch | Hold | Action 24 | Action 32 | Top actions |
|---|---:|---:|---:|---:|---:|---|
| high | 10702 | 0.012054 | 0.987946 | 0.000000 | 0.000000 | [('1', 10567), ('41', 82), ('45', 47), ('0', 4), ('2', 2)] |
| low | 20713 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | [('1', 14120), ('0', 6585), ('2', 8)] |
#### route_congestion_proxy
| Segment | Rows | Dispatch | Hold | Action 24 | Action 32 | Top actions |
|---|---:|---:|---:|---:|---:|---|
| high | 10702 | 0.012054 | 0.987946 | 0.000000 | 0.000000 | [('1', 10567), ('41', 82), ('45', 47), ('0', 4), ('2', 2)] |
| low | 20713 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | [('1', 14120), ('0', 6585), ('2', 8)] |
#### route_disruption_proxy
| Segment | Rows | Dispatch | Hold | Action 24 | Action 32 | Top actions |
|---|---:|---:|---:|---:|---:|---|
| high | 156 | 0.294872 | 0.705128 | 0.000000 | 0.000000 | [('1', 109), ('41', 31), ('45', 15), ('0', 1)] |
| low | 31259 | 0.002655 | 0.997345 | 0.000000 | 0.000000 | [('1', 24578), ('0', 6588), ('41', 51), ('45', 32), ('2', 10)] |

### NYC_HVFHS Segmentleri
#### confidence
| Segment | Rows | Dispatch | Hold | Action 24 | Action 32 | Top actions |
|---|---:|---:|---:|---:|---:|---|
| high | 100000 | 0.040920 | 0.959080 | 0.000510 | 0.000020 | [('1', 79220), ('0', 13929), ('2', 2754), ('41', 2510), ('25', 1007)] |
#### missingness
| Segment | Rows | Dispatch | Hold | Action 24 | Action 32 | Top actions |
|---|---:|---:|---:|---:|---:|---|
| low | 100000 | 0.040920 | 0.959080 | 0.000510 | 0.000020 | [('1', 79220), ('0', 13929), ('2', 2754), ('41', 2510), ('25', 1007)] |
#### dispatch_pressure
| Segment | Rows | Dispatch | Hold | Action 24 | Action 32 | Top actions |
|---|---:|---:|---:|---:|---:|---|
| high | 35855 | 0.113959 | 0.886041 | 0.001255 | 0.000056 | [('1', 30886), ('41', 2510), ('25', 1007), ('2', 617), ('29', 281)] |
| low | 64145 | 0.000094 | 0.999906 | 0.000094 | 0.000000 | [('1', 48334), ('0', 13668), ('2', 2137), ('24', 6)] |
#### demand_pressure
| Segment | Rows | Dispatch | Hold | Action 24 | Action 32 | Top actions |
|---|---:|---:|---:|---:|---:|---|
| high | 16831 | 0.003803 | 0.996197 | 0.000000 | 0.000000 | [('1', 14995), ('0', 1006), ('2', 766), ('29', 61), ('25', 3)] |
| low | 83169 | 0.048432 | 0.951568 | 0.000613 | 0.000024 | [('1', 64225), ('0', 12923), ('41', 2510), ('2', 1988), ('25', 1004)] |
#### fleet_pressure
| Segment | Rows | Dispatch | Hold | Action 24 | Action 32 | Top actions |
|---|---:|---:|---:|---:|---:|---|
| high | 71259 | 0.020447 | 0.979553 | 0.000716 | 0.000000 | [('1', 56783), ('0', 11009), ('2', 2009), ('25', 930), ('29', 277)] |
| low | 28741 | 0.091681 | 0.908319 | 0.000000 | 0.000070 | [('1', 22437), ('0', 2920), ('41', 2441), ('2', 745), ('25', 77)] |
#### lateness_risk
| Segment | Rows | Dispatch | Hold | Action 24 | Action 32 | Top actions |
|---|---:|---:|---:|---:|---:|---|
| high | 35855 | 0.113959 | 0.886041 | 0.001255 | 0.000056 | [('1', 30886), ('41', 2510), ('25', 1007), ('2', 617), ('29', 281)] |
| low | 64145 | 0.000094 | 0.999906 | 0.000094 | 0.000000 | [('1', 48334), ('0', 13668), ('2', 2137), ('24', 6)] |
#### route_congestion_proxy
| Segment | Rows | Dispatch | Hold | Action 24 | Action 32 | Top actions |
|---|---:|---:|---:|---:|---:|---|
| high | 64593 | 0.029291 | 0.970709 | 0.000279 | 0.000000 | [('1', 50667), ('0', 10887), ('41', 1394), ('2', 1143), ('25', 241)] |
| low | 35407 | 0.062135 | 0.937865 | 0.000932 | 0.000056 | [('1', 28553), ('0', 3042), ('2', 1611), ('41', 1116), ('25', 766)] |
#### route_disruption_proxy
| Segment | Rows | Dispatch | Hold | Action 24 | Action 32 | Top actions |
|---|---:|---:|---:|---:|---:|---|
| high | 35855 | 0.113959 | 0.886041 | 0.001255 | 0.000056 | [('1', 30886), ('41', 2510), ('25', 1007), ('2', 617), ('29', 281)] |
| low | 64145 | 0.000094 | 0.999906 | 0.000094 | 0.000000 | [('1', 48334), ('0', 13668), ('2', 2137), ('24', 6)] |

### Olist Segmentleri
#### confidence
| Segment | Rows | Dispatch | Hold | Action 24 | Action 32 | Top actions |
|---|---:|---:|---:|---:|---:|---|
| medium | 96476 | 0.036434 | 0.963566 | 0.002063 | 0.000000 | [('1', 52201), ('0', 38699), ('43', 1661), ('3', 1190), ('2', 871)] |
#### missingness
| Segment | Rows | Dispatch | Hold | Action 24 | Action 32 | Top actions |
|---|---:|---:|---:|---:|---:|---|
| high | 96476 | 0.036434 | 0.963566 | 0.002063 | 0.000000 | [('1', 52201), ('0', 38699), ('43', 1661), ('3', 1190), ('2', 871)] |
#### dispatch_pressure
| Segment | Rows | Dispatch | Hold | Action 24 | Action 32 | Top actions |
|---|---:|---:|---:|---:|---:|---|
| high | 26011 | 0.046057 | 0.953943 | 0.000000 | 0.000000 | [('1', 23992), ('43', 1017), ('3', 555), ('2', 266), ('27', 149)] |
| low | 70465 | 0.032882 | 0.967118 | 0.002824 | 0.000000 | [('0', 38699), ('1', 28209), ('43', 644), ('3', 635), ('2', 605)] |
#### demand_pressure
| Segment | Rows | Dispatch | Hold | Action 24 | Action 32 | Top actions |
|---|---:|---:|---:|---:|---:|---|
| high | 1147 | 0.154316 | 0.845684 | 0.010462 | 0.000000 | [('1', 945), ('28', 103), ('47', 37), ('3', 16), ('29', 15)] |
| low | 95329 | 0.035016 | 0.964984 | 0.001962 | 0.000000 | [('1', 51256), ('0', 38699), ('43', 1661), ('3', 1174), ('2', 862)] |
#### fleet_pressure
| Segment | Rows | Dispatch | Hold | Action 24 | Action 32 | Top actions |
|---|---:|---:|---:|---:|---:|---|
| missing | 96476 | 0.036434 | 0.963566 | 0.002063 | 0.000000 | [('1', 52201), ('0', 38699), ('43', 1661), ('3', 1190), ('2', 871)] |
#### lateness_risk
| Segment | Rows | Dispatch | Hold | Action 24 | Action 32 | Top actions |
|---|---:|---:|---:|---:|---:|---|
| high | 5806 | 0.511540 | 0.488460 | 0.033586 | 0.000000 | [('43', 1660), ('1', 1633), ('3', 1155), ('27', 638), ('24', 195)] |
| low | 90670 | 0.006011 | 0.993989 | 0.000044 | 0.000000 | [('1', 50568), ('0', 38699), ('2', 823), ('45', 420), ('41', 89)] |
#### route_congestion_proxy
| Segment | Rows | Dispatch | Hold | Action 24 | Action 32 | Top actions |
|---|---:|---:|---:|---:|---:|---|
| missing | 96476 | 0.036434 | 0.963566 | 0.002063 | 0.000000 | [('1', 52201), ('0', 38699), ('43', 1661), ('3', 1190), ('2', 871)] |
#### route_disruption_proxy
| Segment | Rows | Dispatch | Hold | Action 24 | Action 32 | Top actions |
|---|---:|---:|---:|---:|---:|---|
| high | 5806 | 0.511540 | 0.488460 | 0.033586 | 0.000000 | [('43', 1660), ('1', 1633), ('3', 1155), ('27', 638), ('24', 195)] |
| low | 90670 | 0.006011 | 0.993989 | 0.000044 | 0.000000 | [('1', 50568), ('0', 38699), ('2', 823), ('45', 420), ('41', 89)] |

## Public Replay vs Env Rollout vs Company Replay

| Evidence Type | Ne s?yler | Ne s?ylemez |
|---|---|---|
| Public replay descriptive action tendency | Public proxy state ?zerinde modelin hangi aksiyonlara e?ildi?ini g?sterir | Counterfactual ?st?nl?k veya ger?ek operasyon ba?ar?s? g?stermez |
| Env rollout / gate | Dijital ikiz senaryolar?nda actual policy performans?n? ?l?er | ?irket operasyonu ve live telemetry kan?t? de?ildir |
| Company historical replay / OPE | Ger?ek dispatch/fleet/reorder/cost davran???n? ve m?mk?nse OPE'yi test eder | Bu sprintte ?irket verisi olmad??? i?in yap?lmad? |

Makine-okunur rapor: `reports/benchmarks/final_evidence_20260614/replay_large_samples/public_replay_action_tendency_deep_dive.json`
