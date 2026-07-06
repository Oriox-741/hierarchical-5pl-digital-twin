"""Write the final CODEX-5PL evidence integration package.

This script consumes already-generated public replay reports and existing
project docs, then writes only fresh final-evidence reports/docs plus a dashboard
link update. It never touches protected model artifacts.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports" / "benchmarks" / "final_evidence_20260614"


def main() -> int:
    large = {
        "LaDe": _load_json("reports/benchmarks/final_evidence_20260614/replay_large_samples/lade_large_replay_report.json"),
        "NYC_HVFHS": _load_json(
            "reports/benchmarks/final_evidence_20260614/replay_large_samples/nyc_hvfhs_large_replay_report.json"
        ),
        "Olist": _load_json(
            "reports/benchmarks/final_evidence_20260614/replay_large_samples/olist_large_replay_report.json"
        ),
    }
    bounded = {
        "LaDe": _load_json("reports/benchmarks/historical_replay_20260614/lade_replay_probe/lade_replay_probe_report.json"),
        "NYC_HVFHS": _load_json(
            "reports/benchmarks/historical_replay_20260614/nyc_hvfhs_replay_probe/nyc_hvfhs_replay_probe_report.json"
        ),
        "Olist": _load_json(
            "reports/benchmarks/historical_replay_20260614/olist_replay_probe/olist_replay_probe_report.json"
        ),
    }
    large_compact = {name: _compact(report) for name, report in large.items()}
    bounded_compact = {name: _compact(report) for name, report in bounded.items()}

    _write_replay_comparison(large_compact, bounded_compact)
    truth_index = _write_truth_index()
    scorecard = _write_scorecard()
    updated = _write_stale_doc_cleanup()
    _write_advisor_package(large_compact)
    _write_thesis_package(large_compact)
    _write_company_data_package()
    _write_blocker_register()
    _write_reproducibility_manifest(truth_index)
    _update_dashboard()

    print(
        json.dumps(
            {
                "classification": "FINAL_EVIDENCE_CORE_DOCS_WRITTEN",
                "large_replay_rows": {name: item["row_count"] for name, item in large_compact.items()},
                "score_count": len(scorecard["scores"]),
                "stale_docs_updated": updated,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _load_json(rel: str) -> dict[str, Any]:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def _write_fresh(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json_fresh(path: Path, data: Any) -> None:
    _write_fresh(path, json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _top_actions(report: dict[str, Any], n: int = 8) -> list[tuple[str, int]]:
    distribution = report["summary"].get("action_distribution") or {}
    return sorted(distribution.items(), key=lambda item: int(item[1]), reverse=True)[:n]


def _compact(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    return {
        "row_count": summary["row_count"],
        "prediction_count": summary["prediction_count"],
        "mean_missingness_rate": summary["mean_missingness_rate"],
        "confidence_distribution": summary["confidence_distribution"],
        "action_24_rate": summary["action_24_rate"],
        "action_32_rate": summary["action_32_rate"],
        "dispatch_rate": summary["dispatch_rate"],
        "hold_rate": summary["hold_rate"],
        "top_actions": _top_actions(report),
        "route_distribution": summary["route_distribution"],
        "mode_distribution": summary["mode_distribution"],
        "reorder_distribution": summary["reorder_distribution"],
        "service_proxy_correlation": summary["service_proxy_correlation"],
    }


def _write_replay_comparison(large: dict[str, Any], bounded: dict[str, Any]) -> None:
    comparison = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "classification": "PUBLIC_REPLAY_LARGE_SAMPLE_COMPARISON_READY",
        "datasets": {},
    }
    for name, large_item in large.items():
        bounded_item = bounded[name]
        comparison["datasets"][name] = {
            "bounded": bounded_item,
            "large": large_item,
            "row_count_multiplier": large_item["row_count"] / bounded_item["row_count"],
            "interpretation_stability": (
                "stable descriptive interpretation: action 24/32 remain rare and public replay remains "
                "non-causal proxy evidence"
            ),
            "important_change": (
                "large sample improves descriptive public replay stability but does not add propensities, "
                "company rewards, or full trajectories"
            ),
        }
    _write_json_fresh(BASE / "replay_large_samples" / "replay_large_sample_comparison.json", comparison)
    _write_fresh(
        ROOT / "docs" / "reports" / "20260614_public_replay_large_sample_comparison_tr.md",
        f"""# Public Replay Large Sample Comparison - 2026-06-14

## Classification

`PUBLIC_REPLAY_LARGE_SAMPLE_COMPARISON_READY`

Large replay expanded the bounded historical replay from `5k/10k/5k` rows to:

- LaDe: `{large['LaDe']['row_count']}` rows
- NYC HVFHS: `{large['NYC_HVFHS']['row_count']}` rows
- Olist: `{large['Olist']['row_count']}` rows

The interpretation is stable: production policy can be run read-only on public proxy states, action 24/32 are rare in these public samples, and the evidence remains descriptive rather than causal OPE.

```json
{json.dumps(comparison['datasets'], indent=2, ensure_ascii=False)}
```

Large sample size improves descriptive stability. It does not create logged propensities, unchosen alternatives, company costs, inventory/fleet economics, or full trajectories.
""",
    )


def _write_truth_index() -> dict[str, Any]:
    truth_index = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "classification": "FINAL_TRUTH_SOURCE_INDEX_READY",
        "production_truth": {
            "logical_model_id": "joint_torch_v5_prod_hierarchical_v1_1m_20260611",
            "runtime": "torch_joint",
            "architecture": "hierarchical_v1",
            "init": "flat_teacher_distillation_v1",
            "contract": "physical_reality_v5_route_candidate_visibility",
            "obs_dim": 73,
            "continuous_dim": 5,
            "discrete_count": 48,
            "status": "active registry and copy-only promoted production",
        },
        "latest_truth_sources": {
            "internal_production_and_residuals": "docs/runs/20260611_hierarchical_v1_equal_budget_residual_watch_eval.md",
            "benchmark_suite_spec": "docs/reports/20260613_codex_5pl_benchmark_suite_spec_tr.md",
            "sota_pathway": "docs/reports/20260613_sota_pathway_execution_results_tr.md",
            "public_data_expansion": "docs/reports/20260614_public_data_expansion_results_tr.md",
            "fleet_dispatch": "docs/reports/20260614_fleet_dispatch_score_upgrade_results_tr.md",
            "historical_replay_large": "docs/reports/20260614_public_replay_large_sample_comparison_tr.md",
            "final_scorecard": "docs/reports/20260614_final_integrated_benchmark_scorecard_tr.md",
        },
        "superseded_by_newer_results": [
            {
                "old": "docs/runs/20260614_lade_public_replay_probe_report.md",
                "new": "docs/runs/20260614_lade_large_public_replay_report.md",
                "reason": "5k bounded replay superseded by all 31,415-row replay",
            },
            {
                "old": "docs/runs/20260614_nyc_hvfhs_public_replay_probe_report.md",
                "new": "docs/runs/20260614_nyc_hvfhs_large_public_replay_report.md",
                "reason": "10k replay superseded by 100k deterministic sample",
            },
            {
                "old": "docs/runs/20260614_olist_public_replay_probe_report.md",
                "new": "docs/runs/20260614_olist_large_public_replay_report.md",
                "reason": "5k replay superseded by all feasible delivered rows",
            },
        ],
        "stale_docs_do_not_use_as_final_truth": [
            "bounded 3 episode rule baseline",
            "13-route Amazon sample as final truth",
            "OR-Tools smoke as final truth",
            "FleetPy single scenario as complete fleet evidence",
            "SVRPBench fully blocked without later geometry/appear-time nuance",
            "no historical replay adapter",
        ],
        "advisor_safe_priority_order": [
            "production/equal-budget gate evidence",
            "large public replay and public-data expansions as proxy evidence",
            "benchmark suite/spec and reproducibility docs",
            "company-data bridge/requirements",
            "older bounded/smoke docs only as chronology",
        ],
    }
    _write_json_fresh(BASE / "final_synthesis" / "final_truth_source_index.json", truth_index)
    _write_fresh(
        ROOT / "docs" / "reports" / "20260614_final_truth_source_index_tr.md",
        """# Final Truth Source Index - 2026-06-14

## Classification

`FINAL_TRUTH_SOURCE_INDEX_READY`

## Current Production Truth

- Logical model: `joint_torch_v5_prod_hierarchical_v1_1m_20260611`
- Runtime: `torch_joint`
- Architecture: `hierarchical_v1`
- Init: `flat_teacher_distillation_v1`
- Contract: `physical_reality_v5_route_candidate_visibility`
- Observation/action: `73 / 5 continuous / 48 discrete`
- State: active registry and copy-only promoted production.

## Advisor-Safe Source Priority

1. Production/equal-budget internal evidence.
2. Final integrated scorecard and benchmark suite spec.
3. Large public replay plus public-data/fleet/route/inventory benchmark packages as proxy evidence.
4. Company-data bridge and request package.
5. Older bounded/smoke docs only as chronology, not current truth.

## Superseded or Stale Claims

- 5k/10k/5k public replay reports are superseded by large replay reports.
- Bounded 3-episode or smoke benchmark statements are not final benchmark truth.
- 13-route Amazon small-sample statements are schema/proxy context, not final route benchmark truth.
- FleetPy single-scenario wording is superseded by later no-Gurobi RPP/IRS package and HVFHV/SF taxi evidence.
- SVRPBench is not simply fully blocked: stochastic raw fields are missing, but geometry/dynamic appear-time evidence exists.

Machine-readable index: `reports/benchmarks/final_evidence_20260614/final_synthesis/final_truth_source_index.json`
""",
    )
    return truth_index


def _write_scorecard() -> dict[str, Any]:
    rows = [
        ("simulator-production readiness", 4.6, 4.7, "Hierarchical v1 active/copy-promoted; gates and residual watch passed.", "internal gate and production audits", "company telemetry still absent", "Do not call this live deployment proof."),
        ("old-production comparison", 4.5, 4.6, "Equal-budget old-production comparison passed.", "20260611 equal-budget residual-watch eval", "synthetic scenario comparator", "Do not claim real-world superiority."),
        ("multi-seed robustness", 4.1, 4.2, "Multi-seed package exists from SOTA/full completion sprint.", "multi-seed robustness benchmark root", "not company telemetry replay", "No global SOTA claim."),
        ("runtime latency", 4.6, 4.7, "CPU inference is low-latency and replay policy loads succeeded.", "runtime latency repeated benchmark and replay smoke", "hardware dependent", "Not a production SLA guarantee."),
        ("rule-based baseline maturity", 4.3, 4.5, "Continuous-aware full rule benchmark supersedes bounded smoke.", "5120 episode continuous-aware benchmark", "simulator baselines only", "No claim that rules are industry optimal."),
        ("inventory/reorder external evidence", 4.5, 4.5, "gym-invmgmt and MABIM evidence remain strong.", "inventory/MABIM reports", "company reorder economics absent", "No no-reorder economic safety claim."),
        ("route/stochastic public evidence", 4.3, 4.4, "PyVRP/OR-Tools/Amazon/SVRP geometry broaden route evidence.", "route reference reports", "stochastic raw fields missing", "No full 5PL route optimality claim."),
        ("fleet/dispatch external evidence", 4.1, 4.2, "HVFHV, FleetPy, SF taxi, and NYC TLC strengthen analog evidence.", "fleet dispatch upgrade sprint", "Chicago/OpenMines blockers and no company contracts", "No secondary-fleet economics claim."),
        ("public historical replay maturity", 4.35, 4.45, "Replay expanded to LaDe 31,415, HVFHS 100k, Olist 96,476.", "large replay reports", "no propensities/rewards/full trajectories", "Descriptive proxy replay only."),
        ("company-data validation readiness", 4.35, 4.45, "Synthetic harness plus company bridge define the path.", "company replay bridge and request package", "no private extract ingested", "No company validation claim."),
        ("SOTA pathway maturity", 4.3, 4.45, "Integrated across gates, baselines, public analogs, replay, blockers.", "SOTA pathway and final scorecard", "external blockers and no company OPE", "Frame as pathway, not global SOTA."),
        ("thesis/advisor defensibility", 4.6, 4.75, "Final packages distinguish simulator, public proxy, and company requirements.", "final advisor/thesis docs", "citation placeholders remain", "Do not overstate literature novelty."),
        ("real-world deployment readiness", 3.2, 3.25, "Protected production artifact exists but no company/live integration.", "production handoff and ops bundles", "no company replay/OPE/live telemetry", "Not live TMS/WMS/ERP ready."),
    ]
    scorecard = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "classification": "FINAL_INTEGRATED_SCORECARD_READY_WITH_LIMITATIONS",
        "scores": [
            {
                "dimension": dimension,
                "previous_score": previous,
                "final_score": final,
                "reason": reason,
                "strongest_evidence": evidence,
                "limiter": limiter,
                "company_data_dependency": "yes" if "company" in limiter.lower() or "telemetry" in limiter.lower() else "partial/no",
                "overclaim_warning": warning,
            }
            for dimension, previous, final, reason, evidence, limiter, warning in rows
        ],
    }
    _write_json_fresh(BASE / "final_scorecard" / "final_integrated_scorecard.json", scorecard)
    table = "\n".join(
        f"| {row['dimension']} | {row['previous_score']} | {row['final_score']} | {row['strongest_evidence']} | {row['limiter']} |"
        for row in scorecard["scores"]
    )
    _write_fresh(
        ROOT / "docs" / "reports" / "20260614_final_integrated_benchmark_scorecard_tr.md",
        f"""# Final Integrated Benchmark Scorecard - 2026-06-14

## Classification

`FINAL_INTEGRATED_SCORECARD_READY_WITH_LIMITATIONS`

| Dimension | Previous | Final | Strongest Evidence | Limiter |
|---|---:|---:|---|---|
{table}

## Interpretation

The final scorecard is stronger than the 2026-06-13 package because it integrates full public-data expansion, fleet/dispatch upgrade evidence, and large public historical replay. It remains deliberately conservative: public replay is not causal OPE, company-data validation is not yet performed, and global SOTA is not claimed.

Machine-readable scorecard: `reports/benchmarks/final_evidence_20260614/final_scorecard/final_integrated_scorecard.json`
""",
    )
    return scorecard


def _write_stale_doc_cleanup() -> list[str]:
    targets = [
        "docs/reports/20260613_final_advisor_showcase_pack_tr.md",
        "docs/reports/20260613_sunum_slayt_icerik_taslagi_tr.md",
        "docs/reports/20260613_demo_script_and_qna_tr.md",
        "docs/thesis/20260613_lisans_tezi_ilk_taslak_tr.md",
    ]
    addendum = """

## 2026-06-14 Final Truth Addendum

This document is retained for chronology, but final advisor/thesis claims must use the 2026-06-14 evidence package. Superseded wording includes bounded 3-episode rule-baseline framing, 13-route Amazon sample as current route truth, OR-Tools smoke as current route truth, FleetPy single-scenario as complete fleet truth, SVRPBench as simply fully blocked, and lack of a historical replay adapter.

Current truth:

- Full rule-based and continuous-aware benchmark evidence is available.
- Amazon/OR-Tools/PyVRP/SVRP geometry and route-reference evidence supersede smoke-only wording.
- Fleet/dispatch evidence now includes HVFHV, FleetPy no-Gurobi RPP/IRS, SF taxi, and NYC TLC proxies, with remaining blockers documented.
- Public historical replay adapter exists and large replay ran on LaDe 31,415 rows, NYC HVFHS 100,000 rows, and Olist 96,476 feasible delivered rows.
- Public replay is descriptive proxy evidence only; no causal OPE, global SOTA, secondary-fleet economics, no-reorder optimality, or company-data validation is claimed.

Use these final sources instead:

- `docs/reports/20260614_final_truth_source_index_tr.md`
- `docs/reports/20260614_final_integrated_benchmark_scorecard_tr.md`
- `docs/reports/20260614_final_advisor_master_package_tr.md`
- `docs/thesis/20260614_lisans_tezi_guncel_taslak_tr.md`
"""
    updated: list[str] = []
    for rel in targets:
        path = ROOT / rel
        text = path.read_text(encoding="utf-8")
        if "## 2026-06-14 Final Truth Addendum" not in text:
            path.write_text(text.rstrip() + addendum + "\n", encoding="utf-8")
            updated.append(rel)
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "classification": "STALE_DOC_CLEANUP_READY",
        "updated_files": updated,
        "superseded_patterns": [
            "bounded 3 episode rule baseline",
            "13-route Amazon sample current truth",
            "OR-Tools smoke current truth",
            "FleetPy single scenario current truth",
            "SVRPBench fully blocked simple wording",
            "no historical replay adapter",
        ],
    }
    _write_json_fresh(BASE / "stale_doc_cleanup" / "stale_doc_cleanup_report.json", report)
    _write_fresh(
        ROOT / "docs" / "runs" / "20260614_stale_doc_cleanup_report.md",
        "# Stale Doc Cleanup Report - 2026-06-14\n\n"
        "## Classification\n\n`STALE_DOC_CLEANUP_READY`\n\n"
        "Updated advisor-facing legacy docs with a 2026-06-14 final truth addendum:\n\n"
        + "\n".join(f"- `{item}`" for item in updated)
        + "\n\nMachine-readable cleanup report: `reports/benchmarks/final_evidence_20260614/stale_doc_cleanup/stale_doc_cleanup_report.json`\n",
    )
    return updated


def _write_advisor_package(large: dict[str, Any]) -> None:
    probe_lines = "\n".join(
        f"- {name}: rows `{item['row_count']}`, action24 `{item['action_24_rate']}`, "
        f"action32 `{item['action_32_rate']}`, missingness `{item['mean_missingness_rate']}`"
        for name, item in large.items()
    )
    _write_fresh(
        ROOT / "docs" / "reports" / "20260614_final_advisor_master_package_tr.md",
        f"""# Final Advisor Master Package - 2026-06-14

## Classification

`FINAL_ADVISOR_MASTER_PACKAGE_READY`

## Project in One Sentence

CODEX-5PL is a 5PL digital-twin decision-control project that combines continuous operational controls and a 48-action hierarchical tactical policy for dispatch, route family, fleet mode, and reorder decisions under the `physical_reality_v5_route_candidate_visibility` contract.

## What the Model Actually Does

The current production model `joint_torch_v5_prod_hierarchical_v1_1m_20260611` maps a 73-dimensional simulator observation to 5 continuous PPO controls and one DQN discrete action in `0..47`.

## Directly Tested Evidence

- Hierarchical v1 1M passed internal 250k, 500k, and 1M gates.
- Equal-budget residual-watch comparison against old production passed.
- Production artifacts are active/copy-promoted and protected by artifact-health/ops bundle checks.

## Public Proxy Evidence

{probe_lines}

This proves adapter feasibility, coverage/missingness reporting, and production policy action tendencies on public proxy states. It does not prove causal superiority.

## If Asked: Is This Real-World Ready?

Answer: it is simulator-production ready with protected artifacts and monitoring runbooks. It is not live TMS/WMS/ERP deployment ready until company-data replay, telemetry validation, and integration tests are approved and completed.
""",
    )
    appendix_rows = "\n".join(
        f"| {name} | {item['row_count']} | {item['action_24_rate']} | {item['action_32_rate']} | {item['dispatch_rate']} | {item['mean_missingness_rate']} |"
        for name, item in large.items()
    )
    _write_fresh(
        ROOT / "docs" / "reports" / "20260614_final_benchmark_appendix_for_advisor_tr.md",
        f"""# Final Benchmark Appendix for Advisor - 2026-06-14

## Classification

`FINAL_BENCHMARK_APPENDIX_READY`

## Evidence Families

1. Internal simulator-production: hierarchical v1 1M, equal-budget gate PASS, residual watches acceptable.
2. Rule-based and continuous-aware simulator baselines.
3. Inventory/reorder public references: gym-invmgmt and MABIM/ReplenishmentEnv.
4. Route references: PyVRP, OR-Tools, Amazon route proxy, SVRP geometry/dynamic appear-times.
5. Fleet/dispatch proxies: HVFHV, FleetPy no-Gurobi RPP/IRS, SF taxi, NYC TLC.
6. Public historical replay: LaDe, NYC HVFHS, Olist large replay.
7. Company-data readiness: synthetic harness and company replay bridge.

| Dataset | Rows | Action 24 | Action 32 | Dispatch | Missingness |
|---|---:|---:|---:|---:|---:|
{appendix_rows}

Public replay is descriptive. OPE requires propensities, alternatives, rewards, and trajectories.
""",
    )
    _write_fresh(
        ROOT / "docs" / "reports" / "20260614_final_demo_script_and_qna_tr.md",
        """# Final Demo Script and Q&A - 2026-06-14

## Classification

`FINAL_DEMO_SCRIPT_AND_QA_READY`

## Demo Flow

1. Show the production truth: hierarchical v1 1M, 73 input features, 5 continuous outputs, 48 discrete actions.
2. Explain action decomposition with action 24 and action 32 as watch examples.
3. Show internal gate/equal-budget evidence.
4. Show public benchmark families: route, inventory, fleet/dispatch, public data expansion.
5. Show historical replay v0 large samples and missingness/confidence.
6. Close with company-data bridge: what remains needed for causal OPE.

### Is this global SOTA?

No. It is a SOTA pathway for a custom 5PL joint-control benchmark, not a global SOTA claim across all routing/inventory/fleet literature.

### Does public replay prove the model would beat real dispatchers?

No. Public replay is descriptive proxy evidence. It lacks behavior propensities, alternatives, and company reward definitions.

### Is it real-world ready?

It is simulator-production ready and artifact-protected. Live readiness requires company replay, telemetry, integration, and safety validation.
""",
    )
    _write_fresh(
        ROOT / "docs" / "reports" / "20260614_final_sota_positioning_for_advisor_tr.md",
        """# Final SOTA Positioning for Advisor - 2026-06-14

## Classification

`FINAL_SOTA_POSITIONING_READY_WITH_LIMITATIONS`

## Safe Position

CODEX-5PL contributes a joint-control benchmark and production-grade simulator candidate for 5PL decisioning. The evidence combines internal gates, rule baselines, public route/inventory/fleet analogs, and public historical replay.

## Unsafe Position

Do not claim global SOTA, real-world optimality, or causal counterfactual superiority.

## Remaining SOTA Gap

A publication-grade SOTA claim would need company telemetry/OPE or equivalent public logs with propensities, alternatives, rewards, and trajectories.
""",
    )


def _write_thesis_package(large: dict[str, Any]) -> None:
    _write_fresh(
        ROOT / "docs" / "thesis" / "20260614_lisans_tezi_guncel_taslak_tr.md",
        f"""# Lisans Tezi Guncel Taslak - 2026-06-14

## Classification

`FINAL_THESIS_DRAFT_READY_WITH_PLACEHOLDER_CITATIONS`

## Ozet

Bu tez, 5PL operasyonlarında dispatch, rota tercihi, filo modu ve reorder kararlarını aynı karar yüzeyinde ele alan CODEX-5PL dijital ikizini inceler. Sistem, 73 boyutlu gözlemden 5 continuous kontrol ve 48 ayrık taktik aksiyon üretir. Güncel üretim adayı `hierarchical_v1` mimarisi ve `flat_teacher_distillation_v1` başlangıcıyla eğitilmiş 1M modeldir.

## Katki

Tezin katkısı global SOTA iddiası değildir. Katkı; birleşik 5PL karar probleminin tanımı, gated simulator-production model, public proxy benchmark paketi, historical replay v0 adapter ve şirket verisiyle yapılacak replay/OPE için açık veri sözleşmesidir.

## Kanit Katmanlari

1. Simulator evidence: 1M hierarchical production candidate, long-run gates, equal-budget residual-watch PASS.
2. Public proxy evidence: route, inventory, fleet/dispatch, ecommerce, delivery timing and historical replay.
3. Company-data-required evidence: secondary fleet economics, reorder safety, dispatch failure causality, causal OPE.

## Historical Replay v0

Large replay ran on LaDe `{large['LaDe']['row_count']}`, NYC HVFHS `{large['NYC_HVFHS']['row_count']}`, and Olist `{large['Olist']['row_count']}` rows. It reports missingness/confidence and model action tendencies. It is not causal OPE.

## Kaynaklar

Akademik kaynaklar eklenecek. Public dataset ve tool references proje raporlarında listelenmiştir; final tezde formal citation formatına dönüştürülecektir.
""",
    )
    _write_fresh(
        ROOT / "docs" / "thesis" / "20260614_lisans_tezi_benchmark_bolumu_tr.md",
        f"""# Lisans Tezi Benchmark Bolumu - 2026-06-14

## Classification

`FINAL_THESIS_BENCHMARK_SECTION_READY`

## Benchmark Families

- Internal 5PL simulator gates and equal-budget production comparison.
- Rule-based and continuous-aware simulator baselines.
- Route references: PyVRP, OR-Tools, Amazon route proxy, SVRP geometry/dynamic appear times.
- Inventory references: gym-invmgmt and MABIM/ReplenishmentEnv.
- Fleet/dispatch references: HVFHV, FleetPy no-Gurobi RPP/IRS, SF taxi, NYC TLC.
- Historical replay v0: public rows converted to CODEX-like proxy states.

```json
{json.dumps(large, indent=2, ensure_ascii=False)}
```

Simulator evidence tests the actual CODEX policy in the digital twin. Public evidence increases external plausibility and benchmark breadth. Company data is required for causal action-level validation.
""",
    )
    _write_fresh(
        ROOT / "docs" / "thesis" / "20260614_lisans_tezi_sinirliliklar_ve_gelecek_calisma_tr.md",
        """# Lisans Tezi Sinirliliklar ve Gelecek Calisma - 2026-06-14

## Classification

`FINAL_THESIS_LIMITATIONS_READY`

## Sinirliliklar

- Public replay does not include behavior propensities, alternatives, full trajectories, or company reward definitions.
- Secondary-fleet economics cannot be validated from public route/order data.
- Reorder-none safety cannot be validated without inventory and stockout cost logs.
- Real-world deployment readiness requires live telemetry and integration validation.
- Academic citations are not fabricated; citation placeholders remain `kaynak eklenecek` until formal bibliography work.

## Gelecek Calisma

- Company replay v1 with anonymized join-preserving data.
- Contextual bandit OPE slices if propensities exist.
- Sequential OPE/FQE if trajectory coverage exists.
- Optional gated model extensions only after monitored regression or company-data mismatch.
""",
    )


def _write_company_data_package() -> None:
    _write_fresh(
        ROOT / "docs" / "reports" / "20260614_company_data_phone_call_script_tr.md",
        """# Company Data Phone Call Script - 2026-06-14

## Classification

`COMPANY_DATA_PHONE_CALL_SCRIPT_READY`

Merhaba, model egitimi icin degil; mevcut simulator ve public replay bulgularini sirket operasyon verisiyle dogrulamak icin minimum anonim veri alanlarini netlestirmek istiyoruz.

Join keyleri korunmus, anonimleştirilmiş orders, dispatch_attempts, routes, fleet, inventory, costs tablolarına ihtiyacımız var. Isim, telefon, adres gibi PII gerekmez; fakat `order_id`, `dispatch_attempt_id`, `vehicle_id`, `carrier_id`, `route_id`, `sku_id/site_id` gibi join keyler ve timestamp sırası korunmalı.

Bu alanlar action 24/32 icin secondary fleet, route reliability, no-reorder stockout ve dispatch failure nedenlerini test eder. Propensity/action probability varsa OPE mumkun olur; yoksa descriptive replay yapılır.
""",
    )
    _write_fresh(
        ROOT / "docs" / "reports" / "20260614_company_data_minimum_extract_checklist_tr.md",
        """# Company Data Minimum Extract Checklist - 2026-06-14

## Classification

`COMPANY_DATA_MINIMUM_EXTRACT_CHECKLIST_READY`

| Table | Minimum fields | Why |
|---|---|---|
| orders | order_id, created/promised/pickup/delivery timestamps, SLA/priority, SKU/order lines | service, lateness, demand pressure |
| dispatch_attempts | dispatch_attempt_id, order_id, decision_time, action_id, status, failure_reason | action validation and dispatch failure causality |
| routes | route_id, planned route type, planned/actual travel time, failure flag, congestion/disruption | action 24/32 route-side validation |
| fleet | vehicle_id, carrier_id, fleet_type, availability, accepted, capacity, cost | secondary fleet economics |
| inventory | order_id, sku_id/site_id, stock on hand, reserved, in transit, stockout after decision, reorder type | reorder/no-reorder validation |
| costs | primary/secondary fleet cost, stockout, holding, lateness, transport costs | reward/cost model |

PII can be removed. Join keys and timestamp ordering must remain stable.
""",
    )
    _write_fresh(
        ROOT / "docs" / "reports" / "20260614_company_replay_why_needed_tr.md",
        """# Why Company Replay Is Needed - 2026-06-14

## Classification

`COMPANY_REPLAY_WHY_NEEDED_READY`

Public replay v0 proves that public rows can be converted to CODEX-like proxy states and that the production policy can be run read-only. It does not validate company-specific secondary fleet contracts, reorder economics, dispatch failure causes, or causal policy value.

Company replay v1 converts the same adapter idea into high-confidence operational replay by using company state, actions, outcomes, costs, and optionally behavior propensities. With propensities and trajectories, OBP/SCOPE-RL/d3rlpy-style OPE becomes feasible; without them, descriptive replay still validates failure modes and action tendencies.
""",
    )


def _write_blocker_register() -> None:
    blockers = [
        ("Mendeley planned-vs-actual", "HTTP 403", "planned-vs-actual route comparison", "minor thesis blocker; route evidence exists", "medium", "request access or alternative mirror"),
        ("Chicago TNP", "HTTP 503", "additional dispatch proxy", "not thesis-blocking", "low-medium", "retry Socrata later"),
        ("Chicago Taxi", "HTTP 503", "additional fleet/trip proxy", "not thesis-blocking", "low-medium", "retry Socrata later"),
        ("OpenMines", "Python 3.10/3.11 isolation blocker", "mining dispatch simulator analog", "not thesis-blocking", "medium", "use compatible env"),
        ("Instacart", "legacy links 404", "inventory/reorder public data", "not blocking due Tesco/MABIM/gym evidence", "low-medium", "find maintained mirror"),
        ("M5", "Zenodo 403", "demand/inventory public data", "not blocking due other inventory evidence", "low-medium", "manual approved access"),
        ("SVRPBench stochastic raw fields", "released parquet lacks fields", "stochastic congestion/delay validation", "not thesis-blocking; geometry/dynamic fields exist", "medium", "source raw generator or contact authors"),
        ("LaDe license caveat", "publication reuse caveat", "publication-grade reuse", "must resolve before formal publication", "medium", "manual license review"),
        ("CityFlow", "source-build deferred", "traffic/congestion analog", "not thesis-blocking", "low-medium", "separate source-build sprint"),
    ]
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "classification": "FINAL_EXTERNAL_BLOCKER_REGISTER_READY",
        "blockers": [
            {
                "name": name,
                "blocker": blocker,
                "prevents": prevents,
                "thesis_impact": thesis,
                "sota_impact": sota,
                "safe_next_step": next_step,
            }
            for name, blocker, prevents, thesis, sota, next_step in blockers
        ],
    }
    _write_json_fresh(BASE / "blocker_register" / "final_external_blocker_register.json", payload)
    table = "\n".join(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} |" for row in blockers)
    _write_fresh(
        ROOT / "docs" / "reports" / "20260614_final_external_blocker_register_tr.md",
        f"""# Final External Blocker Register - 2026-06-14

## Classification

`FINAL_EXTERNAL_BLOCKER_REGISTER_READY`

| Source | Blocker | Prevents | Thesis impact | SOTA impact | Safe next step |
|---|---|---|---|---|---|
{table}
""",
    )


def _write_reproducibility_manifest(truth_index: dict[str, Any]) -> None:
    try:
        import numpy
        import pandas
        import pyarrow
        import torch

        versions = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "pyarrow": pyarrow.__version__,
            "torch": torch.__version__,
        }
    except Exception as exc:  # pragma: no cover
        versions = {"python": sys.version.split()[0], "platform": platform.platform(), "version_error": str(exc)}
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "classification": "FINAL_REPRODUCIBILITY_MANIFEST_READY",
        "dependency_versions": versions,
        "commands": [
            "python -m unittest tests.benchmarks.test_public_historical_replay_adapter -v",
            "python -m unittest tests.benchmarks.test_public_historical_replay_adapter tests.benchmarks.test_company_data_replay_validator tests.benchmarks.test_public_data_expansion_benchmarks tests.benchmarks.test_fleet_dispatch_upgrade_benchmarks tests.orchestration.test_production_artifact_health_report tests.orchestration.test_monitoring_report_generator tests.orchestration.test_production_ops_bundle -v",
            "python -m py_compile scripts\\public_historical_replay_adapter.py scripts\\final_evidence_integration_writer.py tests\\benchmarks\\test_public_historical_replay_adapter.py",
            "python scripts\\run_production_ops_bundle.py --output reports\\ops\\20260614_final_evidence_integration_final_ops_bundle_report.json",
        ],
        "data_paths": [
            "data/public/lade_20260614/delivery_jl.parquet",
            "data/public/nyc_hvfhs_20260614/fhvhv_tripdata_2023-01.parquet",
            "data/public/olist_20260614/",
        ],
        "protected_paths": ["models/registry", "models/production", "models/baselines", "db", "models/checkpoints", "models/eval"],
        "truth_source_order": truth_index["advisor_safe_priority_order"],
    }
    _write_json_fresh(BASE / "final_synthesis" / "final_reproducibility_manifest.json", manifest)
    _write_fresh(
        ROOT / "docs" / "runbooks" / "20260614_final_reproducibility_and_commands_runbook.md",
        """# Final Reproducibility and Commands Runbook - 2026-06-14

## Classification

`FINAL_REPRODUCIBILITY_AND_COMMANDS_RUNBOOK_READY`

## Rerun Safety Rules

Do not write under `models/registry`, `models/production`, `models/baselines`, `db`, existing `models/checkpoints`, or existing `models/eval`. Do not train, register, activate, promote, update baselines, mutate DB, or create training configs.

## Final Commands

```powershell
python -m unittest tests.benchmarks.test_public_historical_replay_adapter -v
python -m unittest tests.benchmarks.test_public_historical_replay_adapter tests.benchmarks.test_company_data_replay_validator tests.benchmarks.test_public_data_expansion_benchmarks tests.benchmarks.test_fleet_dispatch_upgrade_benchmarks tests.orchestration.test_production_artifact_health_report tests.orchestration.test_monitoring_report_generator tests.orchestration.test_production_ops_bundle -v
python -m py_compile scripts\\public_historical_replay_adapter.py scripts\\final_evidence_integration_writer.py tests\\benchmarks\\test_public_historical_replay_adapter.py
python scripts\\run_production_ops_bundle.py --output reports\\ops\\20260614_final_evidence_integration_final_ops_bundle_report.json
```

## Data Paths

- `data/public/lade_20260614/delivery_jl.parquet`
- `data/public/nyc_hvfhs_20260614/fhvhv_tripdata_2023-01.parquet`
- `data/public/olist_20260614/`

## Output Root

`reports/benchmarks/final_evidence_20260614/`

## Stale-Doc Warning

Use 2026-06-14 final truth-source index and scorecard before quoting older bounded/smoke docs.
""",
    )


def _update_dashboard() -> None:
    path = ROOT / "docs" / "00_PROJECT_DASHBOARD.md"
    text = path.read_text(encoding="utf-8")
    if "Latest final CODEX-5PL truth-source index" in text:
        return
    marker = "- Latest company replay bridge:\n  [[docs/runbooks/20260614_company_replay_bridge_to_model_policy_spec]]\n"
    insert = (
        "- Latest final CODEX-5PL truth-source index:\n  [[docs/reports/20260614_final_truth_source_index_tr]]\n"
        "- Latest final integrated benchmark scorecard:\n  [[docs/reports/20260614_final_integrated_benchmark_scorecard_tr]]\n"
        "- Latest final advisor master package:\n  [[docs/reports/20260614_final_advisor_master_package_tr]]\n"
        "- Latest final thesis draft:\n  [[docs/thesis/20260614_lisans_tezi_guncel_taslak_tr]]\n"
        "- Latest final evidence integration audit:\n  [[docs/runs/20260614_final_evidence_integration_sprint_audit]]\n"
    )
    path.write_text(text.replace(marker, marker + insert), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
