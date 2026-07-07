# Ablation Protocol for 5PL Digital-Twin RL Framework

This protocol isolates architectural and control components of the 5PL digital-twin RL framework. Results should not be claimed unless the corresponding report artifact is present and generated under the documented evaluation protocol.

## Minimum Variants

| ID | Variant | Implementation | Question | Metrics |
| --- | --- | --- | --- | --- |
| A0 | Neutral PPO evaluation | Replace PPO outputs with neutral projected values during evaluation | Does PPO add value beyond DQN? | service, lateness, dispatch success, stockout, cost |
| A1 | Route-candidate masking | Neutralize 16 route-candidate features | Does route visibility help route disruption? | route failures, lateness, route-family use |
| A2 | Safety-projection diagnostic | Record projection counts and reasons while safety remains enabled by default | How often does safety projection rescue policy output? | projection/block rates, hard blockers |
| A3 | Flat-vs-hierarchical comparison | Compare preceding flat model under same scenarios/seeds/budget | Does factorization help? | service delta, hard blockers, action concentration |
| A4 | No-teacher-distillation training-time plan | Train hierarchical model from cold start under same budget in a future approved run | Does distillation improve sample efficiency? | pass rate, early blockers, stability |
| A5 | Reward-blend sensitivity training-time plan | Vary PPO/DQN local-global reward weights in future approved runs | Is performance robust to reward shaping? | service, cost, stockout, action diversity |

## Required Controls

- Same eight scenario configs.
- Same episode count and deterministic inference mode.
- Same seed policy; use seeds 42-46 if compute allows.
- Separate inference-only ablations from retraining ablations.
- Any unsafe sandbox-only diagnostic must require an explicit guardrail flag, remain outside normal use, and never be described as deployable behavior.

## Output Artifacts

Recommended output folder: `reports/ablation/YYYYMMDD_<run_name>/`

Required files:

- `ablation_summary.csv`
- `ablation_report.json`
- `ablation_report.md`
- `episode_metrics.jsonl` if episode-level metrics are generated
- `run_config.json`

## Claim Boundary

Ablation evidence supports component-level interpretation inside the simulator. It does not prove live enterprise performance, company-level economic optimality, causal public-data performance, or global state-of-the-art status.
