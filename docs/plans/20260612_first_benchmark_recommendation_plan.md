# First Benchmark Recommendation Plan

Date: 2026-06-13

Scope: recommendation only. This plan does not implement or run any benchmark. It does not train, run offline eval, run long-run gate, download data, install dependencies, mutate registry/production/baselines/DB/checkpoints/eval outputs, create training configs, ingest company data, or prepare a firm-facing data request.

## Executive Recommendation

Recommended first benchmark path:

`Rule-based baseline design/spec`

This is the best first benchmark after the Step 6 docs package because it is:

- understandable to a thesis advisor;
- aligned with the current simulator/action semantics;
- possible to specify without new dependency approval;
- possible to design without company data;
- safer than running eval or training;
- a clean bridge from advisor explanation to future measurable baselines.

No benchmark should be implemented or run in this task.

## Candidate Comparison

| Candidate | Value | Safe now? | Needs approval? | Main blocker | Recommendation |
| --- | --- | --- | --- | --- | --- |
| Rule-based baseline design | High advisor clarity and first internal baseline protocol | yes, design/spec only | implementation/eval later | must avoid changing env physics | choose first |
| OR-Tools feasibility plan | Strong route-side classical baseline | plan only | dependency install and run approval | solver dependency and mapping fidelity | second |
| Amazon full route-proxy expansion | External public route plausibility | plan only | dataset download approval | data size/license and route-only limitation | third |
| Runtime latency benchmark | Productization readiness | plan only | implementation/report approval | not model-quality evidence | later productization |
| Multi-seed robustness eval | Statistical simulator confidence | no-run only | offline eval/gate approval | writes fresh eval outputs | later after eval approval |
| Ablation study | Strong causal architecture evidence | no | training/eval/config approval | requires new training and fresh gates | later after hypothesis |

## Why Rule-Based Baseline Design Comes First

Rule-based baselines are the best next academic step because they answer a simple advisor question:

> Does the learned hierarchical policy do better than obvious logistics heuristics?

They also make the current action semantics easier to explain. The baselines can be described using the same dispatch/route/fleet/reorder structure:

- FIFO + shortest + primary
- earliest due date + shortest
- premium-first
- low-congestion under disruption
- stock-threshold reorder

This makes the comparison understandable before introducing heavier solvers, public datasets, multi-seed evals, or ablations.

## Proposed Rule-Based Baseline Design Scope

Future design/spec should define, without running:

- baseline policy names;
- exact decision rules;
- mapping to the 48 external DQN action space;
- required observation fields or simulator state fields;
- scenario applicability;
- expected metrics;
- stop conditions;
- future output schemas;
- exact approvals required before implementation/eval.

## What It Would Prove Later

If implemented and evaluated later under approval, rule-based baselines could show:

- whether hierarchical v1 beats transparent dispatch/route/reorder heuristics in the same simulator scenarios;
- where learned policy adds value;
- whether action 24/32 concentration is better or worse than simple rules;
- which stress scenarios need stronger baselines or public benchmarks.

## What It Would Not Prove

Even if rule-based baselines are later run, they would not prove:

- real-world optimum;
- live deployment readiness;
- secondary-fleet economics;
- reorder-none economics;
- superiority over OR solvers;
- external public benchmark SOTA.

## Approval Boundary

Safe-now:

- write a rule-based baseline design/spec;
- define rules and metrics;
- define output schemas;
- define future commands as do-not-run examples.

Requires separate approval:

- writing baseline implementation code;
- running simulator/eval;
- writing fresh eval outputs;
- running long-run gate;
- installing dependencies;
- comparing to public downloaded datasets;
- training or ablation.

## Future Goal Candidate

If the user approves the first benchmark path later, create an executable goal file such as:

`docs/goals/20260613_execute_rule_based_baseline_design_goal.txt`

That future goal should remain documentation/specification only unless implementation/eval is explicitly approved.

## Final Recommendation

Proceed next with a rule-based baseline design/spec, not OR-Tools, Amazon full download, multi-seed eval, runtime benchmark, or ablation.

Reason:

- It maximizes advisor clarity per unit of risk.
- It requires no unapproved dependency, data, eval, gate, or training action.
- It prepares the ground for later implementation and fair comparisons.

## Final Classification

`FIRST_BENCHMARK_RULE_BASED_BASELINE_DESIGN_RECOMMENDED`

