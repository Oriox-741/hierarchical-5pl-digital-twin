"""Run inference-only real-world scenario evaluations for a trained joint policy."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch

from src.eval.real_world_scenario_arena import (
    build_dry_run_episode,
    build_scenario_environment,
    load_policy_checkpoint,
    load_scenario_configs,
    run_scenario_episode,
    write_evaluation_outputs,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ppo-path", type=Path, default=None, help="Reserved for split-policy checkpoints.")
    parser.add_argument("--dqn-path", type=Path, default=None, help="Reserved for split-policy checkpoints.")
    parser.add_argument("--scenario-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate plumbing and write reports without env rollout.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.ppo_path is not None or args.dqn_path is not None:
        raise ValueError("split PPO/DQN checkpoint paths are reserved but not implemented; use --checkpoint.")
    if args.episodes is not None and args.episodes <= 0:
        raise ValueError("--episodes must be positive when provided.")

    device = torch.device(args.device)
    scenarios = load_scenario_configs(args.scenario_dir)
    policy = load_policy_checkpoint(args.checkpoint, device=device)
    config = policy.checkpoint.get("config", {})

    rows = []
    capabilities_by_scenario: dict[str, list[dict[str, str]]] = {}
    for scenario in scenarios:
        seeds = list(scenario.seeds)
        target_episodes = args.episodes if args.episodes is not None else scenario.episode_count
        if len(seeds) < target_episodes:
            seeds.extend(args.seed + offset for offset in range(len(seeds), target_episodes))
        capability_env, capabilities = build_scenario_environment(config, scenario, seed=seeds[0])
        capability_env.close()
        capabilities_by_scenario[scenario.scenario_id] = capabilities
        for episode_index in range(target_episodes):
            seed = int(seeds[episode_index])
            if args.dry_run:
                rows.append(build_dry_run_episode(scenario, seed=seed, episode_index=episode_index))
            else:
                rows.append(
                    run_scenario_episode(
                        scenario=scenario,
                        policy=policy,
                        seed=seed,
                        episode_index=episode_index,
                        deterministic=bool(args.deterministic),
                    )
                )

    write_evaluation_outputs(
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        scenarios=scenarios,
        episode_rows=rows,
        override_capabilities=capabilities_by_scenario,
    )
    print(f"Wrote evaluation outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
