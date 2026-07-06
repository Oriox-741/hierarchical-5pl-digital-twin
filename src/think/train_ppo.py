"""Deprecated PPO-only training entry point.

The project now trains PPO only through the synchronized joint PPO+DQN runner.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_CONFIG = Path("configs/training_joint.json")

def main() -> None:
    raise SystemExit(
        "PPO-only training is deprecated. Run "
        "`python -m src.learn.train_joint_torch` with "
        f"{Path(DEFAULT_CONFIG)} instead."
    )


if __name__ == "__main__":
    main()
