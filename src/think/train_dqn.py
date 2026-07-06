"""Deprecated DQN-only training entry point.

The project now trains DQN only through the synchronized joint PPO+DQN runner.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_CONFIG = Path("configs/training_joint.json")

def main() -> None:
    raise SystemExit(
        "DQN-only training is deprecated. Run "
        "`python -m src.learn.train_joint_torch` with "
        f"{Path(DEFAULT_CONFIG)} instead."
    )


if __name__ == "__main__":
    main()
