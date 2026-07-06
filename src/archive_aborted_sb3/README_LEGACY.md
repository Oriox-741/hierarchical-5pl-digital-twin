# Aborted SB3 Joint-Training Archive

This folder is locked for thesis autopsy and historical comparison only.

The files here belong to the aborted Stable-Baselines3 staggered joint-training path. That design used large alternating PPO/DQN training blocks, created stale-opponent lag, and was rejected as the final MARL architecture.

Rules:

- Do not import these modules from the new PyTorch MARL trainer.
- Do not register checkpoints produced by these scripts as active production models.
- Do not use this folder as a dependency source for new architecture code.
- Keep these files only for reproducibility, thesis evidence, and behavioral autopsy of the failed SB3 approach.

Active final training work must use the planned raw PyTorch synchronized MARL path:

```text
src.learn.train_joint_torch
```
