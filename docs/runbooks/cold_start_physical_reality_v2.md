# Cold Start Runbook: physical_reality_v2

This runbook resets the active training surface for the physically realistic 5PL MDP. It preserves historical artifacts for analysis, but prevents old speed-free policy weights from contaminating new training.

## Contract

- `mdp_contract_version`: `physical_reality_v2`
- Environment: `FivePLDigitalTwinEnv(action_mode="joint", max_steps=288)`
- Horizon: one 288-step rolling operations day
- Decision interval: 300 simulated seconds
- PPO role: strategic continuous envelope control for speed, capacity, planned replenishment, and safety stock posture
- DQN role: tactical dispatch, route, fleet mode, and emergency override only

## Physical Reality Rules

- Speed is priced by nonlinear operating cost and risk; high speed is not free.
- PPO speed changes are baseline-relative, not compounded.
- PPO owns planned inventory posture.
- DQN emergency and aggressive replenishment are costly tactical overrides.
- Proactive inventory is rewarded only under realistic stockout or volatility risk.
- Overstock, unnecessary emergency procurement, unused reserved capacity, and premium routing in calm conditions are penalized.

## Artifact Isolation

The speed-free joint PyTorch run is preserved under:

```text
models/baselines/physical_gap_speed_free/
models/baselines/physical_gap_speed_free/legacy_active_surfaces/
```

Do not pass any checkpoint from that directory to `--resume`.

Active output directories for the new cold start are:

```text
models/checkpoints/joint_torch/
models/checkpoints/ppo/
models/checkpoints/dqn/
```

These directories must contain no legacy `.zip` or `.pt` files before the production cold start. Smoke-run artifacts may remain in `models/checkpoints/joint_torch_smoke/` because the production trainer does not read from that path.

## Required Pre-Training Checks

Run:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
python -m src.learn.pretrain_mdp_sanity_check
```

The sanity gate must show:

- max speed costs more than nominal speed in calm conditions
- max speed is not universally better
- planned reorder helps under stockout risk
- planned reorder hurts under overstock
- safety stock helps under demand volatility
- emergency reorder hurts under low inventory risk
- high-resilience routing is not free in calm conditions
- observations are finite and normalized

## Smoke Training

Create a local smoke copy of `configs/training_joint.json` with:

```text
shared_global_parameters.total_timesteps = 10000
shared_global_parameters.max_steps = 288
mdp_contract_version = physical_reality_v2
```

Run smoke training without database writes if the local database is not running:

```powershell
python -m src.learn.train_joint_torch --config configs/training_joint_smoke.json --disable-trace-logging --skip-registry
```

Expected startup line:

```text
global_step=0 mdp_contract_version=physical_reality_v2 resume=none replay_buffer=empty optimizers=fresh
```

## Production Start

Use the production config unchanged:

```powershell
python -m src.learn.train_joint_torch
```

Do not use `--resume` unless the checkpoint was created under `physical_reality_v2`.
