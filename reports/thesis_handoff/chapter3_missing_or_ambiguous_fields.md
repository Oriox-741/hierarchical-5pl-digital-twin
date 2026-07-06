# Chapter 3 Missing or Ambiguous Fields

Generated UTC: `2026-06-23T00:16:34.601185+00:00`

Fields listed here were requested but absent, ambiguous, or intentionally represented by metadata rather than binary artifacts. No values were inferred.

| Requested field/path | Replacement or value | Reason/limitation |
| --- | --- | --- |
| src/act/reward_model.py | src/think/rewards.py | requested path absent; src/think/rewards.py defines RewardModel and RewardWeights |
| PPO learning-rate schedule | NOT_FOUND_IN_REPOSITORY | No scheduler field or scheduler source use found. |
| Number of vectorized environments | NOT_FOUND_IN_REPOSITORY | Trainer constructs one FivePLDigitalTwinEnv instance; no n_envs config found. |
| Dynamic curriculum stage-transition criteria | NOT_FOUND_IN_REPOSITORY | Final configs contain fixed stage_schedule entries, not performance-triggered criteria. |
| Dedicated JSON config schema | NOT_FOUND_IN_REPOSITORY | No config schema file was located under configs/. |
| Checkpoint metadata file separate from binary | production manifest and registry metadata used | Large checkpoint binaries were intentionally excluded. |
| Terminal/truncation reward coefficient | NOT_FOUND_IN_REPOSITORY | No distinct terminal or truncation reward coefficient located. |
