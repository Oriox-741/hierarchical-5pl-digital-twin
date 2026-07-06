# Lisans Tezi Sinirliliklar ve Gelecek Calisma - 2026-06-14

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
