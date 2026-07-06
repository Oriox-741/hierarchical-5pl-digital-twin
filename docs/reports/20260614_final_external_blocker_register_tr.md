# Final External Blocker Register - 2026-06-14

## Classification

`FINAL_EXTERNAL_BLOCKER_REGISTER_READY`

| Source | Blocker | Prevents | Thesis impact | SOTA impact | Safe next step |
|---|---|---|---|---|---|
| Mendeley planned-vs-actual | HTTP 403 | planned-vs-actual route comparison | minor thesis blocker; route evidence exists | medium | request access or alternative mirror |
| Chicago TNP | HTTP 503 | additional dispatch proxy | not thesis-blocking | low-medium | retry Socrata later |
| Chicago Taxi | HTTP 503 | additional fleet/trip proxy | not thesis-blocking | low-medium | retry Socrata later |
| OpenMines | Python 3.10/3.11 isolation blocker | mining dispatch simulator analog | not thesis-blocking | medium | use compatible env |
| Instacart | legacy links 404 | inventory/reorder public data | not blocking due Tesco/MABIM/gym evidence | low-medium | find maintained mirror |
| M5 | Zenodo 403 | demand/inventory public data | not blocking due other inventory evidence | low-medium | manual approved access |
| SVRPBench stochastic raw fields | released parquet lacks fields | stochastic congestion/delay validation | not thesis-blocking; geometry/dynamic fields exist | medium | source raw generator or contact authors |
| LaDe license caveat | publication reuse caveat | publication-grade reuse | must resolve before formal publication | medium | manual license review |
| CityFlow | source-build deferred | traffic/congestion analog | not thesis-blocking | low-medium | separate source-build sprint |
