# V8 All-Action Statistics Audit

Public replay rows: `227891`
All 48 action IDs present in Appendix A: `True`

## Top Ten Actions

| Action | Count | Rate | Interpretation |
| --- | --- | --- | --- |
| 1 | 156108 | 0.685012 | hold; shortest route; secondary fleet; conservative reorder |
| 0 | 59217 | 0.259848 | hold; shortest route; secondary fleet; none reorder |
| 2 | 3635 | 0.015951 | hold; shortest route; secondary fleet; aggressive reorder |
| 41 | 2793 | 0.012256 | dispatch; high resilience route; secondary fleet; conservative reorder |
| 43 | 1674 | 0.007346 | dispatch; high resilience route; secondary fleet; emergency reorder |
| 3 | 1195 | 0.005244 | hold; shortest route; secondary fleet; emergency reorder |
| 25 | 1028 | 0.004511 | dispatch; shortest route; secondary fleet; conservative reorder |
| 45 | 782 | 0.003431 | dispatch; high resilience route; primary fleet; conservative reorder |
| 27 | 645 | 0.002830 | dispatch; shortest route; secondary fleet; emergency reorder |
| 29 | 296 | 0.001299 | dispatch; shortest route; primary fleet; conservative reorder |

## Family Distribution

| Family | Value | Count | Rate |
| --- | --- | --- | --- |
| dispatch | dispatch | 7736 | 0.033946 |
| dispatch | hold | 220155 | 0.966054 |
| route | high_resilience | 5390 | 0.023652 |
| route | low_congestion | 3 | 0.000013 |
| route | shortest | 222498 | 0.976335 |
| fleet | primary_fleet | 1287 | 0.005647 |
| fleet | secondary_fleet | 226604 | 0.994353 |
| reorder | aggressive | 3638 | 0.015964 |
| reorder | conservative | 161008 | 0.706513 |
| reorder | emergency | 3617 | 0.015872 |
| reorder | none | 59628 | 0.261651 |

Zero-count actions: `4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26, 33, 34, 35, 36, 38, 39, 44, 46`
