# Chicago Taxi Fleet Proxy Report - 2026-06-14

## Classification

`CHICAGO_TAXI_FLEET_BLOCKED_SOCRATA_503`

JSON report:

- `reports/benchmarks/fleet_dispatch_upgrade_20260614/chicago_taxi_fleet/chicago_taxi_fleet_report.json`

## Attempted Sources

- `https://data.cityofchicago.org/Transportation/Taxi-Trips-2024-/ajtu-isnz`
- `https://data.cityofchicago.org/Transportation/Taxi-Trips-2013-2023-/wrvz-psew`

Attempted one-row/count SODA endpoints returned HTTP 503.

## Expected Value If Access Recovers

Chicago Taxi should provide anonymous taxi id, trip start/end, miles, duration, pickup/dropoff area, fare, and trip total. That would support trips-per-vehicle, utilization, movement, and approximate repositioning metrics.

## Substitute Evidence

The sprint completed San Francisco taxi trajectory movement evidence instead. That is weaker than Chicago Taxi for utilization because it lacks trip economic fields, but it still provides public movement/spatial coverage evidence.

