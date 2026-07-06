# Tez Danismani Operasyon Walkthrough Diyagram Notlari

Tarih: 2026-06-12

Bu dosya, sunum veya rapor icin kullanilabilecek diyagram notlarini icerir. Diyagramlar gercek step trace degil, kodla uyumlu operasyon akisini gosterir.

## 1. One-Operation Sequence

```mermaid
sequenceDiagram
    participant Env as "env_5pl digital twin"
    participant Obs as "Observation Builder (73)"
    participant PPO as "PPO continuous policy"
    participant DQN as "Hierarchical DQN policy"
    participant Map as "Action Mapper"
    participant Safe as "Safety / Feasibility"
    participant Sim as "Simulated logistics world"
    participant Mon as "Monitoring / Metrics"

    Env->>Obs: Build current operation state
    Obs->>PPO: 73-dim observation
    Obs->>DQN: 73-dim observation
    PPO->>Safe: 5 continuous controls
    DQN->>Map: action id 0..47
    Map->>Safe: dispatch + route + fleet + reorder
    Safe->>Sim: projected safe command
    Sim->>Env: next state, reward, info
    Env->>Mon: service, lateness, dispatch, route, no-work counters
```

## 2. PPO ve DQN Ayrimi

```mermaid
flowchart LR
    O["Observation 73"] --> PPO["PPO"]
    O --> DQN["hierarchical_v1 DQN"]
    PPO --> C["Continuous controls: reorder_fraction, dispatch_intensity, speed_multiplier, safety_stock_multiplier, capacity_buffer_fraction"]
    DQN --> A["Discrete action 0..47"]
    C --> S["SafetyProjector"]
    A --> M["DiscreteActionMapper"]
    M --> S
    S --> E["env_5pl applies command"]
```

Ana mesaj:

- PPO ince ayar kollari gibi calisir.
- DQN ana taktik operasyon karari verir.
- Joint modda PPO dispatch icin makro release/budget etkisi verir, fakat dogrudan dispatch'i DQN yapar.

## 3. Action 24 Decode Diyagrami

```mermaid
flowchart TD
    A24["Action id 24"] --> R0["24 mod 4 = 0: reorder none"]
    A24 --> M0["(24 // 4) mod 2 = 0: secondary_fleet"]
    A24 --> Route0["(24 // 8) mod 3 = 0: shortest route"]
    A24 --> D1["24 // 24 = 1: dispatch"]
    R0 --> Cmd["dispatch + shortest + secondary_fleet + none"]
    M0 --> Cmd
    Route0 --> Cmd
    D1 --> Cmd
```

Sunum cumlesi:

> Action 24 tek bir sayi gibi gorunur, fakat aslinda dort parcali bir lojistik komuttur: dispatch yap, shortest route kullan, secondary fleet kullan, reorder override yapma.

## 4. Action 32 Decode Diyagrami

```mermaid
flowchart TD
    A32["Action id 32"] --> R0["32 mod 4 = 0: reorder none"]
    A32 --> M0["(32 // 4) mod 2 = 0: secondary_fleet"]
    A32 --> Route1["(32 // 8) mod 3 = 1: low_congestion route"]
    A32 --> D1["32 // 24 = 1: dispatch"]
    R0 --> Cmd["dispatch + low_congestion + secondary_fleet + none"]
    M0 --> Cmd
    Route1 --> Cmd
    D1 --> Cmd
```

Sunum cumlesi:

> Action 32 route disruption altinda daha dusuk congestion maliyeti olan rota ailesini tercih eden dispatch kararidir.

## 5. Simulator ve Gercek Dunya Siniri / Real-world Boundary

```mermaid
flowchart LR
    subgraph Simulator["Simulasyon tarafinda dogrulanan"]
        A["env_5pl closed loop"]
        B["8 scenario eval"]
        C["long-run gate PASS"]
        D["artifact health / monitoring / ops bundle"]
    end

    subgraph RealWorld["Gercek dunya icin gereken"]
        E["Company orders"]
        F["Dispatch attempts"]
        G["Carrier / fleet availability"]
        H["Inventory / reorder / costs"]
        I["TMS / WMS / ERP integration"]
    end

    Simulator --> Boundary["Simulation-to-real boundary"]
    Boundary --> RealWorld
```

Ana mesaj:

- Proje simule digital twin icinde otonomdur.
- Gercek dunyada full autonomous claim icin sirket verisi ve entegrasyon gerekir.

## 6. Monitoring Watch Diyagrami

```mermaid
flowchart TD
    Act["Action 24 / 32 selected"] --> Counters["Operational counters"]
    Counters --> NC["no-current-work"]
    Counters --> NU["no-unassigned"]
    Counters --> FN["failed-noop"]
    Counters --> RF["route_failure"]
    Counters --> NV["no_vehicle"]
    Counters --> AA["already_assigned"]
    Counters --> S["service / lateness"]
    NC --> Decision["Monitoring decision"]
    NU --> Decision
    FN --> Decision
    RF --> Decision
    NV --> Decision
    AA --> Decision
    S --> Decision
```

Ana mesaj:

> Action concentration tek basina hata degildir. Hata olup olmadigini service, lateness ve no-work/failure counter'lariyla birlikte okuyoruz.
