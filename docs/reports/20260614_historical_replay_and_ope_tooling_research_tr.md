# Historical Replay ve OPE Tooling Research - 2026-06-14

## Classification

`HISTORICAL_REPLAY_OPE_TOOLING_RESEARCH_READY`

## ?zet

CODEX ?retim politikas? 73 boyutlu g?zlemden 5 boyutlu continuous kontrol ve 48 ayr?k taktik aksiyon ?retir. Public veri setleri bu politikan?n ger?ek ?irket ortam?nda counterfactual olarak daha iyi oldu?unu kan?tlamaz; fakat public sat?rlar? CODEX-benzeri proxy state'lere d?n??t?r?p action tendency, coverage, missingness ve route/timing plausibility ?l?mek i?in de?erlidir.

Bu sprint i?in g?venli ayr?m:

- Public replay: proxy state, model action distribution, action 24/32 oran?, timing/route proxy korelasyonu.
- OPE: sadece logged behavior action, reward/cost, trajectory continuity ve tercihen propensity/action probability varsa.

## Tooling Katalo?u

Makine-okunur katalog:

`reports/benchmarks/historical_replay_20260614/literature_and_tooling/historical_replay_tooling_catalog.json`

| Ara? | CODEX i?in rol | Gerekli veri | Plug-and-play durumu | S?n?r |
|---|---|---|---|---|
| Open Bandit Pipeline | Tek ad?ml? contextual bandit OPE: dispatch/route/fleet slice | context, action, reward, pscore, evaluation policy distribution | K?smen | Sequential 5PL'yi do?rudan ??zmez |
| SCOPE-RL | Full trajectory offline RL/OPE/OPS metodolojisi | s,a,r,s_next,done ve policy probability/density/value deste?i | K?smen | Public logistics tablolar? do?rudan yeterli de?il |
| d3rlpy FQE | Offline policy selection / DQN aday rank | observations, actions, rewards, terminals, coverage | K?smen | Coverage yoksa extrapolation riski y?ksek |
| D4RL-style | Offline dataset format ve benchmark protokol? | MDP trajectory + reward + env metadata | Metodoloji | Ger?ek lojistik iddias? ?retmez |
| Public logistics datasets | Replay/proxy benchmark | timestamp, location, route/timing/outcome proxy | Replay i?in evet | OPE/counterfactual i?in hay?r |

## Public Data ile G?venli ?ddialar

G?venli:

- "Public historical rows CODEX-like proxy state'e ?evrildi."
- "Modelin action 24/32 e?ilimleri ve missingness/confidence da??l?m? ?l??ld?."
- "Route/timing proxy correlation raporland?."

G?vensiz:

- "Action 24/32 public veride counterfactual olarak daha iyi olurdu."
- "Secondary fleet ekonomik olarak kan?tland?."
- "No-reorder g?venli."
- "Global SOTA veya ger?ek d?nya optimumu kan?tland?."

## Kaynaklar

- [Open Bandit Pipeline docs](https://zr-obp.readthedocs.io/en/latest/)
- [OBP GitHub](https://github.com/st-tech/zr-obp)
- [SCOPE-RL docs](https://scope-rl.readthedocs.io/en/latest/)
- [SCOPE-RL GitHub](https://github.com/hakuhodo-technologies/scope-rl)
- [d3rlpy offline policy selection](https://d3rlpy.readthedocs.io/en/v2.8.1/tutorials/offline_policy_selection.html)
- [d3rlpy FQE reference](https://d3rlpy.readthedocs.io/en/v2.1.0/references/generated/d3rlpy.ope.FQE.html)
- [D4RL paper](https://arxiv.org/abs/2004.07219)
- [D4RL repository](https://github.com/Farama-Foundation/D4RL)
- [Amazon Last Mile Open Data Registry](https://registry.opendata.aws/amazon-last-mile-challenges/)

## Sonu?

Public replay adapter bu sprint i?in do?ru ara katmand?r. OPE tooling ara?t?rmas? ise gelecekte company-data trajectory/propensity paketi gelirse kullan?lacak yol haritas?d?r; public data tek ba??na OPE kan?t? de?ildir.
