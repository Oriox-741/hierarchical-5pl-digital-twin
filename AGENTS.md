# AGENTS.md

Future agents working in this sanitized repository must follow these rules:

- Never mutate protected artifacts: registry, production models, checkpoints, baselines, databases, raw datasets, or existing evaluation outputs.
- Never run training, offline evaluation, long-run gates, registration, activation, production promotion, baseline updates, dataset downloads, tunnels, or dashboard servers without explicit approval.
- Run safe checks first: Python compile checks and focused tests only after reviewing whether they trigger heavy workflows.
- Keep public replay framed as descriptive proxy evidence, not causal OPE.
- Do not claim live TMS/WMS/ERP deployment, company-data validation, global superiority, or global SOTA.
- Preserve the thesis title: Multi-Layered Digital Twin Framework for Autonomous Supply Chain Orchestration.
- Keep secrets, tokens, local credential caches, and private data out of Git.
