# Security Policy

## Reporting Security Issues

Report suspected vulnerabilities privately to the repository owner. Do not open public issues containing secrets, credentials, private data, or exploit details.

## Sanitization Boundary

This repository should not contain credentials, tokens, environment files, private company data, raw datasets, production registry state, production model binaries, checkpoints, baselines, databases, DevSpace material, tunnel configuration, or browser/session/auth files.

## Secret Handling

Never commit API keys, GitHub tokens, OpenAI keys, passwords, private keys, OAuth secrets, owner tokens, or local credential caches. If a secret is accidentally committed, rotate it immediately and rewrite history before any public release.
