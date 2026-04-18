# Security Policy

This is a local benchmark harness. It runs unprivileged containers against models on disk and makes HTTP calls to `127.0.0.1`. There is no network service, no multi-tenant surface, and no persistent credential store.

## Supported versions

Only `main` is supported. Fixes land on `main`; there is no release branch.

## Reporting a vulnerability

**Do not open a public issue** for security-sensitive findings. Report privately via GitHub's Security tab ("Report a vulnerability"):

<https://github.com/Rethunk-AI/bakeoff/security/advisories/new>

Please include:

- Affected commit SHA
- Reproduction steps (exact command, `config.yaml` relevant keys)
- Observed vs. expected behavior
- Impact assessment (local-only, requires user interaction, etc.)

## In scope

- Command injection or path traversal via `config.yaml` or CLI flags
- Container escape or privilege escalation from `bin/serve.sh`
- Arbitrary code execution via a malicious GGUF path or model alias
- SSRF or request forgery from the HTTP client in `bench/clients.py`
- Credential leakage (logs, results artifacts)

## Out of scope

- Issues requiring root on the benchmark host
- Bugs in `llama.cpp`, `podman`, or upstream container images (report those upstream)
- Denial of service caused by loading an oversized model (this is a resource-sizing question, not a vulnerability)
- Findings in third-party dependencies without a demonstrable exploit path through this repo
