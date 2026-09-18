# Security Policy

PTR is an experimental agent/model runtime and should not be treated as a security boundary until the relevant invariants are implemented and tested.

## Reporting a vulnerability

Please do not open a public issue for a vulnerability that could enable unauthorized code execution, secret disclosure, capability bypass, stale-generation resurrection, prompt/context injection with real effects, or consensus/state corruption.

Use GitHub's private security-advisory/reporting channel for this repository when available. Include reproduction steps, affected commit, impact, and whether the issue crosses the hard action boundary.

## Supported versions

Until the first tagged release, only the current `main` branch is maintained.

## High-value security invariants

- untrusted context cannot grant capabilities;
- learned scores cannot override hard denies;
- revoked generations cannot become usable after restart/cache/index lag;
- secrets are redacted from generic telemetry and inspection;
- external effects require explicit capability, permission and freshness checks.
