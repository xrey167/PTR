# Recommended GitHub Repository Settings

The repository should protect `main` with a ruleset requiring the CI workflow, blocking force-push/deletion and requiring pull requests for non-maintainer contributions. Keep an owner/admin bypass for emergency research maintenance if desired.

Recommended repository metadata:
- description: Probabilistically Typed Reasoning — typed cognitive runtime and model architecture research
- topics: rust, llm, agents, reasoning, ai-systems, machine-learning, distributed-systems
- delete head branches after merge: enabled
- vulnerability reporting/security advisories: enabled

These settings are documented here because the current GitHub connector does not expose repository-administration mutations.
