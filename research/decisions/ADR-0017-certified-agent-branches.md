# ADR-0017 — Agent Branches Are Certified Candidates Merged Through Verified Deltas

## Status
Accepted for the prototype (`ptr-branch`). Throughput and lost-update claims are gated on S003; the arbiter's risk claims on F003.

## Decision
An agent branch is a private overlay on one immutable semantic snapshot that declares its dependencies: a digest of every value it read (the canonical journal bytes neural-state admission digests), a digest of every prefix it scanned (phantoms), and every lifecycle generation it relied on. Certification against a newer snapshot either refuses or yields one ordinary `SemanticDelta` plus the revision it was certified against; counters and sets merge commutatively by rebasing onto the target value. The plan is committed through `Runtime::apply_verified_semantic_delta`, which verifies the prepared post-state before publication. Using that path is the caller's obligation until a plan can be consumed only by a verifying entry point: the runtime's unverified `apply_semantic_delta` is public. Triage is verifier-bounded: only a `Pass` at full-semantic or deterministic level with no hard finding is eligible for auto-proposal; a uniform calibration slice of eligible branches is escalated for adjudication, and the auto-propose threshold is chosen on a fixed grid with a Clopper-Pearson or conformal risk bound. Logged propensities make off-policy evaluation possible.

## Consequences
A branch never writes state and never outranks verification (INVARIANT 11). Raw request text and Pod outputs are reserved namespaces. Triage can move a branch towards more human review, never past verification. Domain invariants over rebased commutative operations are the verifier's to check.
