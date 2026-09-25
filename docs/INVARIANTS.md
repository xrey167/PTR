# PTR Global Invariants

These are architecture-level requirements. Implementations may strengthen them but not silently weaken them.

1. **Raw preservation:** typed semantics never erase the source representation.
2. **Revision isolation:** a reasoning run is bound to one immutable semantic revision unless explicitly resumed on a new snapshot.
3. **Generation safety:** a revoked/superseded generation cannot become usable through cache, restart, index lag or replay.
4. **Authority:** uncommitted state is never exposed as authoritative semantic state.
5. **Derived search:** search/index/cache data cannot self-promote to truth.
6. **Evidence promotion:** retrieval candidate → source resolution → observation → verification → stronger semantic state.
7. **Hard effect boundary:** external effects require capability, permission, current revision/generation and required verification.
8. **Backpressure:** core queues are bounded; saturation is explicit.
9. **Single-owner isolate state:** shared mutation is not the default execution model.
10. **Provider independence:** external libraries do not define PTR semantics.
11. **Verifier precedence:** deterministic contradiction cannot be overridden by a learned positive score.
12. **Unknown is valid:** the system is allowed to preserve uncertainty.
13. **Secret redaction:** generic inspection/telemetry never exposes secret/private values by default.
14. **Replayability:** authoritative state can be rebuilt from committed history + validated snapshot.
15. **Falsifiability:** model/architecture claims require baseline, ablation and failure criteria.
16. **Projection fidelity:** a projection advances only by records verified against the ledger's own anchors; a projection ahead of or diverging from the ledger is refused and rebuilt, never reconciled.
17. **Revocable derivation:** derived state that folds lifecycle-managed inputs (fast memories, adapters, labels) names the input generations it depends on, so revoking an input identifies its influence and, where the fold is deterministic, removes it exactly.

