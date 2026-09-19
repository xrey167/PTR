# ptr-ledger — Causal Ledger & Consensus Boundary

> **Role:** Records the authoritative ordered lifecycle of semantic changes and, in cluster mode, applies distributed consensus before state materialization.  
> **Maturity:** architecture + contract scaffold; production behavior must be proven by the linked experiments and component evaluations.

<!-- PTR:STATUS:BEGIN -->
## Current implementation status

> **Generated section.** Source of truth: [`component.toml`](component.toml) plus code-derived metrics from `src/`. Run `python3 scripts/update_component_docs.py --write` after editing implementation metadata. Do not hand-edit inside this block.

**Maturity:** `prototype`  
**Last reviewed:** 2026-09-19  
**Code footprint:** 7 Rust source files · 2642 nonblank source lines · 10 integration-test files · 57 `#[test]` markers

### Implemented now

- PTRLOG02 bounded SHA-256 record chain with checked header, sequence, payload and trailer
- Strict nonmutating open; independently anchored rollback checks and explicit partial-tail recovery
- Create-new checked-log restore and guarded explicit legacy inspection/migration
- Explicit RAII unlock prevents duplicate-descriptor retention across concurrent process spawn
- SemanticDeltaCommitted tag 8 preserves base/result revisions and opaque transaction bytes; legacy event tags are unchanged
- PTRANC01 protected anchor storage: HMAC-SHA256 under a host-held key, fixed-length record, atomic rename publication, monotonic epoch/index/digest and carried chain origin
- Retained-epoch witness rejects rollback of the anchor file itself; authenticity alone is documented as insufficient for freshness
- AcknowledgedLedger log-first append/acknowledge ordering with writer fencing when anchor publication fails
- RecoverableLog classifies every log/anchor split under one held lock and repairs only the tail per explicit TailPolicy
- integrity scan/encode/decode accept a trusted compaction floor so a log need not begin at index 1
- LogPaths floor-in-filename addressing: the protected anchor alone names the live log, so a cutover is unambiguous on both sides of its commit point
- RetentionPolicy/CompactionBarrier planning capped by snapshot coverage and blocked by lagging consumers or unresolved revocations
- Create-new compaction cutover with the anchor advance as commit point, revalidated plans and explicit orphan reclamation
- Erasure audit over live and superseded logs: byte-level presence search biased toward still-retained, with host-retained artifacts folded in explicitly
- Named permanent erasure boundaries (host-retained artifacts, storage residue, model-derived state) reported by every audit rather than as situational caveats
- Cross-process single-writer advisory lock retained for FileLedger handle lifetime
- Poisoned writer after ambiguous append failure; reopen/replay required before subsequent writes
- Lifecycle LedgerEvent enum
- CommittedEvent with CommitIndex
- Ledger trait and in-memory reference implementation, with an optional compaction-floor offset so restored records keep their committed indices
- CompactionBarrier scaffold
- Durable reference FileLedger with checked versioned frames, file synchronization and anchor-required tail recovery
- Feature-gated fail-rs injection points around record write/payload/fsync/memory-commit boundaries
- Feature-gated raft-engine 0.4.2 durable adapter stores ordered PTR ledger events with synchronous writes and reopen validation
- Feature-gated raft-rs 0.7 single-node consensus harness proposes and commits PTR LedgerEvents through RawNode

### Missing for the target architecture

- Hardware power-loss evidence; anchor key custody, rotation and hardware-backed sealing
- Multi-node raft-rs consensus adapter with transport, persistent Raft storage and membership changes
- fsync/durability modes and revocation barriers
- Runtime compacted materialized snapshot establishing snapshot_covers, so exact compacted-state reconstruction is not yet demonstrable end to end
- Distributed snapshot/compaction protocols and cross-node cutover
- Retention schedule/policy engine and durable snapshot lifecycle tracking; PTR does not enumerate or reclaim host-retained snapshots
- Storage-residue and model-derived-state erasure, so no all-state-deleted declaration is available

### Next milestones

- Define storage/consensus interfaces around existing Ledger contract
- Benchmark FileLedger against raft-engine, then connect raft-rs RawNode to raft-engine persistence and network transport
- Add a ptr-runtime compacted materialized snapshot that covers a floor and restores from a compacted log
- Expand L001 failpoint matrix to process-abort and durable-backend cases, then run L002 cluster recovery

### Linked experiments

- [L001](../../experiments/lifecycle/L001-revocation-crash/README.md) — `running`
- [L002](../../experiments/lifecycle/L002-raft-recovery/README.md) — `planned`
- [E004](../../experiments/system/E004-long-horizon/README.md) — `planned`

### Technology evaluations

- [consensus](../../evaluations/components/consensus/README.md) — `open`
- [ledger](../../evaluations/components/ledger/README.md) — `open`

### Decision records

- [ADR-0002-authority-hierarchy.md](../../research/decisions/ADR-0002-authority-hierarchy.md)
- [ADR-0009-consensus-ledger-state-separation.md](../../research/decisions/ADR-0009-consensus-ledger-state-separation.md)

### Current automated checks

- Every single-bit mutation of the reference log, independent hashlib golden vector, ordering and hostile-length rejection
- Every partial-frame cut, wrong anchor, complete suffix loss and rehashed alternative-history rejection
- Create-new destination safety, guarded legacy migration and duplicate-descriptor unlock regression
- RFC 4231 HMAC-SHA256 known answers, constant-time comparison and redacted key Debug output
- Every single-bit mutation and every length variation of the 168-byte anchor record rejected
- Every log/anchor split outcome: aligned, unacknowledged, lost suffix, rehashed divergence, foreign origin and base mismatch
- Stale-witness anchor rollback, interrupted publication and fenced writer after failed acknowledgment
- Retention bounds, barrier blocking and snapshot-coverage capping of the proposed floor
- Cutover interrupted on both sides of its commit point, orphan reclamation and exact retained-suffix reconstruction
- Stale, non-advancing, above-tail and wrong-digest plans refused without mutation; destination never overwritten
- Logical deletion asserted to leave history intact; erasure requires both the raised floor and orphan reclamation
- Host-retained snapshot defeats erasure once folded in; destroying the anchor key removes verifiability only
- FileLedger all-event reopen and partial-tail crash recovery tests
- FileLedger strict open never truncates; explicit recovery requires an independent matching prefix anchor
- fail-rs panic-after-length-prefix recovery test
- raft-engine durable append/reopen ordering integration test
- raft-rs single-node proposal/commit ordering integration test
- workspace fmt/check/test/clippy

<!-- PTR:STATUS:END -->

## Position in PTR

```mermaid
flowchart LR
    A["Validated proposal"] --> B["ptr-ledger\nCausal Ledger & Consensus Boundary"]
    B --> C["Committed causal history"]
    B -. "contracts" .-> T["ptr-types"]
    B -. "telemetry" .-> O["ptr-observe"]
```

Dedicated diagram source: [`docs/diagrams/components/ptr-ledger.mmd`](../../docs/diagrams/components/ptr-ledger.mmd)

**Upstream:** ptr-verifier, ptr-security  
**Downstream:** ptr-state, ptr-events

## Mission

Records the authoritative ordered lifecycle of semantic changes and, in cluster mode, applies distributed consensus before state materialization.

PTR keeps this responsibility in its own crate so the semantics remain stable even when an external library or implementation is replaced.

## Responsibilities

- ledger event model
- commit index
- revocation barriers
- snapshots/compaction barriers
- consensus adapter
- durable log adapter

## Explicit non-responsibilities

- query-friendly state
- search indexes
- network transport implementation

## Data flow

| Direction | Contract |
|---|---|
| Input | Validated semantic proposals and lifecycle events |
| Output | CommittedEvent sequence and durable snapshots |
| Failure | Explicit typed error / rejected state; no silent fallback that changes semantics |
| Observability | Standard PTR tracing fields and a stable component span |

## Technical approach

- raft-rs candidate for consensus
- raft-engine candidate for durable Raft log
- fail-rs for adversarial failure injection

External projects are **candidates**, not architectural authority. The PTR-owned traits and domain types must remain usable with a replacement backend.

## Core invariants

1. Uncommitted events never become authoritative.
2. Revoked generations cannot resurrect after restart.
3. Compaction requires snapshot, consumer and generation safety barriers.

These invariants should be executable wherever possible through unit, property, lifecycle or chaos tests.

## Failure model

The component must fail closed for semantic or effect-safety violations. Infrastructure failures should surface as typed errors that preserve request, revision, generation and provenance context. Retries must be idempotent whenever the operation may cross a process or network boundary.

## Performance model

Measure before optimizing. Benchmarks should record at least latency distribution, throughput, allocations/resident memory, queue depth or working-set size where relevant, and the cost of verification. Performance optimizations may not bypass generation, revision, capability or evidence checks.

## Security and privacy

- Treat external inputs and backend outputs as untrusted until validated.
- Do not put raw secrets/private evidence into generic tracing or inspection.
- Preserve provenance on every promotion from raw/possible evidence to stronger semantic state.
- External effects pass through `ptr-security` even if this component already performed local validation.

## Technology evaluation

- [consensus](../../evaluations/components/consensus/README.md)
- [ledger](../../evaluations/components/ledger/README.md)

A new candidate should be added with a reproducible benchmark and failure-semantics analysis rather than replacing the default ad hoc.

## Tests required before production use

- Contract/unit tests for all domain transitions.
- Invalid, stale-generation and stale-revision cases.
- Cancellation/retry behavior.
- Property tests for invariants where practical.
- Cross-backend equivalence if more than one backend exists.
- Observability and redaction checks.

## Related architecture

- [System architecture](../../docs/architecture/00-system.md)
- [Technical architecture](../../docs/TECHNICAL_ARCHITECTURE.md)
- [Component contracts](../../docs/COMPONENT_CONTRACTS.md)
- [Global invariants](../../docs/INVARIANTS.md)



## P0.2 semantic journal integration

[Durable semantic-state contract](../../docs/architecture/22-durable-semantic-state.md)
records the new code/codec, ownership and replay boundaries. Publication follows
successful journal append. Typed Pod bytes and source identity participate in
semantic revisions. Logical removals do not erase log history; neural checkpoints
and authenticated framing remain separate gates. Execution evidence is in the PR.

## Persistence contract update

See [P0.3 checked records and replay-backed recovery snapshots](../../docs/architecture/23-persistence-integrity.md)
for strict reopen, explicit legacy migration, independent anchors, create-new
restore and format/API compatibility. Old automatic crash-tail repair is replaced
by explicit anchored recovery. No authority or neural checkpoint is restored.
