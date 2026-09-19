# PTR Priority Ledger

This is the implementation order after the architecture-scaffold phase. Status values are evidence-oriented rather than aspirational.

| Priority | Work item | Status |
|---|---|---|
| P0 | Repository licensing, lockfiles, MSRV/stable/Linux/Windows CI | verification infrastructure implemented; exact-commit results in security/P0 review; release license obligations remain |
| P0 | Component docs-as-code, local config.toml and tests/ ownership | done |
| P0 | Central ptr-runtime orchestrator | prototype |
| P0 | Hard revision/generation/capability effect boundary | prototype — typed ptr-security decisions now own freshness + capability/effect checks, all freshness/capability/effect gates are mandatory for hard effects; diagnostic receipts are not execution tokens; P0.1 adds opaque host-authenticated sessions, exact capsule/project grants, registered verifiers/executors and a single-use synchronous gateway; network authentication, durable idempotency and scoped Pod access remain |
| P0 | Durable single-node ledger + crash-tail/replay safety | prototype — single-writer/poisoned append/lifecycle replay protections; semantic payload/dependency replay implemented in the reference path; checksummed framing, real snapshots, neural checkpoint admission and cluster durability remain open |
| P0 | Executable experiment/evaluation runners | implemented — declared argv execution, immutable success/failure evidence and subprocess tests exist; benchmark coverage remains incomplete |
| P0 | Reproducible Python training-run manifest | prototype |
| P1 | Prost-generated typed wire contracts | prototype |
| P1 | Typed file → environment → CLI configuration precedence | done for core settings |
| P1 | HTTP server + TypeScript client contract | prototype |
| P1 | Strong RAG / matched plain-model baselines | in progress — execution gates prevent unpinned claims |
| P1 | Component evaluations with measured evidence | in progress |
| P1 | Supply-chain audit/deny, SBOM, provenance attestations | every owned workspace covered; scan outcome and release packaging are separate gates |
| P1 | Dataset cards and contamination governance | prototype |
| P1 | Burn PTR-A0 typed neural path | prototype |
| P1 | Epistemic/validity/provenance neural metadata + typed attention bias | prototype |
| P1 | Latent recurrent refinement + operator router | prototype |
| P1 | Multi-step model resume after verified Pod observation | prototype — bounded verified observation resume implemented |
| P1 | L001 harder process/failpoint crash schedules | in progress — fail-rs + 250 real child-process abort cases executed |
| P2 | raft-engine/raft-rs, Turso, Iroh production adapters | prototypes — raft-rs, raft-engine, Turso and direct Iroh adapters integrated; multi-node/session hardening remains |
| P2 | Full model training + matched M001–M005 ablations | planned |
| P2 | GitHub branch ruleset/admin metadata | setup scripts/templates ready; admin application remains manual |
| P2 | Stable release/multi-platform packaging | planned |

The next scientific gate is not a larger model. It is a matched, reproducible M001/M002/M003/M004 experiment where the new neural mechanisms are compared against controlled ablations.

See [the security/P0 review](SECURITY_P0_REVIEW_20260919.md) for concrete repairs, remaining architecture gates and the limits of green CI.

Current bounded component increment: [P0.2 durable semantic transactions](architecture/22-durable-semantic-state.md).
This builds on [P0.1 scoped synchronous execution](architecture/21-scoped-execution.md)
and closes the in-memory-only ingestion and type-only Pod-output data path.

The next persistence gate is record integrity and verified snapshot/checkpoint
admission. Scoped network/Pod access and durable effect reconciliation remain
separate execution gates. Do not mark issue #15 complete from a successful local
mechanism or a green dependency scan.
