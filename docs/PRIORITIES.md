# PTR Priority Ledger

This is the implementation order after the architecture-scaffold phase. Status values are evidence-oriented rather than aspirational.

| Priority | Work item | Status |
|---|---|---|
| P0 | Repository licensing, lockfiles, MSRV/stable/Linux/Windows CI | done |
| P0 | Component docs-as-code, local config.toml and tests/ ownership | done |
| P0 | Central ptr-runtime orchestrator | prototype |
| P0 | Hard revision/generation/capability effect boundary | prototype — typed ptr-security decisions now own freshness + capability/effect checks and emit allow receipts; principal/resource scopes and verifier requirements remain |
| P0 | Durable single-node ledger + crash-tail/replay safety | done for reference FileLedger, including runtime reopen/replay; cluster durability remains separate |
| P0 | Executable experiment/evaluation runners | prototype |
| P0 | Reproducible Python training-run manifest | prototype |
| P1 | Prost-generated typed wire contracts | prototype |
| P1 | Typed file → environment → CLI configuration precedence | done for core settings |
| P1 | HTTP server + TypeScript client contract | prototype |
| P1 | Strong RAG / matched plain-model baselines | in progress — execution gates prevent unpinned claims |
| P1 | Component evaluations with measured evidence | in progress |
| P1 | Supply-chain audit/deny, SBOM, provenance attestations | done |
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