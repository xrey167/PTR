# PTR Priority Ledger

This is the implementation order after the architecture-scaffold phase. Status values are evidence-oriented rather than aspirational.

| Priority | Work item | Status |
|---|---|---|
| P0 | Repository licensing, lockfiles, MSRV/stable/Linux/Windows CI | done |
| P0 | Component docs-as-code, local config.toml and tests/ ownership | done |
| P0 | Central ptr-runtime orchestrator | prototype |
| P0 | Hard revision/generation/capability effect boundary | prototype |
| P0 | Durable single-node ledger + crash-tail/replay safety | prototype |
| P0 | Executable experiment/evaluation runners | prototype |
| P0 | Reproducible Python training-run manifest | prototype |
| P1 | Prost-generated typed wire contracts | prototype |
| P1 | Typed file → environment → CLI configuration precedence | done for core settings |
| P1 | HTTP server + TypeScript client contract | prototype |
| P1 | Strong RAG / matched plain-model baselines | in progress |
| P1 | Component evaluations with measured evidence | in progress |
| P1 | Supply-chain audit/deny, SBOM, provenance attestations | done |
| P1 | Dataset cards and contamination governance | prototype |
| P1 | Burn PTR-A0 typed neural path | prototype |
| P1 | Epistemic/validity/provenance neural metadata + typed attention bias | prototype |
| P1 | Latent recurrent refinement + operator router | prototype |
| P1 | Multi-step model resume after verified Pod observation | next |
| P1 | L001 harder process/failpoint crash schedules | next |
| P2 | raft-engine/raft-rs, Turso, Iroh production adapters | planned |
| P2 | Full model training + matched M001–M005 ablations | planned |
| P2 | GitHub branch ruleset/admin metadata | manual repository setting |
| P2 | Stable release/multi-platform packaging | planned |

The next scientific gate is not a larger model. It is a matched, reproducible M001/M002/M003/M004 experiment where the new neural mechanisms are compared against controlled ablations.
