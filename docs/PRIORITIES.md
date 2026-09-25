# PTR Priority Ledger

This is the implementation order after the architecture-scaffold phase. Status values are evidence-oriented rather than aspirational.

| Priority | Work item | Status |
|---|---|---|
| P0 | Repository licensing, lockfiles, MSRV/stable/Linux/Windows CI | verification infrastructure implemented; exact-commit results in security/P0 review; release license obligations remain |
| P0 | Component docs-as-code, local config.toml and tests/ ownership | done |
| P0 | Central ptr-runtime orchestrator | prototype |
| P0 | Hard revision/generation/capability effect boundary | prototype — typed ptr-security decisions now own freshness + capability/effect checks, all freshness/capability/effect gates are mandatory for hard effects; diagnostic receipts are not execution tokens; P0.1 adds opaque host-authenticated sessions, exact capsule/project grants, registered verifiers/executors and a single-use synchronous gateway; PR #21 adds peer admission by host policy, project-scoped Pods, a journaled execution audit with durable fencing and at-most-once keys, and an execution wire on ALPN_EXEC where the peer is the authenticated connection; a signed receipt, a settlement channel for detached work and a journaled admission policy remain |
| P0 | Durable single-node ledger + crash-tail/replay safety | reference path implemented — P0.1 scoped execution, P0.2 semantic replay and P0.3 checked frames/replay-backed snapshot restore; production anchors, compaction, neural checkpoint admission and cluster durability remain release gates |
| P0 | Executable experiment/evaluation runners | implemented — declared argv execution, immutable success/failure evidence and subprocess tests exist; benchmark coverage remains incomplete |
| P0 | Reproducible Python training-run manifest | prototype |
| P1 | Prost-generated typed wire contracts | prototype |
| P1 | Typed file → environment → CLI configuration precedence | done for core settings |
| P1 | HTTP server + TypeScript client contract | prototype |
| P1 | Strong RAG / matched plain-model baselines | in progress — execution gates prevent unpinned claims |
| P1 | Component evaluations with measured evidence | in progress |
| P1 | Supply-chain audit/deny, SBOM, provenance attestations | every owned workspace covered; scan outcome and release packaging are separate gates |
| P1 | Dataset cards and contamination governance | prototype |
| P1 | Burn PTR-A0 typed neural path | prototype — the preregistered A0 mechanism ablation study v1 has run on the synthetic operator-routing v1 ([results](../research/falsification/A0-ablations-v1/RESULTS.md)); A0-internal evidence only, not the matched baseline comparison: typed slot content helped at 1500 steps (M001-primary SUPPORTS), but the raw-token arm nearly caught up at 4000 steps (M001-secondary INCONCLUSIVE) |
| P1 | Epistemic/validity/provenance neural metadata + typed attention bias | prototype — A0-internal evidence only: a benefit of 2 points or more from the rank-1 typed attention bias is excluded in A0 (M002-necessity FALSIFIES); the admission mask's own effect is not established, because the only comparison of it set a trained arm against an undertrained one |
| P1 | Latent recurrent refinement + operator router | prototype — A0-internal evidence only: the per-slot nonlinearity is used (M003-nonlinearity SUPPORTS), a second tied refinement step adds nothing measurable (M003-depth FALSIFIES), and the frozen-router control is INCONCLUSIVE |
| P1 | Multi-step model resume after verified Pod observation | prototype — bounded verified observation resume implemented |
| P1 | L001 harder process/failpoint crash schedules | in progress — fail-rs + 250 real child-process abort cases executed |
| P2 | raft-engine/raft-rs, Turso, Iroh production adapters | prototypes — raft-rs, raft-engine, Turso and direct Iroh adapters integrated; PR #21 adds durable Raft state on disk, a three-member group over ALPN_RAFT and snapshot transfer; durable cross-node leadership fencing, negotiated membership, an accept loop and session reuse remain |
| P2 | Full model training + matched M001–M005 ablations | planned — the A0 ablation study does not count toward this; the baseline comparisons need the plain model baseline (owner decision O2 in [the recommendations](RECOMMENDATIONS_20260924.md)) |
| P2 | GitHub branch ruleset/admin metadata | setup scripts/templates ready; admin application remains manual |
| P2 | Stable release/multi-platform packaging | planned |

The next scientific gate is not a larger model. It is a matched, reproducible M001/M002/M003/M004 experiment where the new neural mechanisms are compared against controlled ablations.

The A0 mechanism ablation study v1 is the A0-internal half of that: controlled ablations of A0's own mechanisms, preregistered and run under the M001–M004 manifests with separate `a0_*` entrypoints. Its verdicts are A0-internal evidence only. They hold for one block at d_model 48 on a synthetic benchmark, and they are not the matched plain-model comparison, so M001–M004 stay `planned`.

See [the security/P0 review](SECURITY_P0_REVIEW_20260919.md) for concrete repairs, remaining architecture gates and the limits of green CI.

Current bounded corrections: [P0.1 scoped execution](architecture/21-scoped-execution.md),
[P0.2 semantic replay](architecture/22-durable-semantic-state.md), and
[P0.3 record integrity/recovery snapshots](architecture/23-persistence-integrity.md).

After exact-commit validation of this reference path, return to the existing
cognitive development sequence: minimum semantic values and versioned codebook,
then model/data integration and controlled training. Do not grow infrastructure
indefinitely before testing that architecture. Unimplemented network, cluster,
compaction, secure-erasure and deployment trust requirements were carried from
issue #15 into #20, which PR #21 closes; what survives that pull request is
recorded in `OPEN_ITEMS_PLAN_20260921.md` — among it two clauses of #15's own gate
sentences that no checkbox ever covered, connecting actual semantic payloads to the
model and durable cross-node leadership fencing. They still block their respective
production claims. They do not turn a
local CPU/model experiment into a production release.
