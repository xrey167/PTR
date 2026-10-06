# PTR Priority Ledger

This is the implementation order after the architecture-scaffold phase. Status values are evidence-oriented rather than aspirational.

| Priority | Work item | Status |
|---|---|---|
| P0 | Repository licensing, lockfiles, MSRV/stable/Linux/Windows CI | verification infrastructure implemented; exact-commit results in security/P0 review; release license obligations remain |
| P0 | Component docs-as-code, local config.toml and tests/ ownership | done |
| P0 | Central ptr-runtime orchestrator | prototype |
| P0 | Hard revision/generation/capability effect boundary | prototype — typed ptr-security decisions now own freshness + capability/effect checks, all freshness/capability/effect gates are mandatory for hard effects; diagnostic receipts are not execution tokens; P0.1 adds opaque host-authenticated sessions, exact capsule/project grants, registered verifiers/executors and a single-use synchronous gateway; PR #21 adds peer admission by host policy, project-scoped Pods, a journaled execution audit with durable fencing and at-most-once keys, and an execution wire on ALPN_EXEC where the peer is the authenticated connection; a signed receipt, a settlement channel for detached work and a journaled admission policy remain |
| P0 | Durable single-node ledger + crash-tail/replay safety | reference path implemented — P0.1 scoped execution, P0.2 semantic replay and P0.3 checked frames/replay-backed snapshot restore; protected anchors ([doc 24](architecture/24-protected-anchors.md)), compaction (`ptr-runtime`), neural-state admission ([doc 27](architecture/27-neural-state-admission.md)) and a durable three-member raft group ([doc 31](architecture/31-cluster-integrity.md)) now exist at library level; deploying anchors in production and the items each of those documents lists under what it does not close remain release gates |
| P0 | Executable experiment/evaluation runners | implemented — declared argv execution, immutable success/failure evidence and subprocess tests exist; benchmark coverage remains incomplete |
| P0 | Reproducible Python training-run manifest | prototype |
| P1 | Prost-generated typed wire contracts | prototype |
| P1 | Typed file → environment → CLI configuration precedence | done for core settings |
| P1 | HTTP server + TypeScript client contract | prototype |
| P1 | Strong RAG / matched plain-model baselines | in progress — execution gates prevent unpinned claims |
| P1 | Component evaluations with measured evidence | in progress |
| P1 | Supply-chain audit/deny, SBOM, provenance attestations | every owned workspace covered; scan outcome and release packaging are separate gates |
| P1 | Dataset cards and contamination governance | prototype |
| P1 | Burn PTR-A0 typed neural path | Corrected 2026-09-29: gated A0 mechanism hypotheses are INCONCLUSIVE because G6 lacks commit-bound correctness evidence. Historical raw scores are descriptive only; admission fixes require a new frozen study. See [results](../research/falsification/A0-ablations-v1/RESULTS.md). |
| P1 | Epistemic/validity/provenance neural metadata + typed attention bias | Corrected 2026-09-29: gated A0 mechanism hypotheses are INCONCLUSIVE because G6 lacks commit-bound correctness evidence. Historical raw scores are descriptive only; admission fixes require a new frozen study. See [results](../research/falsification/A0-ablations-v1/RESULTS.md). |
| P1 | Latent recurrent refinement + operator router | Corrected 2026-09-29: gated A0 mechanism hypotheses are INCONCLUSIVE because G6 lacks commit-bound correctness evidence. Historical raw scores are descriptive only; admission fixes require a new frozen study. See [results](../research/falsification/A0-ablations-v1/RESULTS.md). |
| P1 | Multi-step model resume after verified Pod observation | prototype — bounded verified observation resume implemented |
| P1 | L001 harder process/failpoint crash schedules | in progress — fail-rs + 250 real child-process abort cases executed |
| P1 | Agentic substrate (doc 35): PostgreSQL projection substrate, certified branches, revocable fast memory, adapter lineage, weak supervision | prototype — anchor-verified projection, live-only derived caches, hybrid retrieval and working-state storage tested against a real server in CI; branches merge only through verified deltas; experiments L004 and L003 completed on PostgreSQL 18 (five seeds each, mutation-checked); Q003, S003, F003, M008, R004, F002 and E005 planned; TLS, roles, a business-table effect applier and lineage/labeling adapters remain |
| P2 | raft-engine/raft-rs, Turso, Iroh production adapters | prototypes — raft-rs, raft-engine, Turso and direct Iroh adapters integrated; PR #21 adds durable Raft state on disk, a three-member group over ALPN_RAFT and snapshot transfer; the fencing token (`PTR_RAFT_FENCED`, a read before each write, not atomic with it) and single-voter membership changes exist and are bounded as [doc 31](architecture/31-cluster-integrity.md) states; an accept loop, discovery, session reuse and an independently retained snapshot anchor remain |
| P2 | Full model training + matched M001–M005 ablations | planned — the A0 ablation study does not count toward this; the baseline comparisons need the plain model baseline (owner decision O2 in [the recommendations](RECOMMENDATIONS_20260924.md); decided 2026-10-06, ADR-0021: A0-scale baseline is the pinned A0-matched plain model, the pretrained LM baseline is deferred to the GPU phase, M005 is out of scope at the A0 tier) |
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
model and durable cross-node leadership fencing. Connecting semantic payloads to the
model is still open as an end-to-end path; durable fencing is implemented as a fencing token
checked before each write, which is not atomic with the write (a window
`architecture/31-cluster-integrity.md` states), so it is not closed. What
remains open still blocks the respective production claims. They do not turn a
local CPU/model experiment into a production release.
