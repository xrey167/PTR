# Repository Status

PTR is a **compiling architecture/research prototype**, not yet a complete model/runtime product and not yet evidence of superiority over strong baselines.

## Implemented and validated

- 33 Rust workspace crates plus four binaries (`ptr-bench`, `ptr-worker`, `ptrctl`, `ptrd`), including typed file/environment/CLI configuration and a central `ptr-runtime` orchestrator; [docs/components/STATUS.md](docs/components/STATUS.md) is the generated inventory and the authority for the count
- foundational domain types, semantic revision/snapshot prototype and dependency invalidation
- bounded-mailbox/isolate contracts and Pod lease typestate
- typed model → semantic Pod registry → Pod → verifier → SemDB observation loop for Pure/Read cognitive Pods
- typed ActionIR authorization decisions in `ptr-security`, with revision/generation freshness, independent effect authority, capability/effect denials and allow receipts (`ptr-runtime`'s diagnostic `authorize_action` returns only `Result<(), _>`, so no runtime path retains a receipt yet)
- durable single-node reference `FileLedger` with PTRLOG02 integrity framing, strict reopen and independently anchored explicit crash-tail recovery
- `ptr-runtime` durable standalone mode can open/replay `FileLedger` and reconstruct lifecycle/materialized state across restart
- feature-gated raft-engine durable log, raft-rs consensus (single-node, and a three-member group with elections, partitions and snapshot transfer tested on loopback, composed over `ALPN_RAFT` in `ptr-cluster`), Turso materialized-state and direct Iroh transport adapters with dedicated CI jobs
- monotonic materialization with duplicate/out-of-order/gap rejection
- explicit retrieval evidence-promotion stages that reject stale generations and direct promotion to Known
- Prost-generated protobuf schemas with validated PodCall → domain conversion
- per-component `component.toml`, local `config.toml`, README status blocks and tests directories
- every first-class workspace area has a local `config.toml` and a matching tests directory, checked by `scripts/check_repo.py`
- Cargo and uv lockfiles; Rust 1.99 MSRV plus current-stable Ubuntu/Windows CI
- dataset cards, experiment/evaluation validators, setup scripts, devcontainer/Docker and release scaffolding
- executable experiment and component-evaluation runners with declared no-shell commands and immutable success/failure process evidence
- supply-chain checks with cargo-audit/cargo-deny and Dependabot configuration
- `ptrctl doctor` repository/configuration diagnostics
- CI-tested dependency-free TypeScript SDK contract and static documentation landing page
- release packaging with SHA-256 checksums, CycloneDX SBOM generation and GitHub/Sigstore provenance/SBOM attestations
- Pod-MoE / KV-reuse contracts, all prototype-level:
  - `ptr-lineage` AIMER expert scoring (`crates/ptr-lineage/src/aimer.rs`: `aimer_score`, `rank_experts_for_pruning`)
  - `ptr-pods` expert-proposal contracts and lifecycle guards (`crates/ptr-pods/src/expert.rs`): `ExpertProposal`, `StructuralOp`, `ExpertOperationRequest`, `ExpertPodSpec`, `ExpertTarget`, `ExpertState` and `ExpertLevel`; a proposal is inert data; its acceptance path is `PtrRuntime::admit_expert_proposal` in `crates/ptr-runtime/src/expert.rs` (grant-bound proposer, evidence bound to proposal, operation, artifact, base model and eval manifest, compare-and-set against the registry, commit-time identity and grant revalidation, ledger event `ExpertStructuralCommitted`), with the runtime fence path (`ExpertFenceCommitted`: fence, escalate, gated readmit, revocation-driven retirement) beside it; it is a library path, not wired into ptrd
  - the `ptr-router` `Router` trait (`crates/ptr-router/src/router.rs`), whose distribution over pods is validated against the registry and recorded with the sampled pod's quantized probability as propensity
  - the PyO3 binding `bindings/ptr-py`, built and tested by `make py-bindings`; CI does not build it
  - the Python `trl-moe` backend in `training/src/ptr_training/moe/`, with configs `training/configs/stage-moe-smoke.toml` and `training/configs/stage-moe-olmoe.toml`

## Executed evidence so far

- **L001 is running, not completed.** The reference in-process crash-tail slice executed 500 cases across five seeds with 0 stale-generation false accepts, 0 recovery-consistency failures and 0 tail-trim failures. A second real child-process abort slice executed 250 cases across five seeds with 0 stale-generation false accepts, 0 recovery failures, 0 tail-trim failures and 0 unexpected child exits. Feature-gated fail-rs fault injection also runs in CI. Production raft-engine, leader-change and network-partition coverage is still missing.
- Internal component smoke evidence exists for the custom SemDB and PTR isolate/mailbox path; both candidates remain `evaluating`, not selected.
- A runnable BM25 + caller-supplied dense-vector RRF reference baseline exists, but no claim against strong RAG is valid until a pinned embedding/reranking baseline is executed.
- **R009 layer-shared-lora-kv and R010 moe-expert-kv-revocation completed (preregistered at `r009-prereg` / `r010-prereg`, one freeze commit, 5 seeds each 17/29/43/71/101).** Every seed held every coverage and falsifier rule of each frozen `criteria.toml`. R009: per-layer KV sharing between base and LoRA requests matched the predicted layer boundary for QKV (f/L), Q-only and MLP-only ((f+1)/L) and aLoRA (pre-invocation rows of the boundary page physically shared), bit-identical to full adapter prefill, with no cross-adapter, old-generation or post-revocation hit across the 40-cell invalidation matrix; restarts re-resolve to the live binding. R010: on a tiny dropless top-2 MoE, revoking, superseding or fencing expert (j,i) invalidated exactly the layers above j, a fenced expert's old-epoch entries stayed dead after gated readmission, batch-variant routing was tagged Approximate and never reached the Exact catalog, across the 44-cell matrix. Scope: CPU reference path, tiny seeded random model, KV-memory sharing only (no prefill-FLOPs claim), no engine, no GPU. Both archived results carry a staleness marker only because each experiment's records count as provenance of the other. Pre-freeze dry runs found and led to fixing three runtime defects (restored per-layer entries losing their layer scope, expert successors without fence epochs, no backend event hook).
- **R012 approximate-taint-containment completed (preregistered at `r012-prereg`, freeze commit 2b108667, 5 seeds 17/29/43/71/101).** Every seed held every coverage and falsifier rule of the frozen `criteria.toml`. Per-seed counters (identical on all seeds): 18 action admission attempts, 21 SemDB promotion attempts, 3 candidate path attempts, 3 tainted candidate reload attempts, 26 Pod result promotion attempts, 10 misclassified-Raw attempts, 5 cases of Approximate output reintroduced as Raw, 3 exact echo cases, 1 fresh exact recomputation control and 3 candidate taint hops; all violation counters were 0 (approximate_run_actions_admitted, approximate_semdb_promotions, approximate_candidate_exact_admissions, taint_lost, approximate_pod_result_promotions_accepted, misclassified_raw_admissions, laundered_exact_admissions, taint_cleared_by_echo) and all three mismatch counters were 0. Mechanism: ledger event `KvTaintCommitted` (tag 29), a persistent registry bounded at 65,536 entries, taint-first ordering, derivation hops, a Raw rule, clearing only by cold exact recomputation, and Pod/SemDB/merge containment in `ptr-runtime`. Scope and stated limits: stub Approximate producer (not a real approximate engine); exact whole-segment digest only, so paraphrase, substring and whitespace variants are not detected and honest identical text is tainted; the backend's reuse-mode report is trusted; the catalog guard is structural (Exact-only); the Request subject is never cleared; Pod-side taint is permanent; cleared tombstones grow; Clear is forgeable under a trusted-ledger model; CPU reference path on a tiny seeded model; production loops (run_resumable_*, ptrd) do not route through the taint API yet. A pre-freeze invariant review of the runtime and a dry run of the harness found and led to fixing two real gaps (plain-run output laundering as Raw; Pod outputs bound to tainted stored content) and three smaller ones (clean-mint memory exhaustion, projection/registry divergence on duplicate records, reused request id) before the freeze; a harness review also tightened counting (reload and hop attempts count only when admitted, error types are classified).
- **R005 exact-prefix-reuse completed (Exact half; preregistered at `r005-prereg`, 5 seeds 17/29/43/71/101), now marked stale by later code changes (`results/STALE.toml`; no rerun under the same preregistration, since each seed runs once; fresh evidence would need a successor experiment).** At the run commit every seed held all 112 coverage and 101 falsifier rules of the frozen `criteria.toml`: bitwise parity of warm (catalog fork) and cold (full prefill) runs, no stale, revoked, cross-principal, cross-project, cross-lineage, cross-regime or cross-environment hit across the 52-cell invalidation matrix (normal, lag, restore, in-flight), hit rate 0.49 on the pinned workload, capacity bounds and saturation fallback held; witness reuse is evidenced statically (two passing tests). Scope: CPU reference path only (4-layer seeded random Qwen2-style stack, pre-RoPE, in-memory pages), not a trained model, not an engine, no GPU, no speed claim; semantic liveness compares revisions, not content (caller-asserted segments are a documented open risk). A pre-freeze dry run found and led to fixing a semantic-supersession liveness defect.
- **A0-ablations-v2 completed with all gates G0-G6 passing** (`research/falsification/A0-ablations-v2/`, 26 records, 5 seeds, CPU, d_model 32 on synthetic operator-routing v1). M002-necessity FALSIFIES in A0 (weak exclusion), M002-sufficiency and M003-nonlinearity SUPPORT, M001-primary/secondary and M003-depth are INCONCLUSIVE, the M004 negative control was not equivalent (optimisation artefact). This is an A0-internal ablation, not the M001-M004 baseline comparison and not language-model evidence; see `RESULTS.md` and `AUDIT.md` there.
- **WP8 CPU smoke run of the `trl-moe` backend is pipeline evidence, not research evidence.** SFT, then GSPO and SAPO arms, routing metrics and `ptr_py` gates ran end to end on a 0.24M-parameter random-init OLMoE, one seed, 32 eval tasks, no baseline and no ablation. The backend stays unselected (`backend_selection.status = "unselected"`); see the WP8 section of [training/README.md](training/README.md). Real OLMoE-1B-7B runs need GPU hardware (`hardware/rtx3090.toml`, `hardware/rtx5090.toml`) and have not been run.

### M001/M002 paired evidence freeze

- Historical `M001-v2` and `M002-v2` records remain immutable and are not used as
  positive evidence claims. The current research-gate failure for `M002-v2` is
  an intentional historical-manifest mismatch, not a reason to rewrite its
  records or silently change its entrypoint.
- `M001-v4` and `M002-v4` each have complete paired five-seed aggregate records
  for seeds `17, 29, 43, 71, 101`, finite metrics, and paired 95% confidence
  interval artifacts. Both decisions are now explicitly bound as
  `INCONCLUSIVE/NO-GO`: on `ood_compose_regime`, accuracy improves but the
  lower-is-better NLL and ECE15 deltas are significantly positive. M002-v4 also
  compares `no-typed-attention` with the plain Transformer and therefore does
  not isolate Typed Attention. Their registry status remains `prepared` to
  preserve the frozen lifecycle history; `DECISION.toml` is the scientific
  decision and permits no positive claim.
- The research gate now accepts the historical M002-v2 conflict only through
  its exact commit- and digest-bound `results/NO-GO.toml` marker. The marker
  does not permit positive completion or alter any historical run record.
- `M002-v5` is `superseded`, not evidence. Its complete non-claimable 16-cell
  pilot archive remains immutable, but the prepared confirmatory entrypoint
  supplied bare fold names where the Rust backend correctly requires
  `fold=FNV64`. The resulting first preflight wrote no metrics and consumed no
  seed. Since a prepared study's configuration and decision script may not be
  replaced, it is superseded rather than patched. A successor must bind the
  exact fold digests before it can be prepared; M009 remains locked and no
  learned backend is authorized.
- `M002-v6` is superseded before scientific evidence. Its one committed
  Seed-17 preflight proved a second interface mismatch: the Rust arm catalog
  bound the selected arm names only to M002-v5. It emitted no dataset row,
  metric, checkpoint, or completed record. The failure is preserved in v6's
  committed runner record and `SUPERSEDED.md`; v6 is not patched in place.
- `M002-v7` is superseded before execution. Its versioned runner pair is
  retained, but the already-prepared v7 protocol lacked a complete Git-tree
  binding for the executable A0 implementation. It has no run record,
  checkpoint, metric, or claim. The immutable status record explains why it
  is not amended in place; M009 remains locked.
- `M002-v8` is superseded before execution. It had the intended A0 Git-tree
  binding, but its newly introduced gate implementation itself failed
  verification (wrong Rust source for the pair table; generated pilot
  artifacts compared to a code-source commit). It has no run record, metric,
  checkpoint, or claim. M002-v9 will only be registered after its corrected
  gate passes against a committed unregistered candidate.
- `M002-v9` is superseded after five committed seed records (Seeds 17, 29, 43,
  71, 101) and before any decision. It bound the immutable v5 pilot archive,
  FNV folds, versioned runner arms and their source files, and the complete
  tracked `model/burn-a0` Git tree; the A0 lockfile had to change for
  `ptr-types`' `sha2` dependency, which changes that tree, and the records
  cannot be re-pinned. Records and configuration are unamended
  (`SUPERSEDED.md`). No decision exists and no positive claim, M009 unlock, or
  Learned Backend authorization follows; a further study needs a new
  preregistered successor.

## Current research/implementation gap

- PTR-Core now has a separate trainable Burn A0 research package with typed metadata, bidirectional cross-attention, recurrent latent refinement and a router; it is still a small architecture probe rather than a pretrained language model
- bounded model resume after verified Pod observations is now wired and revision-advancing; opaque backend checkpoints, async streaming and router-driven operator selection remain open
- hard ActionIR authorization has diagnostic typed decisions plus a scoped synchronous execution gateway with opaque host-issued sessions, exact capsule/project grants and registered verifiers/executors; peer admission by host policy and project-scoped Pods ([doc 29](docs/architecture/29-peer-admission-and-pod-scope.md)), a journaled execution audit with durable fencing and at-most-once keys ([doc 28](docs/architecture/28-durable-execution-audit.md)) and an execution wire on `ALPN_EXEC` ([doc 32](docs/architecture/32-execution-wire.md)) now exist; a signed receipt, a settlement channel for detached work and a journaled admission policy remain open
- feature-gated raft-engine, raft-rs (including a three-member group over `ALPN_RAFT`), Turso and direct Iroh adapters now exist and are under evaluation; an accept loop, discovery, session reuse and real search-backend adapters remain open (membership changes are implemented one voter at a time through `ClusterMember::propose_membership` over `ALPN_RAFT`, and a joining member still has to be admitted out of band) ([doc 31](docs/architecture/31-cluster-integrity.md) lists what the cluster work leaves open)
- M002-v5 is superseded and the other planned architecture experiments are not positive evidence; L001 remains the only running experiment and has two executed scoped crash/recovery evidence slices
- external component candidates mostly lack comparative benchmark evidence
- strong RAG/GraphRAG/editable-memory and matched plain-model comparative runs are not yet executed
- GitHub main-branch ruleset/repository-admin settings still require manual completion; recommendations are documented in `docs/GITHUB_SETTINGS.md`
- the TypeScript SDK and Axum server now share `/health` and `/v1/requests`; streaming, auth/session policy and a real configured model backend remain open

## Reproducibility

- `Cargo.lock` pins Rust dependencies
- `training/uv.lock` pins the current Python training utility environment
- `rust-toolchain.toml` declares Rust 1.99.0 as MSRV/default
- CI separately tests MSRV and current stable on Linux and current stable on Windows
- experiment preparation/execution records git SHA, hardware profile, lockfile hashes, exact argv, duration, exit status and process output
- small JSON/TOML/Markdown result artifacts and the frozen operator-routing v1/v2
  datasets are versioned; model checkpoints remain out of Git

See [docs/components/STATUS.md](docs/components/STATUS.md) for per-component maturity and [experiments/lifecycle/L001-revocation-crash/results/](experiments/lifecycle/L001-revocation-crash/results/) for the first lifecycle evidence.

P0.2 adds canonical journaled semantic transactions, typed binary Pod results, transactional dependencies and validated revision reconstruction. Invalidated current derivations are evicted; prior log records and immutable snapshots remain. See `docs/architecture/22-durable-semantic-state.md`. (This paragraph was written at P0.2. Neural-state admission ([doc 27](docs/architecture/27-neural-state-admission.md)), erasure and retention ([doc 25](docs/architecture/25-erasure-and-retention.md)) and cluster integrity ([doc 31](docs/architecture/31-cluster-integrity.md)) have since landed at library level; each of those documents states what it leaves open.)

## P0.3 persistence reference boundary

[Checked records and recovery snapshots](docs/architecture/23-persistence-integrity.md)
now cover bounded hash-chain verification, explicit legacy migration and complete
replay-backed snapshot restore. Existing files are never overwritten by restore.
Rollback protection requires an independently trusted anchor; default unanchored
open cannot detect a wholesale valid rewrite. This is not log compaction, remote
authentication, secure erasure or neural-state restoration. Earlier L001 numbers
are historical v1-format evidence, not measurements of this new v2 implementation.
