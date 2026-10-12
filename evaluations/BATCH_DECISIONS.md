# Batch run decisions (B1-B8)

Date: 2026-10-11. Input: `evaluations/SPIKE.md` (2026-10-09), `evaluations/PROTOCOL_FORMAT.md`,
`evaluations/registry.toml`, `evaluations/components/*/candidates.toml`,
`scripts/run_component_eval.py`, `hardware/cpu-4core-15gib.toml`.

**A decision to run is not a selection.** Nothing here makes a candidate `preferred`; no
winner exists without measured evidence (`PROTOCOL_FORMAT.md` status rules). No seed was run
for this document.

## Environment probed on 2026-10-11

| Item | State |
|---|---|
| Machine | 4 vCPU, 15 GiB RAM, no GPU, ext4 on virtio, shared and loaded: `decision_grade = false`, so durable-write numbers (B2, event-streaming fsync) are diagnostic only and cannot make a candidate `preferred` |
| PostgreSQL | 16 on 127.0.0.1:5433 accepting connections; the installed pgvector is 0.6.0 (Ubuntu `postgresql-16-pgvector`), which has no `halfvec` (needs 0.7+) |
| Build tools present | `cc`, `make`, `clang`, `cmake`, `rg`, `psql`, `uv`, python 3.11 |
| Absent (obtainable) | `protoc`, `capnp`, `tokei`, `nats-server`, `emacs`. `apt.postgresql.org` (PGDG), `archive.ubuntu.com` and GitHub release assets answered 200/206 |
| io_uring | `kernel.io_uring_disabled = 0`; whether the seccomp profile admits `io_uring_setup` is only known at run time |
| Free disk on `/home/user` | 9.3 GB; the 6 GB build floor allows one heavy stack (rocksdb, datafusion, torch) at a time |

## Protocol and runner state (applies to every batch)

* There is **no** `protocol.toml` in any component, and `evaluations/bench/` holds only `kit`
  (no batch workspace, no `Cargo.lock` with candidate pins). `check_component_protocols.py`
  therefore has no benchmark evidence to check. Per `PROTOCOL_FORMAT.md` the protocol and the
  batch bench workspace (adapters per candidate, re-spiked lock) must be committed **before**
  the first evidence run.
* `scripts/run_component_eval.py run <component> <candidate>` executes only the `command` of
  a candidate in `candidates.toml`. The only candidates with a command are in-repo or
  feature-test smoke commands (table at the end); they write legacy records that are not
  benchmark evidence. No external candidate (jemalloc, rkyv, tantivy, ...) has a command, so
  none can be executed by the runner today.
* "RUN-NOW-CPU" below therefore means: runnable in this container once the missing protocol and
  bench workspace exist. What is missing is listed per batch.

## Legend

Decision: RUN-NOW-CPU, DEFER-GPU, DEFER-INFRA, BLOCKED (advisory, licence, vendor, retirement).
Licence column: crates.io `license` field of the current version (read 2026-10-11) or, for
non-crate artifacts, from the spike or from memory (marked "to verify"). "OWNER LICENCE
DECISION REQUIRED" candidates are not selected and not run, whatever the technical state.

## Summary

| Batch | Components | Decision | Remark |
|---|---|---|---|
| B1 in-process micro | allocator, local-serialization, network-codec, introspection, observability, semantic-db | RUN-NOW-CPU | excluded: bincode (BLOCKED), serde-json-tree (not obtainable), nemo-relay (OWNER LICENCE DECISION REQUIRED); needs `protoc`, `capnp` installs |
| B2 storage, consensus | ledger, consensus | RUN-NOW-CPU, diagnostic only | durable writes are not decision-grade here; rocksdb is a heavy build; raft crates need the carried vendor patches |
| B3 PostgreSQL, analytics | materialized-state, relational-substrate, analytics-mirror | RUN-NOW-CPU (partial) | PG16 + pgvector 0.8.7 needs PGDG install; PG18, pg-duckdb, iceberg-mirror deferred or blocked |
| B4 retrieval | lexical-search, local-vector-search, structural-code-search | RUN-NOW-CPU (partial) | AGPL candidates and lancedb need an owner licence decision; pg-textsearch needs PG18 |
| B5 network, objects, streaming | network, object-storage, event-streaming | RUN-NOW-CPU (partial) | loopback only; redpanda DEFER-INFRA and licence; iggy server is a source build |
| B6 runtime | execution-runtime | RUN-NOW-CPU | tina-rs not obtainable; compio needs a run-time io_uring check |
| B7 ML on CPU | model-framework, ingress-classifier, label-model, fast-weight-memory | RUN-NOW-CPU (partial) | burn-cubecl DEFER-GPU; gliclass-rs BLOCKED; llm-only-baseline DEFER-INFRA |
| B8 verification, security | code-quality-verifier, security-context | RUN-NOW-CPU (baselines only) | sentrux not obtainable for Rust; brin-signal DEFER-INFRA |

## B1 in-process micro (spike `b1`) - RUN-NOW-CPU

| Component | Candidate | Licence | Obtainability | Decision | Reason / blocking item |
|---|---|---|---|---|---|
| allocator | system | std | built in | RUN-NOW-CPU | incumbent |
| allocator | tikv-jemallocator 0.7.0 | MIT/Apache-2.0 | crate, builds jemalloc with `cc`, `make` (present) | RUN-NOW-CPU | clean |
| allocator | mimalloc 0.1.52 | MIT | crate, needs `cc` | RUN-NOW-CPU | clean |
| local-serialization | rkyv 0.8.18 | MIT | crate | RUN-NOW-CPU | clean |
| local-serialization | bincode 2.0.1 / 3.0.0 | MIT | crate | BLOCKED | RUSTSEC-2025-0141 (unmaintained) on the crate, every version. Registry status `rejected` with note |
| local-serialization | postcard 1.1.3 | MIT OR Apache-2.0 | crate | RUN-NOW-CPU | only with `default-features = false, features = ["use-std"]` (defaults pull `atomic-polyfill`, RUSTSEC-2023-0089) |
| local-serialization | hand codec (`ptr-semdb/src/codec.rs`) | repo | in repo | RUN-NOW-CPU | incumbent |
| network-codec | prost-protobuf (prost 0.14.4) | Apache-2.0 | crate; `prost-build` needs `protoc` | RUN-NOW-CPU | incumbent; install apt `protobuf-compiler` 3.21.12-8.2ubuntu0.3 (sha256 in SPIKE) or use a vendored protoc |
| network-codec | postcard | MIT OR Apache-2.0 | crate | RUN-NOW-CPU | same feature set as above |
| network-codec | capnproto (capnp 0.27.2, capnpc 0.27.0) | MIT | crates; compiler via apt `capnproto` 1.0.1-4 | RUN-NOW-CPU | compiler not installed (obtainable) |
| introspection | valuable 0.1.1 | MIT | crate | RUN-NOW-CPU | clean |
| introspection | serde-json-tree | n/a | no such crate | DEFER (not obtainable) | owner must name the project; `serde_json` is only a stand-in and is not the candidate. Note added in registry |
| introspection | custom-inspect | repo | in repo | RUN-NOW-CPU | |
| observability | tracing 0.1.44 + tracing-subscriber 0.3.23 | MIT | crate | RUN-NOW-CPU | clean |
| observability | opentelemetry 0.33.0 + sdk (`trace`, `testing`) | Apache-2.0 | crate | RUN-NOW-CPU | clean, in-memory exporter |
| observability | nemo-relay 0.10.0 | Apache-2.0 (own); graph needs MIT-0 `borrow-or-share` | crate | OWNER LICENCE DECISION REQUIRED | MIT-0 not in `deny.toml` allow list; non-optional. Not selected, not run |
| semantic-db | salsa 0.28.5 | Apache-2.0 OR MIT | crate | RUN-NOW-CPU | clean |
| semantic-db | ptr-incremental-semdb, custom-dependency-engine | repo | in repo | RUN-NOW-CPU | incumbents |

Missing before a run: `protocol.toml` for six components, `evaluations/bench/b1/` workspace
(adapters, lock pinned `=x.y.z`, re-spike of that lock), `protoc` and `capnp` installs
(system package install is a head decision), a semdb workload definition beyond the existing
`ptr-bench semdb` smoke.

## B2 storage and consensus (spike `b2`) - RUN-NOW-CPU, diagnostic only

Durable-write metrics on `cpu-4core-15gib` are not decision-grade (fdatasync p50 about 4 ms).
A run yields gate results (correctness, crash semantics) and noisy comparisons; `preferred`
needs a decision-grade profile.

| Component | Candidate | Licence | Obtainability | Decision | Reason / blocking item |
|---|---|---|---|---|---|
| ledger | FileLedger | repo | in repo | RUN-NOW-CPU | incumbent |
| ledger | raft-engine 0.4.2 | Apache-2.0 | crate | RUN-NOW-CPU | needs carried `vendor/raft-engine` patch (protobuf 3.7.2) and `default-features = false`; `b2-patched` clean. Vendor not edited here |
| ledger | rocksdb 0.25.0 | Apache-2.0 | crate; clang and cmake present | RUN-NOW-CPU | clean; long C++ build, keep disk above 6 GB |
| ledger | redb | MIT OR Apache-2.0 (newest listed 2.6.4; spike pinned 4.3.0, re-check) | crate | RUN-NOW-CPU | clean |
| consensus | raft-rs 0.7.0 (`prost-codec`) | Apache-2.0 | crate | RUN-NOW-CPU | needs carried `vendor/raft` + `vendor/raft-proto` |
| consensus | openraft 0.9.25 (`serde`) | MIT OR Apache-2.0 | crate | RUN-NOW-CPU | clean; the registry tag on crates.io is now a 0.10 alpha, pin 0.9.25 |

Missing: protocols and bench workspace (copy the root `[patch.crates-io]` entries), a crash
and torn-write failure-injection gate set (the repo has `failpoints` features for the incumbent
only).

## B3 PostgreSQL and analytics (spike `b3`) - RUN-NOW-CPU (partial)

| Component | Candidate | Licence | Obtainability | Decision | Reason / blocking item |
|---|---|---|---|---|---|
| materialized-state | turso 0.8.2 | MIT | crate | RUN-NOW-CPU | clean (informational lru 0.16.4 unsound) |
| materialized-state | sqlite (rusqlite 0.40.2 `bundled`) | MIT | crate | RUN-NOW-CPU | clean |
| materialized-state | postgresql 16.15 | PostgreSQL licence (to verify) | installed, port 5433 | RUN-NOW-CPU | clients tokio-postgres 0.7.18 / postgres 0.19.14 |
| relational-substrate | postgresql-16-pgvector | PostgreSQL licence (to verify) | installed pgvector is 0.6.0; 0.8.7 from PGDG (reachable) or PGXN source | RUN-NOW-CPU after install | system install of `postgresql-16-pgvector` 0.8.7 is a head decision |
| relational-substrate | postgresql-18-pgvector | PostgreSQL licence (to verify) | PGDG PG18 18.6 not installed | DEFER-INFRA | needs a second server install; reachable but not present |
| relational-substrate | turso-projection-only | MIT | crate | RUN-NOW-CPU | clean |
| analytics-mirror | ptr-analytics-metric-sql | repo | in repo | RUN-NOW-CPU | incumbent |
| analytics-mirror | datafusion 55.2.0 | Apache-2.0 | crate | RUN-NOW-CPU | only with `default-features = false`, no `compression` (bzip2-1.0.6 outside allow list) and the feature list in SPIKE; heavy build |
| analytics-mirror | iceberg-mirror 0.10.1 | Apache-2.0 | crate | BLOCKED (vendor) | needs a new `paste-alias` patch of `parquet 58.4.0` (RUSTSEC-2024-0436); `vendor/` is not edited in this task. Not rejected: retry when a patch is created or iceberg moves to parquet 59 |
| analytics-mirror | pg-duckdb | MIT (to verify) | source only, GitHub archive answers 403 | DEFER-INFRA | not obtainable here; note added |

Missing: protocols, bench workspace, pgvector 0.8.7 install, a PG schema/cleanup harness on
a scratch database (DSN stays in environment variables, never logged).

## B4 retrieval (spike `b4`) - RUN-NOW-CPU (partial)

| Component | Candidate | Licence | Obtainability | Decision | Reason / blocking item |
|---|---|---|---|---|---|
| lexical-search | tantivy 0.26.2 | MIT | crate | RUN-NOW-CPU | clean |
| lexical-search | custom-inverted-index | repo | in repo | RUN-NOW-CPU | |
| lexical-search | postgres-fts | PostgreSQL licence (to verify) | built into PG16 | RUN-NOW-CPU | |
| lexical-search | pg-textsearch-bm25 1.5.1 | PostgreSQL licence (to verify) | PGDG PG17/18 only | DEFER-INFRA | no PG16 build; needs PG18 |
| lexical-search | paradedb-pg-search 0.26.1 | AGPL-3.0 | PGXN source, pgrx build | OWNER LICENCE DECISION REQUIRED | not selected, not run |
| local-vector-search | hnsw-library (usearch 2.26.4, hnsw_rs 0.3.5) | Apache-2.0; MIT/Apache-2.0 | crates; usearch needs `cc` | RUN-NOW-CPU | clean |
| local-vector-search | zvec 0.1.0 | Apache-2.0 | crate; native lib from PyPI wheel sha256 `ee2febd1...a502b7` | RUN-NOW-CPU | PyPI reachable |
| local-vector-search | lancedb-embedded 0.40.0 | Apache-2.0; graph has MPL-2.0 `option-ext` | crate | OWNER LICENCE DECISION REQUIRED | MPL-2.0 exception plus a new `paste-alias` patch; not selected, not run |
| local-vector-search | pgvector-halfvec-hnsw | PostgreSQL licence (to verify) | needs pgvector 0.8.7 (installed 0.6.0 has no halfvec) | RUN-NOW-CPU after install | as B3 |
| local-vector-search | vectorchord 1.1.1 | AGPL-3.0 (spike; to verify for other licences) | PGXN source | OWNER LICENCE DECISION REQUIRED | not selected, not run |
| structural-code-search | tree-sitter-custom (tree-sitter 0.27.1, tree-sitter-rust 0.24.2, `syn` for truth) | MIT | crates | RUN-NOW-CPU | clean |
| structural-code-search | ripgrep-only-baseline (15.2.0) | Unlicense OR MIT (to verify) | release binary, sha256 `33e15bcf...7b149c`; assets reachable | RUN-NOW-CPU | baseline |
| structural-code-search | gritql | MIT (to verify) | external binary `v0.1.0-alpha.1743007075`, sha256 `94b34641...163149`; cannot enter a lock (git dep refused) | RUN-NOW-CPU as external process | alpha; process-only adapter |

Missing: protocols, bench workspace, an embedding/corpus generator (kit `gen` has Zipf corpora
and Gaussian mixtures), recall ground truth (exact scan) as a gate, `syn`-based truth for
structural search.

## B5 network, object storage, streaming (spike `b5`) - RUN-NOW-CPU (partial)

Network results are loopback only: no WAN, NAT or multi-host claim can follow.

| Component | Candidate | Licence | Obtainability | Decision | Reason / blocking item |
|---|---|---|---|---|---|
| network | iroh 1.3.0 | MIT OR Apache-2.0 | crate | RUN-NOW-CPU | exactly `default-features = false, features = ["tls-ring"]` and carried `vendor/netlink-packet-core-0.8.2` |
| network | quinn 0.11.12 | MIT OR Apache-2.0 | crate | RUN-NOW-CPU | clean |
| network | grpc (tonic 0.14.6, prost 0.14.4) | MIT; Apache-2.0 | crates | RUN-NOW-CPU | clean |
| object-storage | opendal 0.59.4 (`services-fs`, `services-memory`) | Apache-2.0 | crate | RUN-NOW-CPU | incumbent family |
| object-storage | object-store-rs (object_store 0.14.2) | MIT/Apache-2.0 | crate | RUN-NOW-CPU | clean |
| object-storage | direct-backends | repo | in repo | RUN-NOW-CPU | |
| event-streaming | postgres-projection-event-log | repo + PG licence | PG16 on 5433 | RUN-NOW-CPU | fsync numbers not decision-grade |
| event-streaming | nats (async-nats 0.50.0, nats-server v2.15.1) | Apache-2.0 | server tarball sha256 `bf5edc7b...954c70` from the release `SHA256SUMS` | RUN-NOW-CPU | binary not installed (download only) |
| event-streaming | iggy (client 0.11.0, server 0.9.0 source) | Apache-2.0 | server only from source (sha512 in SPIKE); client newer than server | DEFER-INFRA | source build and a protocol-compatibility check first |
| event-streaming | redpanda-kafka | BSL / source-available (to verify) | container or cluster | DEFER-INFRA, OWNER LICENCE DECISION REQUIRED | needs containers and a licence decision; not selected, not run |

Missing: protocols, bench workspace, a local S3 server (`s3s-fs` 0.17.0 via `cargo install
--locked`) for the S3 path, a bounded-queue/backpressure gate per transport.

## B6 runtime (spike `b6`) - RUN-NOW-CPU

| Component | Candidate | Licence | Obtainability | Decision | Reason / blocking item |
|---|---|---|---|---|---|
| execution-runtime | ptr-isolates | repo | in repo | RUN-NOW-CPU | incumbent; `ptr-bench mailbox` smoke exists |
| execution-runtime | tokio-custom (tokio 1.53.2) | MIT | crate | RUN-NOW-CPU | clean |
| execution-runtime | compio-runtime 0.19.2 | MIT | crate | RUN-NOW-CPU | confirm `io_uring_setup` is permitted under the sandbox seccomp before building the adapter; otherwise record as INVALID for this profile, not as a candidate failure |
| execution-runtime | tina-rs | n/a | no such crate | DEFER (not obtainable) | owner must name the project; crates.io `tina` is unrelated and blocked. Note added |

Missing: protocol, bench workspace, bounded-mailbox and cancellation gates for the external
runtimes. Three runnable candidates, so a comparison is possible.

## B7 ML on CPU (spike `b7`) - RUN-NOW-CPU (partial)

| Component | Candidate | Licence | Obtainability | Decision | Reason / blocking item |
|---|---|---|---|---|---|
| model-framework | candle (core/nn 0.11.0) | MIT OR Apache-2.0 | crate | RUN-NOW-CPU | needs carried `paste-alias` patches (`gemm*`, `pulp`); tokenizers only at 0.22.2 |
| model-framework | burn-0.18-ndarray-a0 | MIT OR Apache-2.0 | A0 workspace, in repo | RUN-NOW-CPU | incumbent; moving to burn 0.22.0 stable needs seven new alias patches, so stay on the A0 lock |
| model-framework | pytorch-reference (torch 2.14.1 CPU) | BSD-3-Clause (to verify) | `download.pytorch.org/whl/cpu`, sha256 `5e38154c...e34f37`; large wheel | RUN-NOW-CPU | not advisory-scanned; mind the disk floor |
| model-framework | burn-cubecl | MIT OR Apache-2.0 | crate | DEFER-GPU | GPU path |
| ingress-classifier | gliclass-rs 1.0.0 | to verify | crate | BLOCKED | RUSTSEC-2025-0119, RUSTSEC-2024-0436, MPL-2.0; not trimmable. Registry `rejected` with note |
| ingress-classifier | burn-small-classifier | repo | in repo | RUN-NOW-CPU | only a lone runnable candidate here |
| ingress-classifier | llm-only-baseline | n/a | no model or runtime pinned | DEFER-INFRA | human choice of model and runtime |
| label-model | ptr-labeling-dawid-skene, majority-vote | repo | in repo | RUN-NOW-CPU | |
| label-model | snorkel-label-model 0.10.0 | Apache-2.0 | PyPI wheel sha256 `42a2f354...f927c8`; needs torch, pandas, scikit-learn | RUN-NOW-CPU | Python, not advisory-scanned (`pip-audit` not run) |
| fast-weight-memory | ptr-fastmem-gated-delta, recency-buffer, hybrid-retrieval-only | repo | in repo | RUN-NOW-CPU | no external candidate; baseline-versus-incumbent only |

Consequence: ingress-classifier has one runnable candidate and cannot reach `preferred`
(lone survivor rule) until another credible candidate exists.

## B8 verification and security (spike `b8`) - RUN-NOW-CPU (baselines only)

| Component | Candidate | Licence | Obtainability | Decision | Reason / blocking item |
|---|---|---|---|---|---|
| code-quality-verifier | compiler-tests-only | repo | baseline | RUN-NOW-CPU | |
| code-quality-verifier | custom-static-metrics (syn 3.0.6, quote, proc-macro2) | MIT OR Apache-2.0 (to verify) | crates | RUN-NOW-CPU | `rust-code-analysis` is MPL-2.0 plus RUSTSEC-2025-0057 and is not an option |
| code-quality-verifier | sentrux 0.1.6 | MIT (spike) | PyPI, Python-only analyser | DEFER (not obtainable for Rust) | cannot analyse Rust; owner must identify the tool. Note added |
| security-context | ptr-policy | repo | in repo | RUN-NOW-CPU | |
| security-context | custom-scanners (regex 1.13.1, aho-corasick 1.1.5) | MIT OR Apache-2.0 (to verify) | crates | RUN-NOW-CPU | |
| security-context | brin-signal | unknown | external service, unidentified | DEFER-INFRA | note added |

## Not in B1-B8 (unchanged)

DEFER-GPU: gpu-vector-search, inference-serving, adapter-serving, training-backend.
DEFER-INFRA: distributed-data-compute, distributed-search (cluster), environment-runtime
(orchestrator and LLM policy), rust-editor-emacs (human judgement, no emacs).

## Commands the runner can execute today

All are existing smoke commands (legacy records, not benchmark evidence), run from the
repository root as `python3 scripts/run_component_eval.py run <component> <candidate>`.
Not executed here.

| Component | Candidate | Declared command | Cost note |
|---|---|---|---|
| semantic-db | ptr-incremental-semdb | `cargo run --release --locked -p ptr-bench -- semdb 100000` | one `ptr-bench` build, then seconds; build time not measured |
| execution-runtime | ptr-isolates | `cargo run --release --locked -p ptr-bench -- mailbox 100000` | same |
| label-model | ptr-labeling-dawid-skene | `cargo test -p ptr-labeling --locked` | small crate |
| fast-weight-memory | ptr-fastmem-gated-delta | `cargo test -p ptr-fastmem --locked` | small crate |
| ledger / consensus | raft-engine / raft-rs | `cargo test -p ptr-ledger --features raft-engine-backend --test raft_engine --locked` (and `raft-rs-backend --test raft_rs`) | feature-gated C/Rust build, minutes |
| materialized-state, relational-substrate, lexical-search, local-vector-search, event-streaming | postgresql, postgresql-18-pgvector, postgres-fts, pgvector-halfvec-hnsw, postgres-projection-event-log | `cargo test -p ptr-pg --features postgres-backend --test postgres --locked` | needs `PTR_PG_TEST_DSN` (not logged); the same test is declared for five candidates, so it does not distinguish them |
| materialized-state | turso | `cargo test -p ptr-state --features turso-backend --test turso --locked` | heavy build |
| network | iroh | `cargo test -p ptr-net --features iroh-backend --test iroh --locked` | heavy build |
| model-framework | burn-0.18-ndarray-a0 | `cargo test --manifest-path model/burn-a0/Cargo.toml --locked` | separate workspace |
| analytics-mirror | ptr-analytics-metric-sql | `cargo test -p ptr-analytics -p ptr-pg --locked` | needs PG DSN for the `ptr-pg` part |

Because the same test serves several candidates, repeating these does not produce evidence
for a comparison; the comparison needs the per-batch protocol and bench workspace first.

## Registry fields changed

`candidates.toml` supports `status` (`to-evaluate`, `evaluating`, `preferred`, `rejected`,
`superseded`) and `note`; nothing else was added. `rejected` with a `note` for the hard
blocks (bincode, gliclass-rs). A `note` only (status unchanged) for the owner-decision and
not-obtainable candidates: paradedb-pg-search, vectorchord, nemo-relay, lancedb-embedded,
redpanda-kafka, tina-rs, serde-json-tree, sentrux, pg-duckdb, brin-signal.

## Open owner questions

*Answered by delegation on 2026-10-11: see the section "Owner delegation decisions (2026-10-11)" at the end; the questions are kept as asked.*

1. AGPL-3.0 (paradedb-pg-search, vectorchord): allow as external-process-only candidates, or retire? Not run until answered.
2. MIT-0 into the `deny.toml` allow list (unblocks nemo-relay)?
3. MPL-2.0 exception for `option-ext` (lancedb-embedded; gliclass-rs is also blocked by advisories)?
4. Redpanda licence terms: acceptable for a benchmark-only container, and is a container host available?
5. Which projects are meant by `tina-rs`, `serde-json-tree`, `sentrux` (Rust-capable), `brin-signal`?
6. May system packages be installed for the runs (protoc, capnproto, pgvector 0.8.7 from PGDG, nats-server)?

## Owner delegation decisions (2026-10-11)

The owner delegated the open questions above ("do everything necessary in the best execution
without my answers", about 20:25 on 2026-10-11). The head decided conservatively and
reversibly; source: `finish-decisions.md` of the head, entry "Owner delegation". Each decision
can be replaced by a later explicit owner decision. None of them is evidence, and no candidate
became `preferred`.

| # | Question above | Decision | Effect on the candidates |
|---|---|---|---|
| 1 | AGPL-3.0 (paradedb-pg-search, vectorchord) | Not used. A network-facing AGPL component would bind the project's licence choice. Not selected, not run, not vendored. | status `rejected`, note "licence (AGPL-3.0), owner-delegated decision 2026-10-11, reversible" |
| 2 | MIT-0 into the `deny.toml` allow list | MIT-0 is strictly more permissive than MIT (no attribution clause), so it is allowed for `evaluations/bench/*` workspaces only, and only when the candidate is actually built. No build in this goal; `deny.toml` is not changed here. | nemo-relay: note only, status unchanged |
| 3 | MPL-2.0 exception for `option-ext` | Deferred. It needs a new vendor patch, which `docs/VENDOR_PATCH_POLICY.md` treats as its own change. Not granted now. | lancedb-embedded: note only |
| 4 | Redpanda terms | Not used (BSL, source-available, needs a container host). Event streaming stays on the open candidates. | redpanda-kafka: note only |
| 5 | Unidentified projects (`tina-rs`, `serde-json-tree`, a Rust `sentrux`, `brin-signal`) | Marked not obtainable. No guess at a different project under the same name. | note only on each |
| 6 | System package installs (protoc, capnproto, pgvector 0.8.7, nats-server) | No package is installed in this goal. They are prerequisites of the measurement study below, not of any open experiment. | none |

**Recommended next study (not started in this goal).** Running a batch needs a new study; it
is not required by the completion criteria of this goal. Its prerequisites, as listed in
"Protocol and runner state" above:

1. a `protocol.toml` per component (`evaluations/PROTOCOL_FORMAT.md`);
2. a pinned bench workspace under `evaluations/bench/` with a re-spiked `Cargo.lock`;
3. one adapter per candidate in that workspace;
4. a quiet host (the present one is shared and `decision_grade = false`), so that durable-write
   numbers can decide something.

Until then no external candidate can be executed by `scripts/run_component_eval.py`, and
the comparative evidence for these components stays absent.
