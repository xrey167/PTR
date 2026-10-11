# Advisory and obtainability spike (WP-E0)

Date: 2026-10-09. Input for the benchmark batches B1-B8 (`evaluations/PROTOCOL_FORMAT.md`):
before a candidate crate enters a bench workspace it is classified here. A candidate with a
security advisory in its graph is **blocked** unless a cheap vendor patch exists
(`docs/VENDOR_PATCH_POLICY.md`); there are no advisory ignores. The spike changed no product
crate and compiled nothing (`cargo generate-lockfile` only).

## Method and tool versions

| Item | Value |
|---|---|
| cargo-deny | 0.20.2, release binary `cargo-deny-0.20.2-x86_64-unknown-linux-musl.tar.gz`, sha256 `9f12ed4c49936e09b48bf862b595cde2fe64fcbd9d74dfacac6131ca824c8d5f` (equals the upstream-published `.sha256`); the version `security.yml` installs |
| cargo-audit | 0.22.2, release binary `cargo-audit-x86_64-unknown-linux-musl-v0.22.2.tgz` from the `cargo-audit/v0.22.2` release, sha256 `7fb9497f8594b389e5fce5ef9b92db08432996895b2e0c5a0167a69ed445c428`. Upstream publishes no checksum for it; two independent downloads agree (trust on first use) |
| advisory-db | `rustsec/advisory-db` @ `7eebec69c352c7191b1f13eb95dd510eeca5d1de` (2026-10-09 10:12 +02:00, 1296 advisories); cargo-audit ran `--no-fetch` on this checkout and cargo-deny's own fetch resolved to the same commit |
| cargo / rustc | 1.99.0 (`cargo +1.99.0 generate-lockfile`), crates.io sparse index of 2026-10-09 |
| deny configuration | the repository `deny.toml` unchanged (`all-features = true`, `advisories.ignore = []`, the licence allow list and the single `colored` MPL exception); `cargo deny check advisories licenses sources` |
| audit command | `cargo-audit audit --file Cargo.lock --db advisory-db --no-fetch --json` |

Each batch is one throwaway manifest **outside the repository** listing the batch's candidate
crates with direct dependencies pinned `=x.y.z` at the newest stable crates.io version of the
day (the lock records everything else). Both tools ran against the resulting lock; findings were
attributed to the direct dependency that reaches them by walking the lock. Variants (`-min`,
`-patched`) re-ran a batch with a trimmed feature set and/or the repository's own
`[patch.crates-io]` vendor entries to see whether a finding is avoidable.

What counts as an advisory: whatever fails `cargo deny` in CI, that is vulnerabilities **and
unmaintained** crates. `cargo audit` alone fails only on vulnerabilities. *Unsound* advisories
(informational; deny only fails them for workspace members) and yanked crates are listed but do
not block. Licences are judged by `deny.toml`; a candidate whose graph needs a licence outside
the allow list is blocked **by policy** (an owner decision, not a patch).

Caveats, all in the direction of "re-check before committing a lock":

* One lock per batch, as the real bench workspace will have. Version unification across
  candidates in a batch can, in principle, hide or cause a finding; flagged candidates were
  re-run in variants. The actual bench lock must be spiked again when it is created.
* Transitive versions float to the newest compatible on 2026-10-09; the findings change as
  advisories are published (the database commit above is the reference).
* Feature selection matters and is part of the verdict ("clean with `default-features = false`").
* Only Rust graphs were scanned. Python packages (`snorkel`, `torch`) were not run through
  `pip-audit`; non-Rust artifacts are pinned by checksum, not scanned.

## Classification vocabulary

| Class | Meaning |
|---|---|
| clean | no advisory, no licence rejection under the repository `deny.toml`, as stated (features noted) |
| needs-vendor-patch | an advisory whose fix is a patch of a class `docs/VENDOR_PATCH_POLICY.md` already carries; "carried" = the patch exists in `vendor/` and verified clean in a `-patched` variant; "cheap" = same class, one crate; "new" = would have to be created |
| blocked | an advisory with no cheap patch, or a licence outside the allow list; the id is given. A blocked candidate is `rejected` with a `note`, or waits for the stated owner decision |
| not-obtainable | the artifact cannot be identified or fetched with a pinned version and checksum from this sandbox (reason given) |
| stays-open | needs a GPU, a cluster, a container orchestrator, or human judgement; not spiked |

## Result by component

CPU-capable components of `plan-l-e-r-evals.md` 4.2 (24 of 32). In-repo ("internal") candidates are in
the root or A0 lock, which the existing `security.yml` audits; they are **clean** by that CI and
are not repeated per row. Batch letters are the plan's batches; spike lock = `evals-spike/<batch>/`.

### B1 in-process micro (spike `b1`, 406 packages; `b1-min`, 400)

| Component | Candidate | Version | Class | Detail |
|---|---|---|---|---|
| allocator | system | std | clean | no external graph |
| allocator | tikv-jemallocator | 0.7.0 | clean | builds jemalloc from vendored source: needs `cc` and `make` |
| allocator | mimalloc | 0.1.52 | clean | needs `cc` |
| local-serialization | rkyv | 0.8.18 | clean | |
| local-serialization | bincode | 2.0.1, 3.0.0 | **blocked** | RUSTSEC-2025-0141 (unmaintained), on the crate itself, every version; nothing to patch without replacing the candidate |
| local-serialization | postcard | 1.1.3 | clean with `default-features = false, features = ["use-std"]` | default features add `heapless 0.7` and `atomic-polyfill 1.0.3`, RUSTSEC-2023-0089 (unmaintained); the trimmed set is clean (`b1-min`). Use exactly this feature set |
| local-serialization | hand codec (`ptr-semdb/src/codec.rs`) | in repo | clean | incumbent |
| network-codec | prost-protobuf | prost 0.14.4 | clean | incumbent. `prost-build` also needs `protoc`: apt `protobuf-compiler` 3.21.12-8.2ubuntu0.3, sha256 `f308788203b1fc4292783115b9a0517eb2f3d1dbbfe28916f4aea6aca702691a`, not installed here |
| network-codec | postcard | 1.1.3 | clean | as above |
| network-codec | capnproto | capnp 0.27.2, capnpc 0.27.0 | clean (crates); compiler obtainable | `capnp` compiler: apt `capnproto` 1.0.1-4, sha256 `5c2750e83cb5c27d51b6162e2c08824933bda56021f4f0958f674faa531b3c3a`; not installed |
| introspection | valuable | 0.1.1 | clean | |
| introspection | serde-json-tree | n/a | **not-obtainable** | no crate of that name exists on crates.io; `serde_json` 1.0.151 (clean) is the nearest stand-in for a JSON-tree walker. The owner should confirm which project was meant |
| introspection | custom-inspect | in repo | clean | |
| observability | tracing | tracing 0.1.44, tracing-subscriber 0.3.23 | clean | |
| observability | opentelemetry | opentelemetry 0.33.0, opentelemetry_sdk 0.33.0 (`trace`, `testing`) | clean | in-memory exporter via `testing` |
| observability | nemo-relay | 0.10.0 | **blocked (licence)** | NVIDIA NeMo Relay core SDK. No advisory. Its non-optional dependency `jsonschema 0.46.10` -> `referencing` -> `fluent-uri 0.4.1` -> `borrow-or-share 0.2.4` is **MIT-0**, not in `deny.toml`'s allow list. MIT-0 is OSI-approved and permissive; adding it is an owner policy decision, after which the candidate is clean |
| semantic-db | salsa | 0.28.5 | clean | |
| semantic-db | ptr-incremental-semdb, custom-dependency-engine | in repo | clean | |

### B2 storage and consensus (spike `b2` 232 packages; `b2-min`; `b2-patched` 205)

| Component | Candidate | Version | Class | Detail |
|---|---|---|---|---|
| ledger | raft-engine | 0.4.2 | **needs-vendor-patch (carried)** + feature trim | unpatched: `protobuf 2.28.0` RUSTSEC-2024-0437 (vulnerability) and, with default features, `rhai` -> `smartstring 1.0.1`: RUSTSEC-2026-0249 (unmaintained) and MPL-2.0. Carried: `vendor/raft-engine` (class `raft-protobuf`, protobuf 3.7.2). Use `default-features = false` (drops `scripting`/rhai). `b2-patched` (root `[patch.crates-io]` entries for raft, raft-proto, raft-engine + that feature set): 0 advisories, 0 licence rejections |
| ledger | rocksdb | 0.25.0 | clean | needs `clang`/libclang and a long C++ build (present: clang, cmake) |
| ledger | redb | 4.3.0 | clean | |
| ledger | FileLedger | in repo | clean | incumbent |
| consensus | raft-rs | raft 0.7.0 (`prost-codec`) | **needs-vendor-patch (carried)** | unpatched: `protobuf 2.28.0` RUSTSEC-2024-0437 and `fxhash 0.2.1` RUSTSEC-2025-0057 (unmaintained). Carried: `vendor/raft` + `vendor/raft-proto` (fxhash -> rustc-hash, protobuf optional). `b2-patched` clean |
| consensus | openraft | 0.9.25 (`serde`) | clean | |

### B3 PostgreSQL and analytics (spike `b3` 594 packages; `b3-min` 585)

| Component | Candidate | Version | Class | Detail |
|---|---|---|---|---|
| materialized-state | turso | 0.8.2 | clean | informational: `lru 0.16.4` RUSTSEC-2026-0253 (unsound) |
| materialized-state | sqlite | rusqlite 0.40.2 (`bundled`) | clean | |
| materialized-state | postgresql | server 16.15 (installed; PGDG `postgresql-16` 16.15-1.pgdg24.04+2 sha256 `abe560fc574383f520753dfaef79ab24081cda488293d5c991a9f77ac98ef18f`); clients tokio-postgres 0.7.18, postgres 0.19.14 | clean, obtainable | `initdb`/`postgres`/`psql` present in `/usr/lib/postgresql/16/bin` |
| relational-substrate | postgresql-18-pgvector | PostgreSQL 18.6 (PGDG `postgresql-18` 18.6-1.pgdg24.04+2 sha256 `c0773357df8769abb70e92722196d9b53422d71c0f29a4f39c834b94276cfa3d`; source `postgresql-18.6.tar.gz` sha256 `983ee554ec53dbeb9b70797bef9fcf4e67e117e7e48ca1463cc80b3ff8e8ff3f`) + pgvector 0.8.7 (`postgresql-18-pgvector` 0.8.7-1.pgdg24.04+1 sha256 `a4a428204f05aa18092ecbfc9eb04b2a7fb2d3cead796106b4d662bfe2ee49f1`); crate pgvector 0.4.2 | clean, obtainable | not installed. The distro PostgreSQL has no 18 and its `postgresql-16-pgvector` is 0.6.0 (no `halfvec`); PGDG (signed apt repo, reachable) has both |
| relational-substrate | postgresql-16-pgvector | PGDG `postgresql-16-pgvector` 0.8.7-1.pgdg24.04+1 sha256 `da584c72f1d84203734a59a773eeb64e0b2b4245393c3fe8eb3013b96ac29daa` | clean, obtainable | PGXN source alternative: `vector-0.8.7.zip` sha256 `cf09c09624f7fc9eb31560e48539799ee516030d89b22155c36f3cc2a86fc452` (sha1 equals the one PGXN publishes) |
| relational-substrate | turso-projection-only | turso 0.8.2 | clean | |
| analytics-mirror | ptr-analytics-metric-sql | in repo | clean | |
| analytics-mirror | datafusion | 55.2.0 | clean with `default-features = false` and without `compression` | default features add `bzip2 0.6.1` -> `libbz2-rs-sys 0.2.5`, licence **bzip2-1.0.6**, not in the allow list (permissive; an owner could instead allow it). Trimmed set: `nested_expressions, crypto_expressions, datetime_expressions, encoding_expressions, regex_expressions, string_expressions, unicode_expressions, parquet, recursive_protection, sql` is clean (`b3-min`) |
| analytics-mirror | iceberg-mirror | iceberg 0.10.1 | **needs-vendor-patch (new, cheap by precedent)** | `parquet 58.4.0` -> `paste 1.0.15`, RUSTSEC-2024-0436 (unmaintained). Class `paste-alias` (18 existing patches) applied to `parquet 58.4.0`; none carried for it. `parquet 59.3.0`, which datafusion uses, no longer depends on `paste`, so a later iceberg release may retire the need |
| analytics-mirror | pg-duckdb | n/a | **not-obtainable** (here) | source only. The GitHub tag archive answers 403 from this sandbox, there is no PGDG, PGXN or PyPI package, and the build embeds DuckDB (heavy). Stays conditional on a reachable source |

### B4 retrieval (spike `b4`, 618 packages)

| Component | Candidate | Version | Class | Detail |
|---|---|---|---|---|
| lexical-search | tantivy | 0.26.2 | clean | informational: `lru 0.16.4` RUSTSEC-2026-0253 (unsound) |
| lexical-search | custom-inverted-index | in repo | clean | |
| lexical-search | postgres-fts | built in | clean | |
| lexical-search | pg-textsearch-bm25 | PGDG `postgresql-18-pg-textsearch` 1.5.1-1.pgdg24.04+1 sha256 `5f3b2655705c177bbae9163df8d3fbbe388de75f11dd112c1586b01daaa35c23` (PG17: `7d3e631dd9c64516e7c1e3447eb7aa6642e9588d00c50cc212a452e02c03a322`) | obtainable (PostgreSQL 17/18 only) | PGDG has no PG16 build; evaluate on PG18 |
| lexical-search | paradedb-pg-search | PGXN `pg_search` 0.26.1 source `pg_search-0.26.1.zip` sha256 `4c32f3591fcc9417ba8ec0ae5606209d4f885a107a1daf4bc40dc209c7d08897` | obtainable as source, conditional | AGPL-3.0: external process only, never linked or vendored. Rust/pgrx build against PG: heavy |
| local-vector-search | hnsw-library | usearch 2.26.4; hnsw_rs 0.3.5 | clean | usearch builds C++ (`cc`) |
| local-vector-search | zvec | crate 0.1.0 | clean (crate); native library obtainable | bindings; `build.rs` fetches the Alibaba zvec library from the PyPI wheel `zvec-0.3.1-cp311-cp311-manylinux_2_28_x86_64.whl` sha256 `ee2febd1cb872c023eeeeba17bac3fa7471a93b486ac14fb4afca62161a502b7` (the crate checks a hash too) or from `ZVEC_ROOT` |
| local-vector-search | lancedb-embedded | 0.40.0 | **blocked** (licence decision) + needs-vendor-patch (new) | `option-ext 0.2.0` (via `dirs`) is **MPL-2.0** (only `colored` has an exception) and `lance-bitpacking 13.0.0` -> `paste 1.0.15`, RUSTSEC-2024-0436. Both would need to be solved: a `paste-alias` patch of `lance-bitpacking` (cheap by precedent) and an MPL exception like `colored`'s (owner decision, notice obligations) |
| local-vector-search | pgvector-halfvec-hnsw | pgvector 0.8.7 | clean, obtainable | as B3 |
| local-vector-search | vectorchord | PGXN `vchord` 1.1.1 source `vchord-1.1.1.zip` sha256 `024583c9cae05057f16f66778ca4276461fc2a028f2eb33a5cacff05a58f53f7` | obtainable as source, conditional | AGPL-3.0, external process only; pgrx build; depends on pgvector |
| structural-code-search | tree-sitter-custom | tree-sitter 0.27.1, tree-sitter-rust 0.24.2, `syn` 3.0.6 (`full`, `parsing`, `visit`) for the truth | clean | |
| structural-code-search | ripgrep-only-baseline | ripgrep 15.2.0 `x86_64-unknown-linux-musl` sha256 `33e15bcf1624b25cdd2a55813a47a2f95dbe126268203e76aa6a585d1e7b149c` (published `.sha256`); crates grep-searcher 0.1.17, grep-regex 0.1.14, grep-matcher 0.1.9, ignore 0.4.33 | clean | |
| structural-code-search | gritql | GitHub release `v0.1.0-alpha.1743007075`, `grit-x86_64-unknown-linux-gnu.tar.gz` (24423589 bytes) sha256 `94b34641a538ca0e85a92aa7f0ac94077fc6d663c996d0556c781d3d4c163149` (computed; no published checksum); npm `@getgrit/cli` same version, shasum `a958f97c9115d883e9263c4057b3e32d4b458ff6` | obtainable as an external binary only | the engine is not on crates.io (`marzano-*`/`grit-*` are absent), and a git dependency is refused by `deny.toml` (`unknown-git = "deny"`), so it cannot enter a bench lock; it can only be run as a process. Alpha release; no Rust graph to scan |

### B5 network, object storage, streaming (spike `b5` 585 packages; `b5-min`)

| Component | Candidate | Version | Class | Detail |
|---|---|---|---|---|
| network | iroh | 1.3.0 | **needs-vendor-patch (carried)** + feature trim | default features add `portmapper` -> `igd-next` -> `attohttpc 0.30.1`, **MPL-2.0**; `netwatch` -> `netlink-packet-core 0.8.2` -> `paste 1.0.15`, RUSTSEC-2024-0436. `ptr-net` already uses `default-features = false, features = ["tls-ring"]` and carries `vendor/netlink-packet-core-0.8.2`; `b5-min` with exactly that is clean |
| network | quinn | 0.11.12 | clean | |
| network | grpc | tonic 0.14.6, prost 0.14.4 | clean | |
| object-storage | opendal | 0.59.4 (`services-fs`, `services-memory`) | clean | incumbent family |
| object-storage | object-store-rs | object_store 0.14.2 | clean | |
| object-storage | direct-backends | in repo | clean | |
| object-storage | local S3 server | s3s 0.17.0, s3s-fs 0.17.0 | clean (crates) | `cargo install --locked s3s-fs@0.17.0` builds from source (checksum is in the lock); no release binary found |
| event-streaming | postgres-projection-event-log | in repo + PostgreSQL | clean | |
| event-streaming | iggy | client `iggy` 0.11.0 | clean (client); server conditional | no `iggy-server` crate and no discoverable binary release; Apache source release `iggy-0.9.0-src.tar.gz` sha512 `32e33c0a61df4c87f4f27864168fb5da6d1e22f81c4f1568375d3ec8414c6e9ec590133a5615260602c50334df6ecd2ccb4aed6cbf9c7d57cb96c62630a6e456` (release signature `.asc` published) must be built from source; the client (0.11) is newer than the released server (0.9.0): check protocol compatibility first |
| event-streaming | nats | async-nats 0.50.0; `nats-server` v2.15.1 `linux-amd64` | clean, obtainable | server tarball sha256 `bf5edc7b1ce98e885c368cb43a886eb148dc5d0551d99f3cf6141fb5e1954c70` (from the release's published `SHA256SUMS`) |
| event-streaming | redpanda-kafka | n/a | stays-open | container/cluster |


### B6 runtime (spike `b6`, 622 packages)

| Component | Candidate | Version | Class | Detail |
|---|---|---|---|---|
| execution-runtime | ptr-isolates | in repo | clean | incumbent |
| execution-runtime | tokio-custom | tokio 1.53.2 (`rt-multi-thread`, `macros`, `time`, `sync`) | clean | |
| execution-runtime | compio-runtime | compio 0.19.2 | clean (crate) | io_uring: `/proc/sys/kernel/io_uring_disabled` is 0 here; whether the sandbox seccomp profile lets `io_uring_setup` through is only known at run time |
| execution-runtime | tina-rs | n/a | **not-obtainable** (identity) | no crate `tina-rs`. The only match, crates.io `tina` 0.0.2 (2023, "Tina platform"), is an unrelated web/database framework (mysql, mongodb, nacos, grpc clients). Its graph carries 21 advisories (for example `h2 0.3.27` RUSTSEC-2026-0258, `ring 0.16.20` RUSTSEC-2025-0009, `quick-xml 0.25.0` RUSTSEC-2026-0194/0195, `rustls-webpki 0.101.7` RUSTSEC-2026-0098/0099/0104, `idna` RUSTSEC-2024-0421, `bincode 1.3.3`) and 7 licence rejections, so even if it were the one meant it is blocked. The owner should name the intended project |

### B7 ML on CPU (spike `b7`, 660 packages; `b7-patched`, 637)

| Component | Candidate | Version | Class | Detail |
|---|---|---|---|---|
| model-framework | candle | candle-core 0.11.0, candle-nn 0.11.0 | **needs-vendor-patch (carried)** | `gemm-f16 0.19.0` -> `paste 1.0.15`, RUSTSEC-2024-0436. The repository carries `gemm`, `gemm-c32/c64/common/f16/f32/f64` 0.19.0 and `pulp` 0.22.3 as `paste-alias` patches; with the root `[patch.crates-io]` entries copied, `b7-patched` is clean for candle-core and candle-nn (and for `tokenizers 0.22.2`, the only version carried; `0.23.2` also depends on `paste`) |
| model-framework | burn-0.18-ndarray-a0 | A0 workspace: burn 0.22.0-pre.3 + `burn-ndarray` | clean (incumbent) | `model/burn-a0/Cargo.lock` with its carried `paste-alias` patches is audited by `security.yml`. The newer stable `burn` 0.22.0 + ndarray still reaches `paste 1.0.15` through seven packages that the A0 patches only at their pre-release versions: `burn-dispatch` 0.22.0, `burn-ndarray` 0.22.0, `cubecl-core/-cpp/-hip/-std` 0.11.0 and `macerator` 0.5.1 (`b7-patched` still flags them). Moving the A0 candidate forward needs the same alias patch for those seven (**new, moderate**: one class, seven packages) |
| model-framework | pytorch-reference | torch 2.14.1 | clean, obtainable (not scanned) | pinned by `training/uv.lock` (PyPI hashes there); CPU wheel `torch-2.14.1+cpu-cp311-cp311-manylinux_2_28_x86_64.whl` sha256 `5e38154c8896d426a5df58bf2276b603e82184dfae4f54d77dbb212f36e34f37` from `download.pytorch.org/whl/cpu`. No Python advisory scan was run |
| model-framework | burn-cubecl | n/a | stays-open | GPU path |
| ingress-classifier | gliclass-rs | 1.0.0 | **blocked** | `number_prefix 0.4.0` RUSTSEC-2025-0119 (unmaintained, via `tokenizers 0.21.4` -> `indicatif`), `paste 1.0.15` RUSTSEC-2024-0436 (via `tokenizers 0.21.4`; the repository carries only `tokenizers 0.22.2`), and `option-ext 0.2.0` MPL-2.0 (via `hf-hub` -> `dirs`). The crate pins `tokenizers 0.21.4` and `ort =2.0.0-rc.9`, so none of it can be trimmed without forking the crate. Model, if it is ever unblocked: HF `knowledgator/gliclass-small-v1.0` @ `21edefaf7951f68c68c505f9139ba536d3b448f7`, `onnx/model.onnx` sha256 `01ed6e5f0c6e813dfebc10be9b6d6b0f0511ff3c5c8861209f6443663ed19ac5`, `onnx/model-int8-quantized.onnx` sha256 `825c1f9231ec43d7acab7f21ef0b5954a06aaf3cf5660511883a2775e2459bb8` |
| ingress-classifier | burn-small-classifier | in repo | clean | A0 stack |
| ingress-classifier | llm-only-baseline | n/a | stays-open | no model or runtime is pinned (human choice). A pure-Rust route through candle (patched, above) would add no new external binary |
| label-model | ptr-labeling-dawid-skene, majority-vote | in repo | clean | |
| label-model | snorkel-label-model | snorkel 0.10.0 (Apache-2.0, Python >= 3.11) | obtainable (not scanned) | `snorkel-0.10.0-py3-none-any.whl` sha256 `42a2f354d8142f4807c5339ed256c763e12dd198ca5a34583d27344e64f927c8`, sdist sha256 `b1f36412430a810124cc0aac432fa29f41960cff2d9804d3816f9d9d0d6d44ed`. Requires torch, tensorboard, pandas, scikit-learn, scipy, networkx, munkres. No Python advisory scan was run |
| fast-weight-memory | ptr-fastmem-gated-delta, recency-buffer, hybrid-retrieval-only | in repo | clean | no external candidate |

### B8 verification and security (spike `b8`, 71 packages)

| Component | Candidate | Version | Class | Detail |
|---|---|---|---|---|
| code-quality-verifier | compiler-tests-only | n/a | clean | baseline |
| code-quality-verifier | custom-static-metrics | `syn` 3.0.6, `quote` 1.0.47, `proc-macro2` 1.0.107 | clean | `rust-code-analysis 0.0.25` was spiked as a possible engine and is **blocked**: it is itself MPL-2.0 and pulls `fxhash 0.2.1` (RUSTSEC-2025-0057) |
| code-quality-verifier | sentrux | PyPI `sentrux` 0.1.6 (MIT) | **not-obtainable** (the Rust-capable one) / stays-open | no crate, no npm package. The PyPI package (`sentrux-0.1.6-py3-none-any.whl` sha256 `449fd9d542305639da567c86db2bc3f15bbab39d8df8c2337452d8a9908d349d`, sdist sha256 `6c086cc41c4049039cc76d85d65b08386dbcf70ccc438f4117957a4e46e0ef75`) is a pure-Python AST analyser described as "structural quality analysis for Python projects"; it cannot analyse Rust, so it is not a candidate for PTR code. The owner should identify the tool meant |
| security-context | ptr-policy | in repo | clean | |
| security-context | custom-scanners | regex 1.13.1, aho-corasick 1.1.5 | clean | |
| security-context | brin-signal | n/a | stays-open | an external service, unidentified |

## Components not spiked

| Component | Why |
|---|---|
| gpu-vector-search (cuvs, faiss-gpu) | needs a GPU |
| inference-serving, adapter-serving (sglang, vllm, tensorrt-llm, lorax) | need a GPU (`ptr-native` alone is not a comparison) |
| training-backend (axolotl, FSDP, LOMO; trl-peft CPU smoke) | need GPUs |
| distributed-data-compute, distributed-search | need an external cluster |
| environment-runtime | needs container orchestration and an LLM policy |
| rust-editor-emacs | human judgement; no emacs here |

## Licence-policy decisions the spike surfaced

Every item below is permissive or weak-copyleft and is rejected only because it is not in the
`deny.toml` allow list. None is a patch; each is an owner decision that would unblock the named
candidates. Adding any of them also adds notice obligations (`scripts/generate_notices.py`).

| Licence | Package (via) | Candidates it blocks | Avoidable by trimming? |
|---|---|---|---|
| MIT-0 | `borrow-or-share 0.2.4` (nemo-relay -> jsonschema -> referencing -> fluent-uri) | nemo-relay | no (non-optional) |
| bzip2-1.0.6 | `libbz2-rs-sys 0.2.5` (datafusion `compression`) | datafusion default features | yes: drop `compression` |
| MPL-2.0 | `option-ext 0.2.0` (dirs) | lancedb, gliclass-rs | no |
| MPL-2.0 | `attohttpc 0.30.1` (iroh `portmapper`) | iroh default features | yes: `default-features = false` |
| MPL-2.0 | `smartstring 1.0.1` (raft-engine `scripting`) | raft-engine default features | yes: `default-features = false` |
| MPL-2.0 | `rust-code-analysis 0.0.25` | itself | no |
| 0BSD, MPL-2.0, unlicensed | graph of crates.io `tina` | tina (not the intended project) | n/a |

## Obtainability of non-Rust artifacts

| Artifact | Pinned version | Source | Checksum | State |
|---|---|---|---|---|
| cargo-deny | 0.20.2 | GitHub release | sha256 `9f12ed4c...c8d5f` (full value above) | installed in the session tools directory, not in the repository |
| cargo-audit | 0.22.2 | GitHub release | sha256 `7fb9497f...c428` (full value above) | installed likewise |
| PostgreSQL 16 | 16.15 | distro (installed); PGDG 16.15-1 | see B3 | present |
| PostgreSQL 18 | 18.6 | PGDG apt (signed) / `ftp.postgresql.org` source | see B3 | obtainable, not installed |
| pgvector | 0.8.7 | PGDG apt / PGXN | see B3 | obtainable |
| pg_textsearch | 1.5.1 | PGDG apt (PG17, PG18) | see B4 | obtainable |
| pg_search (ParadeDB, AGPL-3.0) | 0.26.1 | PGXN source | see B4 | source only |
| VectorChord `vchord` (AGPL-3.0) | 1.1.1 | PGXN source | see B4 | source only |
| pg_duckdb | none found | GitHub source archive answers 403 here | none | not obtainable here |
| nats-server | v2.15.1 | GitHub release | see B5 | obtainable |
| Apache Iggy server | 0.9.0 | `downloads.apache.org` source | see B5 | source only; build |
| capnproto, protobuf-compiler | 1.0.1-4, 3.21.12-8.2ubuntu0.3 | Ubuntu apt | see B1 | obtainable |
| ripgrep | 15.2.0 | GitHub release | see B4 | obtainable |
| GritQL `grit` | alpha.1743007075 | GitHub release / npm | see B4 | obtainable, binary only |
| zvec native library | 0.3.1 | PyPI wheel | see B4 | obtainable |
| torch (CPU) | 2.14.1 | `download.pytorch.org/whl/cpu`; `training/uv.lock` | see B7 | obtainable |
| snorkel | 0.10.0 | PyPI | see B7 | obtainable |
| GLiClass model | `21edefaf...` | Hugging Face | see B7 | obtainable (unneeded: gliclass-rs is blocked) |
| redpanda, containers, GPUs | | | | stays-open |

GitHub's API and tag-archive downloads answer 403 in this sandbox (release assets and the Go,
npm, PyPI, PGDG, PGXN and Apache mirrors work), so versions were discovered through those
mirrors, not through GitHub release lists. Checksums marked "published" were fetched from the
publisher; the others were computed by downloading the file and are trust-on-first-use.

## Summary

| Class | Count of external candidates |
|---|---|
| clean (as stated, including the feature trims) | tikv-jemallocator, mimalloc, rkyv, postcard\*, prost, capnproto, valuable, tracing, opentelemetry, salsa, rocksdb, redb, openraft, turso, sqlite, postgresql 16/18 (+pgvector), datafusion\*, tantivy, usearch/hnsw_rs, zvec, pgvector-halfvec, tree-sitter-custom, ripgrep baseline, tokio, compio, quinn, grpc, opendal, object_store, S3 server, nats, iggy client, pytorch, snorkel, custom-scanners, custom-static-metrics (\* only with the stated features) |
| needs-vendor-patch, carried | raft-engine, raft-rs, iroh, candle |
| needs-vendor-patch, new | iceberg (one package, `parquet 58.4.0`: cheap by precedent), burn 0.22.0 (seven packages: moderate) |
| blocked | bincode (RUSTSEC-2025-0141), gliclass-rs (RUSTSEC-2025-0119, MPL-2.0), lancedb (MPL-2.0 and paste), nemo-relay (MIT-0), tina (not the intended project) |
| not-obtainable | serde-json-tree (no such crate), tina-rs (unidentified), pg-duckdb (here), sentrux (the Python one cannot analyse Rust) |
| conditional (obtainable, heavy or source-only) | pg-textsearch (PG18), pg_search, vchord, iggy server, gritql (binary only) |
| stays-open | redpanda-kafka, burn-cubecl, llm-only-baseline, brin-signal, and the eight non-CPU components |

## Reproduction

Everything was done in `scratchpad/evals-spike/` (outside the repository; kept: manifests, lockfiles,
`audit.json`, `deny.json`, `analysis.json` per batch, `gen.py`, `rundeny.sh`, `analyze.py`,
`reach.py`) and `scratchpad/tools/` (the two binaries and the advisory-db checkout). Per batch:

```bash
cd evals-spike/<batch> && cargo +1.99.0 generate-lockfile
cargo-audit audit --file Cargo.lock --db ../../tools/advisory-db --no-fetch --json > audit.json
cargo-deny --format json --manifest-path Cargo.toml --config <repo>/deny.toml check advisories licenses sources 2> deny.json
```

Re-run the spike on each real bench workspace when its lock is created; this document is the
input to choosing what goes in, not a replacement for the CI scans (`security.yml` covers every
`evaluations/bench/*` workspace that has a `Cargo.lock`).
