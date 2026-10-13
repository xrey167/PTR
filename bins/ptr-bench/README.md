# ptr-bench

Reference microbenchmarks and lifecycle probes. Run `cargo run -p ptr-bench -- all`.
These probes do not establish model quality or production readiness.

Dependency setup now uses SemanticDelta.dependencies and timed updates use the
fallible typed-value/staged invalidation API. Historical timings of the old
in-place string map are not equivalent baselines. Measure the exact revision.
See [semantic durability](../../docs/architecture/22-durable-semantic-state.md).

The recovery harness now generates real partial PTRLOG02 frames and requires an
anchor derived from its independently known fixture. It does not derive trust
from the damaged file. Ordinary open rejects incomplete tails without rewriting.
Old v1 recovery timings are not comparable to v2 integrity/anchor validation.

## KV reuse smoke

`cargo run --release -p ptr-bench -- kv-reuse-smoke [requests=8]` runs `requests` prompts
that share a 64-token prefix on `ptr_runtime::ReferenceKvReuseBackend` (the seeded random
R005 tiny model on the CPU, KV reused through the runtime prefix catalog), each once with
reuse and once cold, and prints one JSON line `{requests, hit_tokens_total, cold_ms,
warm_ms, bitwise_equal}`; it exits 1 if any pair differs. **This is not an R005 result**:
it uses none of R005's preregistered parameters, writes no result record and measures
nothing about a real model.

## R005 exact prefix reuse

`cargo run --release --locked -p ptr-bench -- exact-prefix-reuse <seed>` runs the harness of
[R005](../../experiments/runtime/R005-exact-prefix-reuse/README.md), Exact half, for one seed
and prints exactly one JSON object. It is the experiment's entrypoint: a run is one seed's
evidence only when `scripts/run_experiment.py` runs it from the frozen commit and `aggregate.py`
judges it under `criteria.toml`; a direct run writes no record and is a **dry run, not a result**. The seed is required (there is no default and a typo
stops the run). A run takes about two minutes in a release build.

It drives `ReferenceKvReuseBackend` and `KvSession` over the seeded tiny Qwen2 stack, with
`PtrRuntime` as the ledger-anchored liveness oracle wherever the configuration asks for
committed-ledger liveness, and a bare `KvPrefixCatalog` for the dependency kinds the stack
backend does not carry (base model, adapter, expert), the forged witnesses and the positions.
The object holds every metric of the manifest (a number, an object, a digest, or `null` with
its reason under `unsupported`), the procedure, the model pins (checked from outside by
reproducing the backend's chain id and a cold run), the producer environment, the corpus
selection digests, the 52-cell invalidation matrix with the layer (`session` or `catalog`)
that carried each cell, and two judgements: `coverage` (a prerequisite below its minimum makes
the run vacuous) and `falsifiers` (a tripped one falsifies R005 on that seed). The exit status
is 1 when a prerequisite is unmet, a falsifier is tripped or a metric has neither a value nor a
reason, and 2 when the harness cannot run a case. The constants of `r005/params.rs` mirror
`config.toml` (a drift test fails on any difference); the values the file leaves open are listed
in `r005/params.rs` under "chosen" and in the preregistration.

## S003 certified branches

`cargo run --release -p ptr-bench -- certified-branches [cases=48] [seed=17]` runs the
harness of [S003](../../experiments/semdb/S003-certified-branches/README.md). It needs no
PostgreSQL and is built in every configuration: it drives a real `PtrRuntime` in memory,
beside an independent reference model, under five arms (serial, certified,
certified-review, last-writer-wins, key-level OCC) at four agent counts and six contention
levels, and runs twenty-six adversarial probes and nine canaries in every case. Its
parameters are the preregistered table in the experiment's `config.toml`, echoed in the
result. It prints one JSON line on stdout, progress and notes on stderr, and exits 1 on any
hard failure. A case takes seconds at the small contention levels and about a minute at the
largest state, in a release build; a debug build is much slower.

## S001 invalidation

`cargo run --release -p ptr-bench -- invalidation <seed>` runs the harness of
[S001](../../experiments/semdb/S001-invalidation/README.md) once for one seed. It needs no
PostgreSQL and is built in every configuration: it drives a real `ptr_semdb::SemanticHost`
(`prepare_delta` and its affected set) beside an independent oracle (no code shared with
`DependencyGraph`) and a global recomputation, over seeded DAGs of 256, 1024 and 4096 keys of five
topology classes, under update streams of leaf and hub edits, rewiring, key removal, no-op writes,
mixed batches, cycle-creating attempts and MissingDependency probes. The recomputation is a
harness-supplied derivation (the crate has no query engine). It prints one JSON line on stdout
(integer metrics, `timing_ns` apart), exits 1 when a coverage prerequisite is unmet or a falsifier
trips, and 2 when the harness itself cannot run (a generated delta the oracle contradicts, an error only a wrong harness delta can cause, a negative control that does not trip); any other error of the host, also while a graph is loaded, is a counted refusal that falsifies.
`invalidation <operations> <seed>` is the form the mutation checker uses (the pinned sizes with
that many operations per graph; the result says `"scale":"custom"` and is no evidence). A seed takes
about ten seconds in a release build.

`ledger-recovery` and `ledger-process-crash` exit with status 1 when any of their
hard-invariant counters (`false_accepts`, `recovery_errors`, `tail_trim_errors`,
and for the process probe `child_exit_errors`) is nonzero, after printing the
JSON line. `scripts/run_experiment.py` records a run as `completed` from the exit
status, so a violating run is now recorded as `failed` with its counters intact.

## PostgreSQL experiment harnesses

With the `postgres-experiments` feature the binary also runs the harnesses of
[L003](../../experiments/lifecycle/L003-fastmem-revocation/README.md) and
[L004](../../experiments/lifecycle/L004-projection-equivalence/README.md) against a
loopback PostgreSQL 16+ server with pgvector 0.8+ named by `PTR_PG_EXPERIMENT_DSN`:

```sh
export PTR_PG_EXPERIMENT_DSN="host=127.0.0.1 port=5432 user=postgres"
cargo run --release -p ptr-bench --features postgres-experiments -- fastmem-revocation 30 17
cargo +stable run --release -p ptr-bench --features turso-oracle -- projection-equivalence 40 17
```

Each prints one JSON line and exits 1 on any hard failure. Every case gets its own
schemas, tagged with the case so a crash can terminate exactly its sessions, and
drops them when it ends. The feature stays out of default builds, so the default
binary keeps its dependencies and toolchain.
