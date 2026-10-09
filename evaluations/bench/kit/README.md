# ptr-evalkit

The shared measurement kit of the component evaluations (WP-E0). Std only: it adds no
package to the graph of any benchmark workspace that depends on it, so a candidate's
advisory or licence verdict is never caused by the kit.

It lives in its own **excluded workspace** (`evaluations/bench/*` is in the root
`exclude`, like `fuzz`, `model/burn-a0` and `vendor/*`) with its own `Cargo.lock`, so
heavy candidate stacks (datafusion, lancedb, rocksdb, tantivy, openraft) never enter the
root lockfile, the root `cargo deny` graph, MSRV 1.99 or `cargo check --workspace`.
`scripts/security_workspaces.py` finds the lock automatically; CI and `make fmt` name
the manifest explicitly (see below).

```text
 bench workspace (evaluations/bench/<batch>/)          component dir
 ┌──────────────────────────────────────────┐   ┌────────────────────────────────┐
 │ adapters: one module per candidate       │   │ protocol.toml   (frozen first) │
 │ main: gates, then timed workload         │   │ candidates.toml (status, note) │
 │        │ uses                            │   │ evidence/*.json (run records)  │
 │        ▼                                 │   └──────────────▲─────────────────┘
 │  ptr-evalkit (this crate)  ──────────────┼── EvidenceRecord ─┘  names sha256(protocol.toml)
 │  rng gen stats rss envcap evidence       │            checked by scripts/check_component_protocols.py
 └──────────────────────────────────────────┘
```

| Module | What it gives | Rule it encodes |
|---|---|---|
| `rng` | SplitMix64, copied from `bins/ptr-bench/src/experiments/rng.rs` (checked against the reference stream) | a seed reproduces its stream on every machine and toolchain |
| `gen` | keys, payloads, permutations, Gaussian, Zipf, Zipf corpora, Gaussian mixtures | every workload is a pure function of a seed |
| `stats` | nearest-rank quantiles on integer basis points, `Summary::latency` (refuses < 10 000 samples), `Repetitions` (warmup, at least 5 measured), `median` | p99 is an observed order statistic of at least 100 samples; the reported value is the median of the measured runs |
| `rss` | `VmHWM` / `VmRSS` from `/proc/self/status` | absent on non-Linux: record the metric as missing, never 0 |
| `envcap` | git sha and dirty flag, `rustc -V`, CPU model, threads, memory, kernel, filesystem type (`stat -f`), cpufreq governor, decision grade | tmpfs/overlay-like filesystems and non-decision-grade hardware profiles make durable-write numbers "not decision-grade" |
| `evidence` | `EvidenceRecord`, `BenchSection`, `RECORD_FIELDS` | exactly the fields `scripts/run_component_eval.py` stores, then `protocol_sha256` and `bench` |
| `json`, `sha256`, `clock` | writer/parser, SHA-256, UTC stamps in the runner's formats | no dependency just for evidence |

The binary `ptr-evalkit` prints the captured environment (`env`), a measured hardware
profile (`hardware ID [DIR]`), a file hash (`sha256`), and checks a record
(`validate FILE`). `examples/smoke_record.rs` is the template for a batch `main`.

## Tests and checks

```bash
cargo test   --manifest-path evaluations/bench/kit/Cargo.toml --locked
cargo clippy --manifest-path evaluations/bench/kit/Cargo.toml --all-targets --locked -- -D warnings
cargo fmt    --manifest-path evaluations/bench/kit/Cargo.toml -- --check
make bench-check        # all of the above plus cargo check
```

Unit tests sit beside the code (quantiles against hand-computed values, generator
determinism, hash vectors, escape/parse round trips); `tests/kit_api.rs` goes through the
public API only.

## Adding a bench workspace

1. `evaluations/bench/<batch>/` with its own `Cargo.toml` (`[workspace]`, path
   dependency on `../kit`), committed `Cargo.lock`, `README.md`, `config.toml` plus
   `tests/`. `check_repo.py` requires the sibling `tests/` of every `config.toml`.
2. If it depends on product crates with the vendored Raft packages, copy the root
   `[patch.crates-io]` entries (excluded workspaces do not inherit them);
   `model/burn-a0/Cargo.toml` is the precedent.
3. Only candidates that `evaluations/SPIKE.md` classifies as clean (or with a cheap
   vendor patch that is carried) enter the manifest. No advisory ignores.
4. Add the manifest to `make fmt`, `make bench-check` and the bench steps of CI's
   `quality` job (`Makefile` `ci-quality` mirrors them; `scripts/tests/test_ci_local.py`
   fails otherwise).
5. Regenerate `THIRD-PARTY-NOTICES.md` (`python3 scripts/generate_notices.py`, after
   `cargo fetch --locked` in the workspace) so `scripts/check_notices.py` stays green.
6. Freeze the component's `protocol.toml` in a commit **before** the first evidence run
   (`evaluations/PROTOCOL_FORMAT.md`).
