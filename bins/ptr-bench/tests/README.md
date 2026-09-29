# Tests for bins/ptr-bench

- `smoke.rs` runs the binary: the semdb and mailbox microbenchmarks, the ledger recovery
  smoke, and `certified-branches 1 17`, one case of the S003 harness in memory, which must
  print one JSON line, exit 0 and count nothing wrong. It builds with default features, so
  it runs in every CI job that tests the workspace.
- The S003 harness carries its own unit tests in `src/experiments/s003/` (the compiled
  preregistration against the frozen table, the reference model against the previous
  implementation of its delta rule on random deltas, the oracle and its canaries, the
  verifiers, the workload, the world that checks the runtime against the model, the arms and
  the probes). They run with `cargo test -p ptr-bench`.
- The PostgreSQL harnesses (L003, L004) are exercised by their experiments' own checks; they
  need a loopback server and the `postgres-experiments` feature.
