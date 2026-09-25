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

`ledger-recovery` and `ledger-process-crash` exit with status 1 when any of their
hard-invariant counters (`false_accepts`, `recovery_errors`, `tail_trim_errors`,
and for the process probe `child_exit_errors`) is nonzero, after printing the
JSON line. `scripts/run_experiment.py` records a run as `completed` from the exit
status, so a violating run is now recorded as `failed` with its counters intact.
