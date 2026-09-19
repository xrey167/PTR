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
