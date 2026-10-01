# E001-end-to-end

## Hypothesis
PTR full pipeline versus unmodified backbone.

## Primary metrics
task success; hard violations; latency; tokens

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.

## Deterministic PR smoke

The repository-level E001 plumbing smoke is the durable HTTP path in
`ptr-server`. It exercises the scenario backend, project-scoped routing,
verification, the `demo-note` mutation, idempotent replay, restart fencing and
explicit reconciliation:

```text
cargo test -p ptr-server --test smoke -- --nocapture
```

This is system-semantics evidence only. It is not an M001/M002 model-quality
run and must not be aggregated as matched-transformer evidence. The full E001
experiment remains planned until a commit-bound runner emits the frozen
`run.json`/`metrics.json` artifacts.
