# Performance Architecture

PTR optimizes the full path rather than only model tokens.

## Budgets

Track:
- model prefill/decode;
- latent reasoning steps;
- semantic invalidation/recompute;
- search;
- Pod compute;
- verifier cost;
- queue time;
- serialization/copies;
- consensus/materialization for writes.

## Tooling

Criterion, tracing spans, flamegraphs/perf, allocator statistics, assembly/LLVM inspection and GPU profilers.

## Rules

- benchmark system allocator vs jemalloc rather than assuming;
- do not introduce exotic buffers/async runtimes without a measured bottleneck;
- preserve safety checks in benchmark builds;
- report end-to-end p95/p99 and token/tool counts, not only microbenchmark throughput.
