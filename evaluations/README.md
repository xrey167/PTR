# Component Evaluations

This area compares implementation technologies for a fixed PTR architectural contract.

A component evaluation is not a model novelty experiment. It asks questions such as “which local vector engine best satisfies ptr-search?” or “which runtime substrate satisfies ptr-exec?”.

Every candidate should record:
- version/commit;
- correctness;
- failure semantics;
- p50/p95/p99;
- throughput and memory;
- startup/recovery;
- portability/operations;
- licensing;
- integration complexity;
- benchmark scripts and raw evidence.

See `registry.toml` for open slots.


## Runner

Candidates with status `evaluating` must declare a versioned `command` in their `candidates.toml`.

```bash
python scripts/run_component_eval.py validate
python scripts/run_component_eval.py run ledger raft-engine
python scripts/run_component_eval.py run network iroh
```

Execution uses argv directly rather than a shell. Each run writes a unique evidence record with the candidate manifest hash, repository commit, exact command, duration, exit status, stdout/stderr and launch errors. A failed command is persisted as failed evidence and never converted into a positive evaluation.
