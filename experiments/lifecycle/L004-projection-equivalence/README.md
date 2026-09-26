# L004-projection-equivalence

## Hypothesis
Is a PostgreSQL projection rebuilt by replay always equal to the reference projection, and does it refuse every foreign or rolled-back history?

## Primary metrics
state and lifecycle divergence count; foreign histories accepted; rebuild time per 100k commits

## Baseline
ptr-state MaterializedState reference and the Turso backend

## Falsification
Any divergence from the reference or any accepted foreign or rolled-back history is a hard failure.

## Design
Mechanism and threat model: [35 — Agentic substrate](../../../docs/architecture/35-agentic-substrate.md).

## Result
**Stale since fd25359:** the counts below describe the code at ad2f8d1 (records c897127..0b54173), and
[`results/STALE.toml`](results/STALE.toml) says so for `scripts/check_research_gates.py` until all five
seeds and the mutation check are rerun at the commit integrating the later changes.

Completed on 2026-09-26: **hard pass**, 5 seeds of 40 cases on
PostgreSQL 18.6 (Ubuntu 18.6-1.pgdg24.04+2) with pgvector 0.8.6 and
`fsync`, `synchronous_commit` and `full_page_writes` on, with Turso as a third implementation.

- 19,587 records projected one by one with 1,180 crashes
  (927 found committed, 253 rolled back and re-sent),
  804 commits the server failed after every statement succeeded,
  1,031 redeliveries, 542 gaps and 705 records
  applied by two projectors at once; 1,797 forked-history probes,
  200 catch-ups from an older backup, rebuilds onto an unrelated and onto the true
  ledger in every case, and 276,530 lifecycle checks against the runtime's
  `generation_validity`.
- No divergence from the runtime replay, the `MaterializedState` reference or Turso; no foreign or
  rolled-back history accepted; no record reported applied that had not committed.
- Rebuild: 311.8 s per 100k commits, one transaction per commit, from the
  2,000-commit replay of each seed (a shared cloud container, not
  a capacity figure).
- Mutation check: 17 of 17 planted projector defects detected, each through the counter
  that shows it ([`results/mutations.json`](results/mutations.json)).
- Before this run the harness found a real defect: a redelivered record carrying the true anchor but other
  content was accepted as a duplicate. The projector now also recomputes the anchor from the record itself,
  with a regression test in `crates/ptr-pg/tests/postgres.rs`.

Per-seed counters: [`results/metrics.json`](results/metrics.json); scope, limitations and the run records
used: [`results/run.json`](results/run.json).

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
