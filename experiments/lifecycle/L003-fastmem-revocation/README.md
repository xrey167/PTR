# L003-fastmem-revocation

## Hypothesis
Can a revoked input ever influence a fast-memory readout after revocation, replay, restore or crash?

## Primary metrics
bit-identity failures; resurrected reads; writes replayed per revocation

## Baseline
memory rebuilt from scratch without the revoked writes

## Falsification
Any bit difference from the never-saw-it fold, or any admitted read that depends on a revoked input, is a hard failure.

## Design
Mechanism and threat model: [35 — Agentic substrate](../../../docs/architecture/35-agentic-substrate.md).

## Result
Completed on 2026-09-27 at ed52931 (seed records, mutation check and aggregation all at that commit):
**hard pass**, 5 seeds of 30 cases on
PostgreSQL 18.6 (Ubuntu 18.6-1.pgdg24.04+2) with pgvector 0.8.6 and
`fsync`, `synchronous_commit` and `full_page_writes` on.

- 292 fast memories, 45,278 acknowledged writes and
  6,473 revocations or supersessions that removed writes
  (38,518 writes removed). After every record the refolded state, the journal
  PostgreSQL kept, every stored checkpoint and the newest one handed out were compared with the
  never-saw-it fold: 28,854 bit-identity and readout checks,
  25,835 journal checks, 5,927 checkpoints verified and
  1,584 stale checkpoints planted in storage.
- 1,016 process crashes, 993 crashes mid-append
  (591 committed), 993 mid-revocation
  (520 committed, 473 rolled back),
  643 appends whose commit the server failed, 968 appends and
  718 checkpoints racing a revocation; every seed saw both outcomes of each.
- No bit difference, no resurrected read, no stale checkpoint handed out, no acknowledged write lost and
  no inadmissible append accepted. A read between a change's commit and the process's refold was denied
  for each of the 6,473 changes that removed writes, and no admissible read was denied.
- A refold replayed 11.06 writes per revocation on average,
  68% of what folding every remaining write from scratch
  would have replayed.
- Mutation check: 16 of 16 planted defects in the projector's cascade, the journal and
  checkpoint store and `ptr-fastmem` detected, each through the counter that shows it
  ([`results/mutations.json`](results/mutations.json)).
- Earlier runs at ad2f8d1 (2026-09-26), dcfbb3e and 7b60216 (2026-09-27) also passed; their seed records stay in
  `results/` beside these, and the aggregate uses the newest record of every seed.

Per-seed counters: [`results/metrics.json`](results/metrics.json); scope, limitations and the run records
used: [`results/run.json`](results/run.json).

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
