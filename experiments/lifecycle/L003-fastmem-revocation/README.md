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
Completed on 2026-09-27 at 5f71d84 (seed records, mutation check and aggregation all at that commit):
**hard pass**, 5 seeds of 30 cases on
PostgreSQL 18.6 (Ubuntu 18.6-1.pgdg24.04+2) with pgvector 0.8.6 and
`fsync`, `synchronous_commit` and `full_page_writes` on.

- 292 fast memories, 45,250 acknowledged writes and
  6,517 revocations or supersessions that removed writes
  (39,514 writes removed). After every record the refolded state, the journal
  PostgreSQL kept, every stored checkpoint and the newest one handed out were compared with the
  never-saw-it fold: 28,724 bit-identity and readout checks,
  25,635 journal checks, 5,751 checkpoints verified and
  1,626 stale checkpoints planted in storage.
- 1,017 process crashes, 972 crashes mid-append
  (584 committed), 978 mid-revocation
  (504 committed, 474 rolled back),
  631 appends whose commit the server failed, 974 appends and
  720 checkpoints racing a revocation; every seed saw both outcomes of each.
- No bit difference, no resurrected read, no stale checkpoint handed out, no acknowledged write lost and
  no inadmissible append accepted. A read between a change's commit and the process's refold was denied
  for each of the 6,517 changes that removed writes, and no admissible read was denied.
- A refold replayed 11.49 writes per revocation on average,
  66% of what folding every remaining write from scratch
  would have replayed.
- Mutation check: 16 of 16 planted defects in the projector's cascade, the journal and
  checkpoint store and `ptr-fastmem` detected, each through the counter that shows it
  ([`results/mutations.json`](results/mutations.json)). Each is now built alone: until ff96ce0 the checker
  restored a planted file with its old modification time, so cargo kept `read-admission-skipped` linked
  while the three `ptr-pg` defects after it ran, and the records at dcfbb3e, 7b60216 and ed52931 show its
  resurrected reads beside their own counters (doc 35 §8). This record detects them alone:
  `append-commit-error-ignored` by commit fault failures (50); `latest-checkpoint-oldest-first` by checkpoint
  violations (39); `checkpoint-cascade-deletes-all` by checkpoint violations (38), checkpoints lost (57).
- Earlier runs at ad2f8d1 (2026-09-26), dcfbb3e, 7b60216, ed52931 and ff96ce0 (2026-09-27) also passed; their seed records stay in
  `results/` beside these, and the aggregate uses the newest record of every seed.

Per-seed counters: [`results/metrics.json`](results/metrics.json); scope, limitations and the run records
used: [`results/run.json`](results/run.json).

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
