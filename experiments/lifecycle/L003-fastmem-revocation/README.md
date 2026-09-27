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
**Stale since b397cb5:** the counts below describe the code at 7b60216, and
[`results/STALE.toml`](results/STALE.toml) says so for `scripts/check_research_gates.py` until all five
seeds and the mutation check are rerun at the commit that merges the change to the execution layer.

Completed on 2026-09-27 at 7b60216 (seed records, mutation check and aggregation all at that commit):
**hard pass**, 5 seeds of 30 cases on
PostgreSQL 18.6 (Ubuntu 18.6-1.pgdg24.04+2) with pgvector 0.8.6 and
`fsync`, `synchronous_commit` and `full_page_writes` on.

- 292 fast memories, 46,408 acknowledged writes and
  6,484 revocations or supersessions that removed writes
  (40,069 writes removed). After every record the refolded state, the journal
  PostgreSQL kept, every stored checkpoint and the newest one handed out were compared with the
  never-saw-it fold: 28,673 bit-identity and readout checks,
  25,707 journal checks, 5,590 checkpoints verified and
  1,572 stale checkpoints planted in storage.
- 998 process crashes, 953 crashes mid-append
  (556 committed), 1,017 mid-revocation
  (536 committed, 481 rolled back),
  660 appends whose commit the server failed, 927 appends and
  729 checkpoints racing a revocation; every seed saw both outcomes of each.
- No bit difference, no resurrected read, no stale checkpoint handed out, no acknowledged write lost and
  no inadmissible append accepted. A read between a change's commit and the process's refold was denied
  for each of the 6,484 changes that removed writes, and no admissible read was denied.
- A refold replayed 11.57 writes per revocation on average,
  67% of what folding every remaining write from scratch
  would have replayed.
- Mutation check: 16 of 16 planted defects in the projector's cascade, the journal and
  checkpoint store and `ptr-fastmem` detected, each through the counter that shows it
  ([`results/mutations.json`](results/mutations.json)).
- Earlier runs at ad2f8d1 (2026-09-26) and dcfbb3e (2026-09-27) also passed; their seed records stay in
  `results/` beside these, and the aggregate uses the newest record of every seed.

Per-seed counters: [`results/metrics.json`](results/metrics.json); scope, limitations and the run records
used: [`results/run.json`](results/run.json).

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
