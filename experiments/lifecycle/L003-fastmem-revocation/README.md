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
**Stale since fd25359:** the counts below describe the code at ad2f8d1 (records d376d54..e420262), and
[`results/STALE.toml`](results/STALE.toml) says so for `scripts/check_research_gates.py` until all five
seeds and the mutation check are rerun at the commit integrating the later changes.

Completed on 2026-09-26: **hard pass**, 5 seeds of 30 cases on
PostgreSQL 18.6 (Ubuntu 18.6-1.pgdg24.04+2) with pgvector 0.8.6 and
`fsync`, `synchronous_commit` and `full_page_writes` on.

- 292 fast memories, 44,301 acknowledged writes and
  6,551 revocations or supersessions that removed writes
  (37,211 writes removed). After every record the refolded state, the journal
  PostgreSQL kept, every stored checkpoint and the newest one handed out were compared with the
  never-saw-it fold: 29,316 bit-identity and readout checks,
  26,348 journal checks, 2,821 checkpoints verified and
  1,638 stale checkpoints planted in storage.
- 984 process crashes, 1,011 crashes mid-append
  (754 committed), 964 mid-revocation
  (546 committed, 418 rolled back),
  666 appends whose commit the server failed, 1,010 appends and
  726 checkpoints racing a revocation; every seed saw both outcomes of each.
- No bit difference, no resurrected read, no stale checkpoint handed out, no acknowledged write lost and
  no inadmissible append accepted. A read between a change's commit and the process's refold was denied
  for each of the 6,551 changes that removed writes, and no admissible read was denied.
- A refold replayed 10.61 writes per revocation on average,
  66% of what folding every remaining write from scratch
  would have replayed.
- Mutation check: 16 of 16 planted defects in the projector's cascade, the journal and
  checkpoint store and `ptr-fastmem` detected, each through the counter that shows it
  ([`results/mutations.json`](results/mutations.json)).

Per-seed counters: [`results/metrics.json`](results/metrics.json); scope, limitations and the run records
used: [`results/run.json`](results/run.json).

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
