# L002-raft-recovery

Recovery of the compaction cutover of an anchored ledger and of a Raft member's snapshot transitions, under crashes at every step, I/O errors, a power loss in the middle of an append, partitions, restarts and message faults. Successor in scope of [L001-v2](../L001-v2-revocation-crash/README.md), which left these transitions uncrashed.

## Hypothesis
After crashes at every step of the compaction cutover of `AcknowledgedLedger` and at every step of the snapshot, compaction and state-file transitions of a Raft member (`FileRaftStorage`, `RaftNode`), after an I/O error returned in place of a write, after a power loss in the middle of an append, and after partitions, restarts, a deposed leader and message drops, duplicates and reorders: (a) no acknowledged revocation is lost and no revoked generation becomes authorizable (checked against an independent oracle); (b) the live floor after a cutover crash is the one a table derived from the protocol gives, orphans are reclaimed without touching the live log, and the cutover can be retried; (c) a member opened again holds at least what it had, and what it presents is a prefix of the workload; (d) every vote granted and every append acknowledged in a message is already in the member's files, and a failed write returns no message; (e) after the faults the leader's heartbeats bring every member to the workload within 6 rounds, the transport reporting the snapshots it lost; (f) a snapshot whose payload omits a revocation is refused by the anchor match and stays refused across a restart.

The scope is an in-process simulated host: fsync-honest crash images (the synced prefix of each file plus a seeded subset of what was written after it), not a real power cut; three members on a deterministic router, no sockets, no election timer, no membership change; `RaftEngineLedger` is not touched (L001-v2's owner decision). The directory fsync is taken on trust. Full list of limits: [PREREGISTRATION.md](PREREGISTRATION.md), sections 0 and 10.

## Primary metrics
The integer metrics of `experiment.toml`: coverage counters read from events the run observed (failpoint hits, recoveries, images that lost bytes, snapshots installed, leader changes, messages dropped), the eighteen hard counters (the falsifiers), and descriptive ones (baseline B1, refusals, recovery rounds, the attempts the leader depositions used). Wall-clock times are in `timing_ns` and are never judged.

## Baseline
B0, the same schedule without a fault, recovered through the same path and held to the same checks (the single-node reference history is the workload itself); B1, the naive recovery on the controls, which must accept a stale revocation or the run is invalid.

## Falsification
[criteria.toml](criteria.toml): 51 coverage rules (a run that misses one is vacuous) and 18 falsifier rules; every one of the five seeds (17, 29, 43, 71, 101) must pass. COMPLETED also needs `results/mutations.json` with every mutation of [tests/mutations.toml](tests/mutations.toml) killed. Decision order and exit statuses: PREREGISTRATION.md section 8.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.

## Layout
| file | what |
|---|---|
| `PREREGISTRATION.md` | claim, plan and power, failure classes, controls, mutations, decision rule, chronology of what was run before the freeze, freeze procedure |
| `criteria.toml` | the coverage and falsifier rules the harness judges and `aggregate.py` applies |
| `config.toml` | `[pinned_thresholds]` (compiled into the harness, drift-tested) and the `[preregistration]` table frozen with the digests |
| `experiment.toml` | manifest: hypothesis, metrics, falsification text, entrypoint, preflight, seeds |
| `aggregate.py` | the five-seed verdict (COMPLETED, FALSIFIED or INVALID) |
| `tests/mutations.toml` | the planted defects the harness must kill |
| `pilots/` | pilot records of seeds 1001 to 1003 on the final code (pilots, not evidence); `pilots/README.md` lists the timing fields |
| `results/` | run records, written by the runner |

The harness is `bins/ptr-bench/src/experiments/l002/` (subcommand `raft-recovery <seed>`, built with `--features ledger-experiments`). Entrypoint and preflight are in `experiment.toml`.

## Status of this directory
L002 is `planned`. Nothing here is a result and no declared seed (17, 29, 43, 71, 101) has been run by any process; the pilots used seeds from 1001 up and are described in PREREGISTRATION.md section 9, including a first pilot on the original product that found six product defects and an independent review of the fixes that found more. The head freezes the experiment (section 11): `freeze_sign.py` pins the two digests, and the runner then runs each seed once.
