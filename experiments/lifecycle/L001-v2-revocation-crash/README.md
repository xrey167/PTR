# L001-v2-revocation-crash

Successor of [L001](../L001-revocation-crash/README.md), which it supersedes (owner decision of 2026-10-09).

## Hypothesis
On the reference path no generation whose revocation was acknowledged becomes authorizable through any enumerated crash, power-loss or leadership schedule, and recovery is consistent: the history a recovery returns is the acknowledged prefix plus at most the record in flight, never a reordered, duplicated or dropped acknowledged record, and a log and its protected anchor that disagree are classified as the table of `crates/ptr-ledger/src/acknowledged.rs` says. Four arms (arm D, Raft, has clean close and reopen only, and the directory fsync of the anchor is a failpoint position whose effect is taken on trust): A, `FileLedger` recovered against the host's anchor; B, `AcknowledgedLedger` with its `ProtectedAnchor`, crashed at the four failpoints of the log append and the four of the anchor publication; C, `RaftEngineLedger` at the three failpoints of raft-engine; D, three in-process `RaftNode`s on a deterministic router (isolation with a minority proposal, term change, restarts, message drop, duplication and reordering). Failure classes are kept distinct: process abort (an in-process panic, and a child process that calls `abort()`) keeps the page cache; power loss is a fsync-honest crash-image model (the synced prefix of each file plus a seeded subset of what was written after it).

## Primary metrics
The 75 integer metrics of `experiment.toml`: the coverage counters (observed events: failpoint hits, images that lost bytes, children that died of SIGABRT, messages dropped, leaders elected), the fourteen hard counters, and descriptive ones; wall-clock times are in `timing_ns` and are never judged.

## Baseline
B0, the same workload without a fault; B1, naive recovery (the log alone, tail trimmed, whatever is left trusted), run on the fsync-lie images, where it must accept a stale revocation.

## Falsification
See `experiment.toml` and [criteria.toml](criteria.toml): fourteen falsifier rules and forty-three coverage rules, every seed must pass; COMPLETED also needs `results/mutations.json` with every mutation of the plan killed.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.

## Status of this directory
L001-v2 is `planned`: the harness (`bins/ptr-bench/src/experiments/l001/`, subcommand `revocation-crash <seed>`, built with `--features ledger-experiments`), the preregistration ([PREREGISTRATION.md](PREREGISTRATION.md)), `criteria.toml`, `aggregate.py` and the mutation plan ([tests/mutations.toml](tests/mutations.toml)) are in place, and the experiment is registered in `experiments/preregistration.toml` with a `[preregistration]` table in `config.toml` whose two digests (`preregistration_text_sha256`, `criteria_sha256`) are still `must-be-pinned-before-prepared`; the head pins them at the freeze, and no seed has been run for the record. `config.toml`'s `[pinned_thresholds]` holds the choices the harness compiles in (a drift test compares them).

Stated limits (section 0 of the preregistration): `RaftEngineLedger` has no hash chain and no protected anchor, so after a lie that drops acknowledged bytes it opens on fewer records without error; `undetected_rollback` counts that and is descriptive, and no product work is attempted on it. Power loss is a fsync-honest crash-image model and not a real power cut; the filesystem is recorded in every result. The durability probe sees file data pages, not directory entries. A corrupt unacknowledged tail that the product refuses is counted and is an availability limit.
