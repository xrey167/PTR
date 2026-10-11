# L002 recovery of a clustered ledger: preregistration

Preregistration text of L002: the `preregistration_text` file of `config.toml`'s `[preregistration]` table, whose SHA-256
is frozen with the table. Until the freeze commit L002 is `planned`: nothing here is a result and no run is evidence. The
decisions are made by `criteria.toml` and applied by `aggregate.py`; this file gives their reasons, the definitions, the
modelling choices, the scope that was chosen and why, the limits and what the claim does not cover. Every criterion is the
text of `experiment.toml` (hypothesis, falsification, metrics) and the pins of `config.toml`'s `[pinned_thresholds]`; the
harness is `bins/ptr-bench/src/experiments/l002/` (subcommand `raft-recovery <seed>`, built with the `ledger-experiments`
feature).

Entrypoint (`experiment.toml`'s `entrypoint`, the runner's command):
`cargo run --release --locked -p ptr-bench --features ledger-experiments -- raft-recovery <seed>`

Preflight (`experiment.toml`'s `preflight`, run by the runner in the command's environment before a seed is reserved, and by
`scripts/run_experiment.py preflight L002`; it records nothing and reads no seed):
`cargo run --release --locked -p ptr-bench --features ledger-experiments -- raft-recovery --preflight`

The preflight checks that the harness can run on this machine at all: the oracle against its hand-written table, the
failpoints compiled in, the durability probe able to tell a written file from a synced one on the scratch directory (a
filesystem on which it cannot, tmpfs, is status 2), one cutover and one unfaulted group scenario end to end with no hard
counter tripped. It exits 0 or 2.

L002 follows L001-v2 (completed): it uses L001-v2's workload generator, oracle, crash-image synthesizer, durability probe and
emission code (`l001::{model, image, probe, check}`, unchanged) and adds the arms below.

## 0. The limits that stand

1. **RaftEngineLedger has no anchor** (L001-v2, owner decision of 2026-10-09). L002 does not touch the raft-engine backend and
   does not repeat or revisit that decision: `undetected_rollback` stays L001-v2's descriptive counter. Rollback detection of
   the engine would need product work the owner declined.
2. **Power loss is a fsync-honest crash-image model, not a real power cut.** Bytes a file had synced are intact; of the bytes
   written after that a seeded subset survives (`drop-all`, `all`, `prefix`, `half`, `holes` of 512 bytes). Images exist for
   the new log and the anchor of the compaction cutover and for the log of a Raft member at the sync of an append; there are
   none for a member's state file or snapshot file (their writes are a write, a sync and a rename, which the in-process
   crashes cover at each step, and the image of a rename whose directory entry is lost is modelled only for the anchor pair
   and the new log). The filesystem is recorded in `environment.filesystem`.
3. **The order of persisting and sending inside `RaftNode::drain` is not observable from outside.** The member returns every
   message of a call from the one call, so no router can tell the order inside it. L001-v2 took it on trust. L002 checks what
   can be checked of the contract, at every call: nothing a call returns promises a vote or an append that is not in the
   member's files when it returns (the storage is opened separately and read), and a call whose write failed returns an
   error and no message (an I/O error injected at the write). A send placed before the persist inside the call, with the
   messages still returned at the end, is not detectable and is not claimed.
4. **The process-abort class is an in-process panic** at a failpoint (the unwinding drops the handles; no destructor of the
   code under test writes, which a reader can check in `ptr-ledger`). L001-v2's child processes that call `abort()` are not
   repeated: its pinned evidence is that the two forms leave the same files.
5. **No real wire, no election timer, no membership change.** Three in-process members on a deterministic router;
   `ptr-cluster`, `ptr-net` and the ALPN transports are not exercised.
6. **The directory fsync is taken on trust** (the probe sees file data pages, not directory entries); `ledger.sync_parent` is
   counted, and the cutover must enter it twice.

## 1. Claim and scope, chosen a priori

L001-v2 stated what it left out: compaction, snapshots and retention under crashes; the order of persisting and sending in a
Raft member; the real wire and a real election timer (its section 8). The scope of L002 was chosen from that list before any
code ran, by two rules: **a gap is in scope when a CPU-only harness can make it falsifiable on the reference path, and when
the product has a code path for it that a crash can interrupt**, and **a claim L001-v2 already measured is not repeated**.

| L001-v2 left out | in L002 | why |
|---|---|---|
| compaction, retention | yes, arm K | `AcknowledgedLedger::plan_compaction/compact/reclaim_orphans` is a multi-file cutover whose commit point is an anchor rename; its safety argument (a crash on either side is unambiguous) is exactly what a crash schedule tests; the runtime has `export_compacted_snapshot/restore_compacted` to tie the retained records to the discarded prefix, so revocation authority across a cutover is measurable against the oracle |
| snapshots | yes, arms S and G | `RaftNode::record_snapshot/take_installed_snapshot_matching/resume_with_applied` and `FileRaftStorage::apply_snapshot/record_snapshot` are multi-file transitions of a member's durable state (payload, state file, log rewrite) with no crash evidence |
| persist and send order | partly (limit 3) | observable only as a contract at the call boundary, with injected write errors |
| real wire, election timer | no | needs sockets and wall-clock time; L002's liveness is a count of leader heartbeat rounds |
| raft-engine rollback detection | no | owner decision, no product work |
| clean restarts, partitions, drops/duplicates/reorders of three members | repeated only where convergence to a reference is the claim | L001-v2 measured the safety of the committed prefix under them; L002 measures convergence |

**Claim.** After crashes at every step of the compaction cutover of an anchored log, and at every step of the snapshot,
compaction and state-file transitions of a Raft member's durable state, after an I/O error returned in place of a write, after
a power loss in the middle of an append, and after partitions, restarts and message faults, (a) no acknowledged revocation is
lost and no revoked generation becomes authorizable (the `PtrRuntime` restored from the compacted snapshot and the retained
records, or replayed over a member's snapshot payload and the events above it, agrees with an independent oracle for every
(subject, generation)); (b) the live floor after a cutover crash is the one a table derived from the protocol gives, orphans
are reclaimed without touching the live log, the cutover can be retried and the log goes on; (c) a Raft member opened again
holds at least what it had, and what it presents is a prefix of the workload numbered 1 to n; (d) every vote a member grants
and every append it acknowledges in a message is already in its files, and a failed write returns no message; (e) after the
partitions and the message faults have ended the leader's heartbeats bring every member to the workload, the single-node
reference history, within 6 rounds, the transport reporting the snapshots it lost; (f) a snapshot whose payload omits a revocation is refused by the anchor match and
stays refused across a restart.

**Falsification.** Any falsifier of `criteria.toml` on any of the five seeds, or any coverage rule unmet (the run is vacuous),
or a missing or partly surviving mutation record (INVALID, not COMPLETED). Baselines and controls are in section 5.

## 2. Workloads and the oracle

L001-v2's generator, at 10 to 16 events (a cutover needs history to discard and history to append) over 3 or 4 subjects
(`capsule:l001:k`, project `l001`), every history holding a revoke-then-recommit; the universe of a history is every
(subject, generation) from 1 to one past the highest the history used. The oracle (`l001/model.rs`, about sixty lines, no
code shared with the product, pinned by a hand-written table at the start of every run, exit 2 on a disagreement) says for
every pair whether an action may proceed and what the generation's standing is. For every recovered or reconstructed history
the harness compares `PtrRuntime::authorize_action` and `generation_validity` with the oracle's reading **of the same
prefix**, and the recovered events exactly (not their number) and their indices.

Two reconstructions, because a compacted log and a member with a snapshot do not hold the whole history in their records:

- **Arm K** keeps the host's compacted snapshot (`export_compacted_snapshot` at the `k` events acknowledged before the
  cutover, with its `CompactedAnchor`). The logical history is the oracle's own first `floor` events (numbered 1 to `floor`)
  followed by the records the recovered log holds; the runtime is *also* restored with `restore_compacted` from the snapshot
  and the retained records above its floor, and compared with the oracle. A refusal of that restore is an unjustified
  refusal.
- **Arms S and G** hold the application state in the snapshot payload (the text of the events it covers, written by the
  harness); a member's presented history is the payload's events followed by the committed events above them.

## 3. The plan and the power analysis

All numbers are per seed; five seeds are run. The plan's numbers come from the plan, not from a run. A coverage threshold is
either a fraction of a quantity the plan fixes (0.42 to 1.0 of it: 200 of 324 cutover recoveries, 150 of 240 panics, 90 of
90 image recoveries, 40 of 57 histories) or, where the product decides the count, a floor; the table at the end of this
section says which is which, and item 4 of section 9 says which thresholds moved after pilots had been seen.

**Arm K.** 6 workloads, 2 cutovers each: 12 cutovers. Per cutover: 1 unfaulted run (baseline B0; the hit of each of the nine
cutover failpoints is counted there, so each is hit 12 times), 9 in-process panics (one per cutover failpoint), 8 panics of an
append above the new floor (one per log or anchor failpoint), at least 5 power-loss images of the new log (policies `drop-all`,
`all`, `prefix`, `half`, `holes`, the anchor still the old one), 2 of the new log's directory entry, the anchor pair at the
rename (2 images, `rename_image_pairs`), 5 images of the anchor's scratch file, and up to 3 lie images (the anchor of the epoch
before the cutover restored over the live one; the live log cut at the record boundary and in the middle of the record before
its last acknowledged revocation). Each recovery is followed by the reclaim of the orphans, the retry of the cutover when it
did not happen, the append of the rest of the workload and a reopen. Structural lower bound: 1 + 9 + 8 + 5 + 2 + 2 = 27
recoveries per cutover, 324 per seed (the pinned threshold is 200, a bit over 0.6); 17 panics per cutover, 204 per seed
(threshold 150).

**Arms S and G.** 3 scenarios for each of the 12 crash failpoints (36), 3 for each of the 2 error-injection failpoints (6), 3
heal scenarios (partitions, a deposed leader, message faults at 12 % drop, 15 % duplicate and 20 % reorder, restarts), 3
forged-snapshot scenarios and 3 image scenarios: 51. Each fault scenario is run twice (without a fault, which counts the hits
of every failpoint and is baseline B0, and with the fault at a uniformly drawn hit of the point in the first run's count). A
schedule: member 1 is elected and proposes `a` events; member 3 is isolated; `b - a` events more are proposed; members 1 and 2
record a snapshot (which compacts their logs); one of them may restart; the partition heals; the leader's heartbeats catch
member 3 up (by a snapshot, which the harness vets against the anchor it retained out of band and accounts for, restarting
member 3 in between in about half the schedules); the rest is proposed under message faults in a heal scenario; a heal
scenario deposes the leader (isolates it and has another member of the two that can hear each other stand for election:
the candidates are tried in turn, up to `election_attempts` = 8 times under the faults and once more with the router's
faults off, so that the leader change does not depend on which votes the faults dropped; a leader that still cannot be
deposed is a failed baseline, `reference_mismatches`) and rejoins it. Because the ninth attempt runs with the router's faults
off, `leader_changes` equals the number of scheduled depositions by construction: the coverage rule `leader_changes >= 3` is a
plan check, redundant with `reference_mismatches` (a deposition that fails already raises it), and it is not evidence of
election robustness under armed faults, which L002 does not judge. How many attempts the depositions used, the latest attempt
on which a leader changed and how many depositions needed the faults-off attempt are recorded as descriptive counters
(`deposition_attempts_total`, `deposition_attempts_max`, `deposition_clean_attempt_needed`) and are in no rule. An image scenario captures the member's directory at
`raft.append.after_write_before_sync` (6 captures, 5 policies each: 30 recoveries per scenario, 90 per seed).

**Power.** What the plan can and cannot show, stated so a COMPLETED is not over-read:

- *Positions.* Each cutover failpoint is crashed 12 times per seed, 60 over the five (a defect that shows at a fraction
  `p` of the crash positions of a point and nowhere else survives with probability `(1 - p)^60`: 4.6 % at `p = 0.05`).
  Each of the 12 member failpoints is crashed at a drawn hit 3 times per seed, 15 times over five seeds (a defect at a
  fraction `f` of the hit positions survives with probability `(1 - f)^15`: 3.5 % at `f = 0.2`, 21 % at `f = 0.1`, 46 % at
  `f = 0.05`). So the member arm is a sample of positions, not an enumeration; a defect that shows at fewer than one hit in
  ten of a point is likely to be missed. The error-injection points are sampled the same way.
- *Zero-failure bound.* With no failure in `n` independent recoveries the one-sided 95 % upper bound on a failure rate is
  `1 - 0.05^(1/n)`, about `3/n`: `n` = 1,620 cutover recoveries over five seeds (324 per seed by the plan, section 3 above)
  gives 0.19 %; `n` = 450 member recoveries (the 90 image recoveries per seed by the plan; the crash and restart recoveries
  of the member arm come on top and are not counted here) gives 0.66 %.
  The recoveries are not independent draws from a distribution of defects (they are one plan), so this is a statement
  about the plan's coverage and not a probability about the product.
- *Sensitivity.* The mutation plan plants 32 defects (section 7); all must be killed at the plan's own scale, two cutover
  workloads and one scenario per failpoint, which is smaller than the pinned scale, so the pinned run sees each of them at
  least as often.
- *Recovery time.* Reported as heartbeat rounds (`rounds_to_converge_total/max`, descriptive); the bound of 6 is judged
  (`convergence_bound_exceeded`) on the phases that follow the end of the partitions and of the message faults, because
  claim (e) is about the heartbeats after the faults, where the router loses nothing. The bound follows from the protocol
  and the pinned raft configuration (heartbeat every 3 ticks; a harness heartbeat is the leader's first tick that speaks;
  the router delivers until it is quiet, so a round is one heartbeat and everything it sets off): a lagging follower needs
  one round to learn the leader's term and log position, one for the probe/append exchange, one more when a snapshot is
  first sent and then acknowledged (the host vets and accounts for it between rounds), and one more when the first snapshot
  was lost and reported: 4 rounds; 6 is that with two rounds of margin (a restart over a pending snapshot takes a vetting
  round of its own). That derivation is for a router that loses nothing and was written before the first pilot. A phase
  that runs while the faults are armed (the one after a deposed leader rejoins, under 12 % drop) is run to settle the
  group and is neither judged nor counted: a chain of four dependent exchanges at 12 % loss per message fails to finish in 6
  rounds with probability `P(Bin(6, 0.88) < 4)`, 2.6 %, a failure of the router's dice and not of the product, which the
  first version of the harness would have called a falsification. That is a harness change after the pilots (item 12 of
  section 9); it changed no pilot's outcome. The maximum seen in the final pilots is 2, so the bound is loose by a factor
  of three and `convergence_bound_exceeded` is a detector of a stalled catch-up, not a latency claim.

**Thresholds: what each is, and where it came from.** "Plan" is the quantity the plan fixes per seed; "kind" is
*structural* (the threshold is a fraction of the plan quantity), *construction* (it holds by how the plan is built) or
*floor* (the product decides the count, so there is no plan quantity: the threshold is a number below the magnitudes the
pilots showed, which makes it pilot-informed, whatever the intent). "First" is the value printed in the record of the first
pilot that carried the rule; "pilot min" is the smallest value of the three final pilots (seeds 1001 to 1003).

| rule | threshold | plan per seed | kind | pilot min | first |
|---|---|---|---|---|---|
| cuts | 10 | 12 (6 workloads x 2 cutovers) | structural, 0.83 | 12 | 10 |
| recoveries_cutover | 200 | 324 (27 per cutover) | structural, 0.62 | 384 | 200 |
| recoveries_member | 90 | 90 (3 image scenarios x 6 captures x 5 policies; the crash and restart recoveries come on top) | structural, 1.0 | 214 | 60 |
| scenarios_group, scenarios_checked | 36 | 51 (36 crash scenarios among them) | structural, 0.71 | 51, 49 | 36 |
| nine cutover failpoint hits | 10 each | 12 (one per cutover) | structural, 0.83 | 12 | 10 |
| ledger_sync_parent | 10 | 24 (two per cutover) | structural, 0.42 | 24 | 10 |
| six member write points (append, atomic x2, state rename, the two error points) | 10 each | none | floor | 1893 | 10 |
| five snapshot and compaction points | 10 each | one per snapshot recorded | floor | 98 | 10 (`raft_snapshot_after_state` rule added after pilot 1) |
| three installation points | 10 each | one per installation | floor | 48 | 10 (`raft_install_after_log_sync_before_rename` rule added after pilot 3) |
| images_lost_bytes | 1000 | none (about two sectors) | floor | 15447 | 1000 |
| rename_image_pairs | 10 | 12 (one pair per cutover) | structural, 0.83 | 12 | 10 |
| lie_images | 10 | none (up to 3 per cutover that has a revocation after the cut) | floor | 18 | 10 |
| unwind_panics_observed | 150 | 240 (17 x 12 cutover panics + 36 member crash scenarios) | structural, 0.63 | 240 | 130 |
| cuts_over_revocation | 6 | none (a property of the drawn workloads; half of 12) | floor | 9 | 6 |
| snapshots_recorded | 100 | none | floor | 162 | 40 |
| snapshots_installed | 40 | none | floor | 83 | 15 |
| restarts_between_install_and_resume | 15 | none (a fair draw per schedule over about 84 schedules: about 42 expected) | floor | 38 | 3 |
| injected_io_errors | 4 | 6 (2 points x 3) | structural, 0.67 | 6 | 4 |
| leader_changes | 3 | 3 (one deposed leader per heal scenario, deterministic since fix round 3) | structural, 1.0 | 3 | 3 |
| messages_dropped | 20 | none | floor | 1231 | 20 |
| promises_checked | 400 | none | floor | 4809 | 400 |
| promises_vote | 100 | none | floor | 235 | 10 |
| tamper_controls | 3 | 3 | construction | 3 | 3 |
| revoke_then_recommit_histories | 40 | 57 (6 cutover workloads + 51 schedules) | structural, 0.70 | 57 | 20 |
| recoveries_with_acked_revocation | 150 | none | floor | 1114 | 150 |
| revoked_pairs_denied | 200 | none | floor | 3598 | 200 |
| subjects_min, events_min | 3, 10 | 3, 10 (the generator's minimum) | construction | 3, 10 | 3, 10 |
| probe_calibrated, probe_dirty_seen | 1 | 1 (one calibration) | construction | 1 | 1 |
| probe_clean_checked | 300 | none | floor | 53499 | 300 |

The structural thresholds are 0.42 to 1.0 of their plan quantity, not "about three quarters": 200 of 324 is 0.62. Seven
thresholds of the "first" column differ from the "threshold" column; those moved, upward, after the first three pilots (item 4
of section 9). The floors are not claims about the product; they are checks that the run exercised what it was built to
exercise, and a floor that a legitimate product could miss would make COMPLETED unreachable, which is why every floor sits
at or below 0.67 of the smallest pilot value (0.39 to 0.67 for the seven that count scheduled events, far lower for the counts
of writes, messages and readings) and not at it.

`scenarios_group` and `scenarios_checked` are two counters of the same plan (51 scenarios per seed: 12 member points x 3 and 2
error points x 3 crash or I/O-error scenarios, 3 heal, 3 tamper, 3 image scenarios). `scenarios_group` is incremented when a
scenario starts, before any of its runs; `scenarios_checked` only when at least one member's final history was judged against
the oracle during the scenario (the member-replay counter grew), so a scenario in which no member is judged is in the first and
not in the second. The pilot record does not say which scenario that was, and it is not a rule problem (`raft_stalled` is 0, and no falsifier counter moved). Pilot 1001 shows 51 and 50 for this reason;
both rules have the floor 36 and neither is a verdict by itself.

## 4. Failure classes

- **Process abort** keeps the page cache. An in-process panic at the n-th hit of a failpoint, then recovery from the files it
  left. For arm K the first hit after `compact()` begins is panicked at; for the member the n-th hit counted in the unfaulted
  run (the same schedule is replayed; a failpoint that does not fire is `reference_mismatches`).
- **Power loss** keeps the synced prefix of each file and a seeded subset of what was written after it (L001-v2's
  synthesizer, `check_in_model` run on every image: an image outside the model is exit 2).
- **An I/O error** returned in place of the write at `raft.append.before_write` or `raft.state.before_write` (the failpoint's
  `return` action, after a seeded number of evaluations). The member that returned the error is treated as a host treats it:
  closed and opened again from its directory.
- **Partition, restart, drop, duplicate, reorder** on a deterministic router: one queue, a set of isolated members, seeded
  per-mille faults. The transport's part is played by the harness: a snapshot message that is dropped, sent to a member that
  is down or taken by a member that crashed is reported to its sender with `RaftNode::report_snapshot(peer, false)`.

## 5. Baselines and controls

- **B0**, the reference: the same schedule without a fault, recovered through the same path and held to the same checks. A
  baseline that trips any hard counter, a step it refuses, a proposal a leader must accept that is refused, a failpoint the
  plan asks for that is never reached, a group that does not go quiet: `reference_mismatches`, which falsifies. The single-node
  reference history of a group scenario is the workload itself, fed to the runtime in order (the oracle shares no code with the
  product); the unfaulted run must reproduce it exactly.
- **B1**, naive recovery, on the controls: the anchor file trusted without the witness (`AcknowledgedLedger::open` with
  epoch 0) on the anchor of the epoch before the cutover; the live log alone, its tail trimmed, whatever is left trusted, on the
  log cut before an acknowledged revocation; the unanchored `take_installed_snapshot` of the forged snapshot. Each must accept
  a stale revocation (the runtime replayed over what it accepted authorizes a pair the oracle revokes) in at least one case of
  the seed, **or the run is invalid (exit 2)**: it could not tell a recovery that notices lost records from one that does not.
- **The lie controls** (outside the honest model): the anchored product must refuse the anchor of an older epoch and the
  truncated live log (`refusals_lost_ack`); accepting one is `anchored_lie_accepted`, a falsifier, and the aggregator calls
  such a seed FALSIFIED when the seed is otherwise valid (it is a falsifier, not a validity condition).
- **The forged snapshot**: the leader's recorded snapshot whose payload omits the revocations, at a position where the
  omitted revocation names a generation that is live at the end of the prefix (so the omission resurrects it). The member
  that receives it must refuse it by the anchor match (`take_installed_snapshot_matching` against the digest of the genuine
  payload, retained by the harness out of band), must stay blocked on it across a restart (offered again by
  `take_installed_snapshot`), and must not hold it as its state afterwards; taking it, or holding it after the restart, is
  `tampered_snapshot_accepted`.

## 6. The durability probe

L001-v2's `probe.rs` (`posix_fadvise(DONTNEED)` and `mincore`): a page that stays cached is a page no flush has written. In
L002 it is read, clean readings only, where the cutover claims the new log durable (after its sync, before the anchor may name
it), before the anchor's scratch file is renamed, and for each of a member's `log`, `state` and `snapshot` files after every
call that returns. A cached page there is `unsynced_acks`. The cutover must also enter the directory synchronization twice (the
new log, the anchor's publication). That the probe sees a dirty file is shown once, by a calibration on the scratch directory;
a filesystem on which it cannot (tmpfs) ends the run with status 2.

## 7. Mutations

`tests/mutations.toml` lists 32 defects planted one at a time (28 in `ptr-ledger` and `ptr-runtime`, 4 in the harness: its
oracle (2), its bookkeeping (1) and its schedule (1)), each naming the hard counters that must show it; `python3
scripts/mutation_check.py L002` runs them at `raft-recovery 2 1001` and writes `results/mutations.json`; every hard counter
is named by at least one (a test checks it).
Six of the product mutations restore the behavior the first pilot found defective and three more (`snapshot-payload-shares-one-file-name`,
`install-resets-the-log-in-place`, `open-applies-what-lies-above-a-pending-snapshot`) restore what the independent review of the
fixes found (section 9), so the killed record also shows that the harness sees those defects and that the fixes are what stops it.
Two were added in fix round 3: `open-accepts-a-log-shorter-than-the-anchor` gives the second lie control (the live log cut
before an acknowledged revocation) a mutant of its own (before, only `open-ignores-the-epoch-witness`, through the first lie,
fed `anchored_lie_accepted`), and `harness-schedule-never-deposes-the-leader` stops the heal schedule from changing the
leader, which the harness sees as a failed baseline (`reference_mismatches`): the only form in which the coverage rule
`leader_changes` can be tied to a mutation.

Not plantable, and why. A mutation is killed by a *hard counter* with exit status 1, so (a) a mutant that only silences a
coverage counter ends in a coverage failure that `mutation_check.py` does not count as a kill: the 51 coverage rules are held
by the harness test that sets each counter to a violating value (`every_rule_fails_when_the_emitted_counter_is_set_to_a_violation`)
and by the aggregator tests, not by mutants; and (b) a control that stops tripping ends the run with status 2
(`a_control_that_does_not_trip_is_a_harness_failure`), which `mutation_check.py` records as a crash and never as a kill.
Not planted either, because the harness cannot see them: the removal of the `sync_all` inside `sync_parent` alone, and a send
before the persist inside `RaftNode::drain` (limits 2 and 3); a torn tail trimmed below the commit index, a tail cut that
drops valid records after a damaged one, a committed-prefix check skipped while a snapshot is pending, a log cut at a record
boundary below the commit index, the in-memory state running ahead of a state file that could not be written, a served
snapshot whose payload the state does not name, a storage that goes on writing after an I/O error past a point of no return
(a harness restarts a member after any failed call, so poisoning is invisible to it) and a failed `resume_with_applied` are
unit tests of `ptr-ledger`
(`damage_below_the_committed_index_is_refused_not_trimmed`,
`a_damaged_record_with_valid_records_after_it_is_refused_above_the_commit_too`,
`a_log_cut_below_the_commit_of_a_pending_snapshot_is_refused`,
`a_log_cut_at_a_record_boundary_below_the_commit_is_refused_and_does_not_panic`,
`a_failed_snapshot_write_leaves_the_memory_as_the_disk_is`,
`a_snapshot_payload_that_is_not_the_one_the_state_names_is_refused`,
`an_error_past_the_point_of_no_return_poisons_the_storage_and_no_append_is_acknowledged`,
`a_write_that_fails_part_way_poisons_and_loses_nothing_acknowledged`,
`a_poisoned_member_answers_nothing_and_a_restart_holds_what_it_acknowledged`,
`a_failed_resume_leaves_the_snapshot_offered_and_the_count_as_it_was`), since no schedule of the fsync-honest images damages
bytes below the synced length.

## 8. Decision rule, exit statuses

Five seeds (17, 29, 43, 71, 101), one JSON object each, `criteria.toml` applied mechanically (a "< N" prerequisite is written
`>= N`, a "> N" violation `<= N`): 51 coverage rules and 18 falsifiers. The order of the decision is the repository's:
validity, then the falsifiers, then the controls. **INVALID** if anything about the run is not sound, even when a falsifier
fired in a record that can be read: a record is missing or duplicated, does not describe the frozen run, a counter is missing
or mistyped, a control did not trip, the probe did not run, the exit status disagrees with the rules, a static-evidence
command fails or cannot start, or `results/mutations.json` is absent or does not count every mutation of `tests/mutations.toml`
killed. **FALSIFIED** if every seed can be judged and a coverage rule or a falsifier fails on any seed. **COMPLETED** iff every
seed meets every coverage rule, no falsifier trips and the mutation record counts every mutation killed. A falsifier that fired
in a record of an INVALID run stays listed in the report (`falsified_seeds`) as information; it is not the verdict. The
harness exits 0 when every rule holds, 1 when one fails, 2 when it cannot run. **Exit 2 is for harness integrity only**: the
oracle failing its table, an image outside the honest model, a control that does not trip, a probe that cannot tell a dirty
file from a synced one, a plan that is not the same twice. A defect of the code under test is never exit 2: it is a falsifier,
and the run goes on. A panic of the harness itself ends the process with status 101, which the aggregator reads as INVALID
(an exit status other than 0 or 1).

One residual risk is stated rather than hidden: `product_panics` and `reference_mismatches` are falsifiers on a record that is
otherwise valid, and a panic that originates in the harness but happens inside a contained call is counted as a product panic
(the first pilot round had one, found by reading its origin, item 8 of section 9), and a failed baseline step counts the same
way, so a harness defect of that kind would falsify the product on a record that looks valid. The origin of every panic and
of every failed baseline step is printed (`explain`) in the run's log, and the pilots were read for their causes.

**Coverage** (51 rules; the run is vacuous if unmet; each counter is read from an event the run observed): cutovers run,
recoveries of the cutover arm and of members, group scenarios run and checked, the 24 failpoints hit (the nine of the cutover,
the directory synchronization, the twelve member crash points and the two error points), images that really lost bytes,
rename image pairs, lie images, panics observed, cutovers that discarded a revocation, snapshots recorded and installed,
restarts between an installation and its accounting, injected I/O errors, leader changes, messages dropped, promises checked
(votes among them), forged-snapshot controls, revoke-then-recommit histories, recoveries that held an acknowledged revocation,
revoked pairs denied, the smallest history and the fewest subjects, the probe's calibration and clean readings. Those that
hold by construction of the plan and not by measurement are named in `criteria.toml`.

**Falsifiers** (18): `stale_accepts`, `acked_revocation_missing`, `recovered_not_prefix`, `reorder_or_duplicate_index`,
`unclassified_cutover`, `unjustified_refusals`, `orphan_hazard`, `promise_not_durable`, `member_regressed`,
`reference_divergence`, `convergence_bound_exceeded`, `tampered_snapshot_accepted`, `unsynced_acks`, `product_panics`,
`product_hangs`, `authority_mismatches`, `reference_mismatches`, `anchored_lie_accepted`; definitions in the falsification text
of `experiment.toml`. Every one is named by at least one planted mutation.

Descriptive, in no rule: `recoveries`, `recovered`, `refusals`, `b1_*`, `rounds_to_converge_*`, `deposition_attempts_total`,
`deposition_attempts_max`, `deposition_clean_attempt_needed`, `restarts`,
`messages_delivered/duplicated/reordered`, `runtimes_restored`, `follow_on_appends`, `retried_cutovers`, `raft_stalled`,
`images_damaged`, and every wall-clock time (`timing_ns`).

## 9. What was run before the freeze (disclosure), and the product findings

Nothing in this section was run on a declared seed. Every run below used a seed from 1001 up (the pilot seeds), and only
those seeds were used to tune anything. The plan of section 3 and the first thresholds of `criteria.toml` were written
before the first pilot; the thresholds were not all kept: seven were raised, and two rules were added, after pilots had been
seen (items 3 and 4). The table of section 3 gives every threshold with its origin, and no threshold was lowered; the one
thing that narrowed is the window of the convergence falsifier: the settle phase under armed faults is no longer judged (item 12,
section 3), and the judged phase after the faults are off still covers it. The thresholds
are therefore a priori in derivation for the structural ones and pilot-informed for the floors, and a reader should weigh
them so.

**Chronology.**

1. The plan, the oracle, the falsifiers and the first thresholds were written before any run; the convergence bound of 6
   heartbeat rounds was derived from the protocol and the pinned raft configuration (section 3) before the first pilot.
2. *Pilot 1001 on the original product* (harness at the mutation plan's scale, 2 cutover workloads, so scale `custom`,
   not evidence): 102 hard failures, `exit 1`. The counters: `recovered_not_prefix` 19, `unjustified_refusals` 18,
   `member_regressed` 15, `reference_divergence` 14, `unclassified_cutover` 10, `convergence_bound_exceeded` 10,
   `reference_mismatches` 9, `product_panics` 5, `tampered_snapshot_accepted` 1, `unsynced_acks` 1; `stale_accepts`,
   `acked_revocation_missing`, `promise_not_durable` and `orphan_hazard` were 0. The record printed `rounds_to_converge_max`
   26 (see item 6); the leader changed once. The record printed 49 coverage rules (item 4 lists how they differ from the
   final 51). The product defects behind these counters are the six findings below.
3. *The product fixes* (the six findings below) were made, and the harness was re-run at the same scale (pilot 1001 again:
   0 hard failures, `rounds_to_converge_max` 29, leader changes 0) and then at the pinned scale (0 hard failures, 47 of 48
   scenarios judged, `rounds_to_converge_max` 88, `leader_changes` 0, so the coverage rule `leader_changes >= 3` was the
   one rule unmet). The first pilot's record lacks the coverage rule `failpoint_hits.raft_snapshot_after_state >= 10`; the
   second has it (a rule added after pilot 1, which only makes the experiment harder to complete).
4. *Thresholds raised, and a rule added, after the first three pilots.* The records of those three pilots print the rules
   as they stood. The next pilot records print these seven different, each higher, none lower:

   | rule | in pilots 1 to 3 | afterwards | pilot 3 saw | origin of the new value |
   |---|---|---|---|---|
   | recoveries_member | 60 | 90 | 214 | structural: the 90 image recoveries of the plan |
   | unwind_panics_observed | 130 | 150 | 237 | structural: 0.63 of 240 |
   | revoke_then_recommit_histories | 20 | 40 | 54 | structural: 0.70 of 57 |
   | snapshots_recorded | 40 | 100 | 153 | floor, 0.65 of the pilot value |
   | snapshots_installed | 15 | 40 | 80 | floor, 0.50 of the pilot value |
   | restarts_between_install_and_resume | 3 | 15 | 31 | floor, 0.48 of the pilot value |
   | promises_vote | 10 | 100 | 255 | floor, 0.39 of the pilot value |

   The new values of the first three are fractions of numbers the plan fixes (section 3); whether the plan or the pilot
   magnitudes drove the choice cannot be reconstructed from what was kept, so all seven are listed. The last four are
   floors whose new numbers sit at 0.39 to 0.65 of what pilot 3 had shown, so they are pilot-informed and not a priori. They were raised to
   make the coverage rules stricter (a vacuous run is falsified), and each is far below what the final pilots give, so
   none of them is a margin a legitimate product could fail by chance; but the chronology says plainly that the numbers
   were chosen with the pilots in view. The rule set also grew, after pilot 3: `failpoint_hits.raft_install_after_log_sync_before_rename
   >= 10` was added with the twelfth member crash point (`raft.install.after_log_sync_before_rename`, item 7), a criteria
   change after the pilots; it also took the crash scenarios from 33 to 36 and the group scenarios from 48 to 51 (the
   threshold `scenarios_group >= 36` was unchanged). No other rule or
   threshold differs between the first pilot records and the final criteria, except the window of the convergence
   falsifier (the settle phase under armed faults is no longer judged, item 12).
5. *The `depose` option was added after that*: a heal scenario isolates the leader, elects member 2 and rejoins the old
   leader, because the pilots showed the schedules never changed the leader (`leader_changes` 1, 0 and 0 against a rule that
   asks for 3). The rule was in `criteria.toml` before; the schedule that satisfies it was not.
6. *The convergence counter.* The records of the first three pilots print `rounds_to_converge_max` 26, 29 and 88 (totals 43,
   57 and 171) with `convergence_bound_exceeded` 10, 0 and 0. A convergence phase is stopped at the bound of 6 and counts an
   exceedance, so a per-phase wait cannot exceed 6, and a record with no exceedance cannot hold a phase of 29 or 88 rounds:
   those numbers were not per-phase waits. The counter was redefined after them (to the maximum over phases of the rounds
   one phase took); the earlier definition was not kept, the old numbers cannot be re-derived, and they are void as
   measurements of the bound. They are reported because they were seen. That the falsifier is not blind does not rest on
   them: it fired 10 times in pilot 1 on the original product (a lost snapshot left a follower unreachable for ever,
   finding 6) and the mutation `lost-snapshot-is-never-reported` kills it. The bound of 6 was derived before pilot 1
   (section 3), was never changed, and is not tuned to the final maximum of 2: it is looser than the product needs.
7. *An independent review of the ledger changes* (a reviewer who did not write them) found three defects of the fixes and
   six smaller ones; every one was fixed before the pilots of item 10, and each that a schedule of the harness can see has a
   planted mutation (section 7):
   - the tail trim cut everything after the first unreadable record, including valid records that may have been
     acknowledged: it now cuts only an all-zero tail or a tail with no valid record after the break, reports the bytes cut
     (`Core::trimmed_tail_bytes`), and refuses a damaged record that valid records follow; the committed prefix is checked
     whether or not a snapshot is pending (unit tests; no schedule damages bytes below the synced length);
   - a snapshot payload written ahead of its state could be served under the wrong position: the payload is now stored
     under the name of its own digest, the state file names that digest (`PTRRST03`), a payload the state does not name is
     refused at open and never served, and a stop leaves the old pair or the new pair (mutation
     `snapshot-payload-shares-one-file-name`);
   - the in-place reset of the log (`set_len(0)`, then the header) could leave a log that is not a log: the reset is now a
     scratch log renamed over the old one, with a failpoint between the sync and the rename (the twelfth member crash point,
     `raft.install.after_log_sync_before_rename`; mutation `install-resets-the-log-in-place`);
   - a member holding committed entries above a snapshot that was not yet accounted for could not be opened again: open
     now keeps them and applies them when the host accounts for the snapshot, and so does the live member that refused
     them (mutation `open-applies-what-lies-above-a-pending-snapshot`);
   - the in-memory state is rolled back when the state file cannot be written, the `u64::MAX` sentinel is a typed state
     (`AppliedEvents`), `report_snapshot` takes `SnapshotDelivery` and not a boolean, the new accessors are `pub(crate)`, and
     the open-sync test counts the sync through a failpoint instead of comparing bytes.
8. *Harness defects found by the tests and pilots that followed*, fixed in the harness, none a product defect:
   the proposer re-proposed an event the leader had accepted but not yet committed when a message was dropped, so a
   history held an event twice (it now waits up to `proposal_settle_rounds` = 6 heartbeats; pinned in `config.toml`);
   a failpoint armed while the three members were first created could fire there, outside any contained call, and was
   counted as a product panic (pilot 1003, `product_panics` 1, exit 1, found by reading the panic's origin): failpoints now
   count and fire only after the members exist; and one unit test compared a reduced run of seed 1002, which draws no lie
   image at that scale, with seed 1001 (it takes the first other seed whose reduced run is valid).
9. *The first full mutation run on that code* (30 planted defects) killed 26 and left four not killed, which were fixed in
   the harness and the plan, not in the product, and the harness was then run again (item 10):
   `reorder_or_duplicate_index` was a counter the group arm never incremented (a member numbering its events from zero was
   seen only through other counters; the group arm now checks the numbering of each member's committed events, which is
   what `member-numbers-events-from-zero` needs); `member-drops-revocations-on-apply` named counters (`stale_accepts`,
   `acked_revocation_missing`) that only the cutover arm can raise, and the group arm sees it as a history that is not
   the reference (`reference_divergence`), so the plan names that; `member-swallows-a-failed-append` is seen as an
   injected error that never reaches the caller (`reference_mismatches`), which the plan now says, and a dropped write of a
   follower also raises `promise_not_durable`; and `open-applies-what-lies-above-a-pending-snapshot` was not killed because
   the schedule restarted the member the moment it was sent its snapshot, before any entry above it could be committed:
   the host now puts the snapshot off (`holding`) until the leader has proposed the rest of the workload, then restarts the
   member over what it kept. None of this changes a threshold; it changes what the schedule exercises, after pilots had
   been seen, which is why the pilots were run again.
10. *Pilots on the code of items 7 to 9* (pinned scale, seeds 1001, 1002, 1003; `exit 0` each, 0 hard failures, 0 coverage
   rules unmet, 0 falsifiers tripped; 261 to 289 s each on one machine): recoveries 821, 828 and 868 (of which cutover 384
   and member 437, 444 and 484), group scenarios 51 (50, 51 and 49 judged), leader changes 3, snapshots recorded 169, 171
   and 162, installed 88, 89 and 83, restarts between an installation and its accounting 38, 40 and 43, panics observed 240,
   lie images 24, 18 and 33, forged-snapshot controls 3 each, `rounds_to_converge_max` 2. The proposer's settle rounds
   (item 8) also let the leader carry a follower along before the explicit phase starts, which changes what the counter
   sees. The leader changes of 3 were exactly the three scheduled depositions, with no margin by construction (item 12 gives
   the fix).
11. *The mutation record.* The first full run killed 26 of 30; the four others were fixed (item 9). After the last change of
   the group arm, the 15 product mutations of the member code (`raft_node.rs`, `raft_storage.rs`) and one bookkeeping
   mutation of the cutover harness were run again, in three batches, and all 16 were killed. The other 14 mutations (the
   cutover arm of `ptr-ledger` and the harness's oracle) were not run again after that change, which touches only the group
   arm (`group.rs`), and they stood on the earlier run (killed). `results/mutations.json` is written by the whole run from a
   committed tree, which the freeze procedure of section 11 asks for. This item is superseded by item 13, which says that
   all 32 mutations were run again on the final code of fix round 3; read item 13, not this one, for the state of the
   mutation evidence.
12. *Fix round 3* (after a second independent review, of the ledger fixes and of this design; nothing was run on a declared
   seed). Product, in `crates/ptr-ledger`, each with a unit test there:
   - an I/O error past a point of no return poisons the storage: a failed directory sync after the state file or the log
     was renamed, an append that failed after the file was shortened or written to, a failed sync of an append, an error
     replacing the log once the state names a snapshot. Every later write returns `StoragePoisoned`
     (`PTR_RAFT_STORAGE_POISONED`), `RaftNode` refuses every call that could change its state, and a restart holds every
     entry acknowledged before; memory follows the files only after the I/O is durable (before, `compact_to` and the log
     replacement changed entries and offsets ahead of the rename, and an append after a failed directory sync would have been
     acknowledged into an unlinked file). New failpoints return an error in place of the halt (`raft.state.after_rename_before_dir_sync`,
     `raft.compact.*`, `raft.install.*`, `raft.append.after_write_before_sync`, `raft.open.log_sync`) and one writes part of a
     record and fails (`raft.append.partial_write`);
   - `RaftNode::resume_with_applied` makes the count durable before it changes the node (a failed write left the snapshot
     taken and the count changed);
   - a log cut exactly at a record boundary below the commit index is refused by the storage (raft panicked on it); the
     state file, the snapshot payload and the log are bounded before they are read; checked arithmetic on counts and indices
     (`resume_with_applied(u64::MAX)`); scratch files are removed at open, and only payload files this module wrote;
   - a call that brings committed entries while a snapshot waits to be accounted for no longer fails: it returns its
     messages and holds the entries for `resume_with_applied` (the error used to drop the acknowledgements of the same call).
     The harness restarts a member after every call that returns an error, which is the likely reason that the member
     recoveries of a pilot fell from 437-484 to 214-225; the threshold of 90 is the image recoveries alone and is
     unaffected. Nothing else in the counters moved.
   Harness and design: a heal scenario's deposition tries the candidates in turn, up to `election_attempts` = 8 times under
   the faults and once with them off, and a leader that still cannot be deposed is `reference_mismatches` (so
   `leader_changes >= 3` cannot be missed by a dropped vote; the threshold is unchanged, and it is the plan's three
   depositions, so it still has no margin above the plan); the settle phase under armed faults is neither judged nor
   counted (section 3); `election_attempts` is a pinned key; `keep_records` is an `int-list` in the registry (it was
   declared a list of strings, which the gate would have refused at the first `prepare`; a test now type-checks every pinned
   key); the manifest and the aggregate said "ten storage failpoints" and "loopback" and now say twelve and in-process; two
   mutations were added (section 7) and two anchors re-planted on the changed text. The pilots were run again on this
   code (pinned scale, seeds 1001, 1002, 1003): `exit 0` each, 0 hard failures, 0 coverage rules unmet, 0 falsifiers
   tripped, leader changes 3, `rounds_to_converge_max` 2, recoveries 609, 598 and 604 (cutover 384; member 225, 214 and
   220), snapshots recorded 169, 171 and 162, installed 88, 89 and 83, restarts between an installation and its accounting
   38, 40 and 43, lie images 24, 18 and 33, forged-snapshot controls 3. The whole mutation set was run on the final
   code without a record (a record needs a committed tree); its result is in the next item.
13. *The mutation record of fix round 3.* All 32 mutations were run on the final product and harness code, without a
   record (a record needs a committed tree): 30 were killed in one run and two were not. `open-accepts-a-log-shorter-than-the-anchor`
   ended as `crashed`, because the anchor store refuses an acknowledgment below its covered index as well (two layers refuse
   that cut, and removing one leaves the other); and `open-applies-what-lies-above-a-pending-snapshot` had become an
   equivalent mutant, because `apply_committed` now holds the entries back itself instead of failing (item 12), so the
   open-time branch alone no longer changes behavior. Both are defects of the plan, not of the product: each mutation now
   plants both layers (`[[mutation.also]]`), and both were then killed (`anchored_lie_accepted` 6; `recovered_not_prefix`
   10, `convergence_bound_exceeded` 15, `reference_mismatches` 7), with no code changed between the two runs (only
   `tests/mutations.toml`). `results/mutations.json` is written by the whole run from a committed tree, which the freeze
   procedure of section 11 asks for.

14. *Fix round 4* (after a third review of the ledger fixes and of this design; nothing was run on a declared seed).
   Product, in `crates/ptr-ledger`, each with a test there:
   - `RaftNode` has a node-level failed flag (`RaftNode::is_failed`): raft hands committed entries and the light ready over
     once, so any error after `ready()` (an append refused, including past the 1 GiB bound or by an invalid-input check, a
     hard-state or commit write that fails, an entry that does not apply part-way, held entries lost part-way in
     `resume_with_applied`) leaves raft ahead of what the host applied even when the storage is not poisoned; every later
     state-changing call is refused with `NodeFailed` (`PTR_RAFT_NODE_FAILED`) until a restart replays the log. Before, such
     a member kept running and numbered its next event from a count that had dropped the lost ones;
   - `resume_with_applied` is refused when no snapshot is pending (`NoSnapshotPending`) and no longer overwrites a durable
     count (the harness only calls it for a vetted snapshot it has just taken, and recording a snapshot already makes its
     count durable);
   - `FileRaftStorage::open` refuses a log that holds records when the state file is missing, instead of rewriting its
     header and mixing old records with new appends;
   - the documentation now says precisely what holds: the state-file fields are changed in memory first and rolled back
     if the state write fails (only the log's entries, offsets and handle follow the I/O), `compact_to` poisons
     conservatively on any error, and the limits are stated: a poisoned or failed member still answers reads of committed
     state (`is_leader` among them), a log at its 1 GiB bound is a permanent refusal with no compaction trigger (about
     2 to 3 GiB at open), and a failed member is restarted by its host.
   Harness and design: three descriptive counters record the leader depositions (`deposition_attempts_total`,
   `deposition_attempts_max`, the latest attempt on which a leader changed, and `deposition_clean_attempt_needed`, the
   depositions that needed the faults-off attempt); they are in no rule and no threshold changed. Section 3 now says that
   `leader_changes` equals the scheduled depositions by construction and is not evidence about elections under armed faults.
   `aggregate.py` and section 8 now decide in the repository's order, validity first (item 12 carried the sibling
   experiments' falsifier-first precedent; a falsifier in an INVALID run is now listed and is not the verdict), with a test
   of each corner; section 11 is rewritten to the procedure of the head's freeze script. These are changes of the planned
   design before the freeze, none after a prepared or frozen record. Pilots on this code (pinned scale, seeds 1001, 1002,
   1003, release; kept in `pilots/`, which says they are not evidence): `exit 0` each, 0 hard failures, 0 coverage rules
   unmet, leader changes 3, `rounds_to_converge_max` 2, recoveries 609, 598 and 604, snapshots installed 88, 89 and 83,
   restarts between an installation and its accounting 38, 40 and 43 (every number of fix round 3 reproduced), and the new
   counters: depositions used 4, 3 and 3 attempts in all (the latest attempt on which a leader changed: 2, 1 and 1), none
   needing the faults-off attempt. Seed 1001 run twice gave the same record apart from `timing_ns`.

15. *The mutation run of fix round 4.* All 32 mutations were run in one pass on the final code (after the pilots of item 14,
   with no code changed in between and the sources checked unchanged after): 32 of 32 killed, none crashed or survived.
   There is still no `results/mutations.json`, which needs a committed tree (section 11, step 6).

**The product findings** (the first pilot found six; all are fixed in `crates/ptr-ledger`, each with a test there and a
mutation here):

1. A snapshot's covered-event count was not persisted, so a leader that recorded a snapshot and restarted numbered the next
   event from zero (`snapshot-record-forgets-the-covered-events`).
2. Crash windows in installing and recording a snapshot left a state no restart could open (a state naming a snapshot over
   an unreset log, a log starting above a state that still named the previous snapshot); installation now marks the events
   as not yet accounted for, the restart finishes it and skips the records the snapshot covers
   (`install-is-not-marked-pending`).
3. A snapshot installed and not yet vetted was adopted by the next restart; it is now offered to the host again and the
   member applies nothing above it meanwhile (`pending-snapshot-forgotten-by-open`).
4. A log whose last record was torn was refused outright; a torn unacknowledged tail is now trimmed when the committed prefix
   is intact, under the rule of item 7 of the chronology (`torn-tail-refuses-to-open`).
5. Opening a log did not synchronize what it read, so a record the previous process wrote and never synced was answered
   for (`open-does-not-sync-the-log-it-read`).
6. A leader that handed a member a snapshot that was lost stopped sending it anything for ever: the transport's knowledge
   is now reported to it (`RaftNode::report_snapshot`, `report_unreachable`; `lost-snapshot-is-never-reported`).

None of the findings is an authority failure (`stale_accepts` and `acked_revocation_missing` were 0 in the first pilot);
they are lost state, lost availability and an unvetted snapshot adopted, which is what the experiment was built to find.
Negative results of the design itself: the first schedules never changed the leader (item 5), and two harness defects
(item 8) put harness faults into the product's counters (`product_panics`, and a duplicated event in a recovered history)
until the pilots were read for their causes.

## 10. What the claim does not cover

- A real power cut, a disk that lies about a flush (only as the control), a torn sector below the synced length, another
  filesystem than the recorded one.
- The syscall of a directory fsync and the order of persisting and sending inside `RaftNode::drain` (limits 3 and 6).
- `RaftEngineLedger`'s rollback detection (limit 1).
- The real wire, a real election timer, membership changes, groups of other sizes than three.
- Availability under a corrupt acknowledged log: a member whose log is damaged below its commit index is refused and needs an
  operator (`damage_below_the_committed_index_is_refused_not_trimmed`).
- A short file that is not a PTR log (8 bytes or fewer) at a member's log path with no state file is overwritten at open:
  such a file is indistinguishable from the stop of a first open, and no test or claim here covers it.
- Any claim about a workload larger than 16 events or more than 4 subjects, or snapshots larger than the harness's text
  payloads.

## 11. Registration and freeze procedure

Registered (this commit): `[experiment.L002.required]` in `experiments/preregistration.toml` lists `seeds` (`int-list`),
`harness` (`str`), every non-prose key of `config.toml`'s `[pinned_thresholds]` with its type (`keep_records` is an
`int-list`; a test type-checks every key of the table with the gate's own `kind_problem`) and the two files
`preregistration_text` and `criteria`. `config.toml` holds the `[preregistration]` table with those keys
(`harness = "raft-recovery"`, values equal to `[pinned_thresholds]`; a test pins them) and `experiment.toml` the `entrypoint` and
the `preflight`. The status is `planned`; `criteria_sha256` and `preregistration_text_sha256` are the placeholder
`must-be-pinned-before-prepared`. No value is signed by the head: L002 has no `must-be-signed-*` placeholder.

Freeze procedure, for the head (nothing here is done by the channel that built L002). Rule of 2026-10-11 (after the F007
false alarm): sign and pin only with `freeze_sign.py`, never by text replacement; `--dry-run` is the first step, before any
commit or file change; `freeze_guard.sh` `guard_on`/`guard_off` span the signing and the freeze commit; the script is checked
with `bash -n` and a dry run. The head's script (`freeze_l002.sh`) does the following, in this order; every commit names
explicit paths (never `git add -A` or a whole directory), and every step that fails stops the script.

1. `freeze_sign.py --id L002 --dir experiments/lifecycle/L002-raft-recovery --dry-run`, before any commit or file change. It
   checks the placeholders on parsed TOML values (comments never count), computes both digests and writes nothing. Then the
   pilot reference `pilots/pilot-1001.json` must exist.
2. The work commit, only if the tree is dirty: the tree this channel handed over (code, tests, `criteria.toml`, this text, the
   pilots), committed with explicit paths (`experiments`, `crates`, `bins`, `docs`, `scripts/tests`, and `STATUS.md` when it
   changed); anything changed outside those paths stops the script. The tree must be clean afterwards.
3. The pre-sign checks, all on the committed tree, in this order: `scripts/check_research_gates.py`; `scripts/check_repo.py`;
   `scripts/update_component_docs.py --check`; `scripts/mutation_check.py L002 --check`; `scripts/run_experiment.py validate`;
   `python3 -m unittest scripts.tests.test_l002_aggregate scripts.tests.test_research_gates.EnrolledExperimentTests
   scripts.tests.test_ci_local`; `cargo fmt --all -- --check` in the root workspace and again in `model/burn-a0`; `cargo clippy
   --workspace --all-targets --locked -- -D warnings`; the same with `-p ptr-bench --features ledger-experiments` and with
   `-p ptr-ledger --features raft-rs-backend,failpoints`; `cargo test -p ptr-ledger --locked` (default features, then
   `--features raft-rs-backend,failpoints`, then `--features raft-engine-backend`); `cargo doc -p ptr-ledger --no-deps
   --locked` with `RUSTDOCFLAGS="-D warnings"`; `cargo test -p ptr-bench --features ledger-experiments --locked l002`. Then
   the build directory is removed and the tree must be clean.
4. Sign under the guard: `guard_on`; `freeze_sign.py` (all or nothing, it writes `config.toml`, `experiment.toml` and
   `experiments/registry.toml`); `update_component_docs.py --write`; `run_experiment.py validate`; the gates; the tests
   `test_l002_aggregate` and `test_research_gates.EnrolledExperimentTests` again. Only `experiments/`, the crate READMEs and
   `docs/components/` may have changed.
5. The freeze commit (`experiments`, `crates`, `docs`) and the tag `l002-prereg`; `guard_off`.
6. The whole mutation set on the committed tree: `python3 scripts/mutation_check.py L002` (all 32 mutations, no `--only`):
   32 of 32 killed, or the script stops. A survivor is a harness defect, not a result. Only `results/` of the experiment may
   have changed; it is committed (mutation evidence commit). The record is stamped with its commit and accepted only while the
   code, the checker and the plan are unchanged, so nothing that is code is touched after it.
7. The pilot rerun: seed 1001 (a pilot seed, never a declared one), pinned scale, release build, on the frozen commit
   (`cargo run --release --locked -p ptr-bench --features ledger-experiments -- raft-recovery 1001`). Its output is compared
   with the committed `pilots/pilot-1001.json` key by key, excluding the fields `timing_ns` and `environment` (`environment`
   carries the filesystem of the machine; `pilots/README.md` lists both). Any other difference stops the script for the head
   to judge.
8. `run_experiment.py preflight L002` and `prepare L002`; only the experiment's directory may have changed, and it is
   committed (prepare-record commit).
9. The declared seeds are read from `config.toml` (`[preregistration].seeds`), and each is run once through the runner,
   `run L002 --seed <s>`. A pilot seed is any seed from 1001 up, and it is not run through the runner. The record of each seed
   is committed whatever the exit status, so a spent seed is always recorded; only then is the exit status read, and an exit
   other than 0 or 1 stops the run (the remaining seeds are not run on a harness that failed). Exit 1 is a recorded product
   finding.
10. `experiments/lifecycle/L002-raft-recovery/aggregate.py --report <path>` writes the results (`results/metrics.json`,
    `results/run.json`) and the script ends; it commits nothing afterwards. The finish step (the STATUS line, the status and
    `STALE.toml`) commits the results.

Seeds 17, 29, 43, 71 and 101 have not been run by any process; the pilot seeds are 1001 and above.
