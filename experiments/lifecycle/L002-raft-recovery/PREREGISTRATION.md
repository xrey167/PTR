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
faults the leader's heartbeats bring every member to the workload, the single-node reference history, within 6 rounds, the
transport reporting the snapshots it lost; (f) a snapshot whose payload omits a revocation is refused by the anchor match and
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

All numbers are per seed; five seeds are run. They come from the plan, not from a run; the thresholds are about three quarters
of the plan's structural lower bound, never of an observed value.

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
scenario deposes the leader (isolates it, elects member 2, rejoins it). An image scenario captures the member's directory at
`raft.append.after_write_before_sync` (6 captures, 5 policies each: 30 recoveries per scenario, 90 per seed).

**Power.** What the plan can and cannot show, stated so a COMPLETED is not over-read:

- *Positions.* Each cutover failpoint is crashed 12 times per seed, 60 over the five (a defect that shows at a fraction
  `p` of the crash positions of a point and nowhere else survives with probability `(1 - p)^60`: 4.6 % at `p = 0.05`).
  Each of the 12 member failpoints is crashed at a drawn hit 3 times per seed, 15 times over five seeds (a defect at a
  fraction `f` of the hit positions survives with probability `(1 - f)^15`: 3.5 % at `f = 0.2`, 21 % at `f = 0.1`, 46 % at
  `f = 0.05`). So the member arm is a sample of positions, not an enumeration; a defect that shows at fewer than one hit in
  ten of a point is likely to be missed. The error-injection points are sampled the same way.
- *Zero-failure bound.* With no failure in `n` independent recoveries the one-sided 95 % upper bound on a failure rate is
  `1 - 0.05^(1/n)`, about `3/n`: `n` = 1,620 cutover recoveries over five seeds gives 0.19 %, `n` = 615 member recoveries 0.49 %.
  The recoveries are not independent draws from a distribution of defects (they are one plan), so this is a statement
  about the plan's coverage and not a probability about the product.
- *Sensitivity.* The mutation plan plants 30 defects (section 7); all must be killed at the plan's own scale, two cutover
  workloads and one scenario per failpoint, which is smaller than the pinned scale, so the pinned run sees each of them at
  least as often.
- *Recovery time.* Reported as heartbeat rounds (`rounds_to_converge_total/max`, descriptive); the bound of 6 is judged
  (`convergence_bound_exceeded`). The bound follows from the protocol and the pinned raft configuration (heartbeat every 3
  ticks, a harness heartbeat is the leader's first tick that speaks): a round is one heartbeat and its responses; a lagging
  follower needs one round to learn the leader's term, one for the probe/append exchange, one more when a snapshot is first
  sent and then acknowledged, and one more when the first snapshot was lost and reported; 6 is that sum with margin. It was
  set before the first pilot; the pilots' maximum is in section 9.

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
  such a seed FALSIFIED and never INVALID.
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

`tests/mutations.toml` lists 30 defects planted one at a time (27 in `ptr-ledger` and `ptr-runtime`, 3 in the harness's oracle
and bookkeeping), each naming the hard counters that must show it; `python3 scripts/mutation_check.py L002` runs them at
`raft-recovery 2 1001` and writes `results/mutations.json`; every hard counter is named by at least one (a test checks it).
Six of the product mutations restore the behavior the first pilot found defective and three more (`snapshot-payload-shares-one-file-name`,
`install-resets-the-log-in-place`, `open-applies-what-lies-above-a-pending-snapshot`) restore what the independent review of the
fixes found (section 9), so the killed record also shows that the harness sees those defects and that the fixes are what stops it. Not planted, because the harness cannot see
them: the removal of the `sync_all` inside `sync_parent` alone, and a send before the persist inside `RaftNode::drain` (limits
2 and 3); a torn tail trimmed below the commit index, a tail cut that drops valid records after a damaged one, a committed-prefix
check skipped while a snapshot is pending, the in-memory state running ahead of a state file that could not be written and a
served snapshot whose payload the state does not name are unit tests of `ptr-ledger`
(`damage_below_the_committed_index_is_refused_not_trimmed`,
`a_damaged_record_with_valid_records_after_it_is_refused_above_the_commit_too`,
`a_log_cut_below_the_commit_of_a_pending_snapshot_is_refused`, `a_failed_snapshot_write_leaves_the_memory_as_the_disk_is`,
`a_snapshot_payload_that_is_not_the_one_the_state_names_is_refused`), since no schedule of the fsync-honest images damages bytes
below the synced length. The coverage rules have
no mutation evidence beyond the tests that set each counter to a violating value.

## 8. Decision rule, exit statuses

Five seeds (17, 29, 43, 71, 101), one JSON object each, `criteria.toml` applied mechanically (a "< N" prerequisite is written
`>= N`, a "> N" violation `<= N`): 51 coverage rules and 18 falsifiers. COMPLETED iff every seed meets every coverage rule and
no falsifier trips **and** `results/mutations.json` counts every mutation of `tests/mutations.toml` killed; FALSIFIED if a
coverage rule or a falsifier fails on any seed; INVALID if a record is missing, does not describe the frozen run, a counter is
missing or mistyped, a control did not trip, the probe did not run, or the exit status disagrees with the rules; a falsifier
that fired in any readable record falsifies whatever else made that record invalid. The harness exits 0 when every rule holds,
1 when one fails, 2 when it cannot run. **Exit 2 is for harness integrity only**: the oracle failing its table, an image outside
the honest model, a control that does not trip, a probe that cannot tell a dirty file from a synced one, a plan that is not the
same twice. A defect of the code under test
is never exit 2: it is a falsifier, and the run goes on. A panic of the harness itself ends the process with status 101, which
the aggregator reads as INVALID.

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

Descriptive, in no rule: `recoveries`, `recovered`, `refusals`, `b1_*`, `rounds_to_converge_*`, `restarts`,
`messages_delivered/duplicated/reordered`, `runtimes_restored`, `follow_on_appends`, `retried_cutovers`, `raft_stalled`,
`images_damaged`, and every wall-clock time (`timing_ns`).

## 9. What was run before the freeze (disclosure), and the product findings

Nothing in this section was run on a declared seed. Every run below used a seed from 1001 up (the pilot seeds), and only
those seeds were used to tune anything. The thresholds of `criteria.toml` and the plan of section 3 were written from the
plan's structural lower bound before the first pilot; the pilots' magnitudes were nevertheless seen before the numbers were
finalized (the chronology below says where), so the thresholds are a priori in derivation and not in innocence, and a reader
should weigh them so.

**Chronology.**

1. The plan, the oracle, the falsifiers and the first thresholds were written before any run; the convergence bound of 6
   heartbeat rounds was derived from the protocol and the pinned raft configuration (section 3) before the first pilot.
2. *Pilot 1001 on the original product* (harness at the mutation plan's scale, 2 cutover workloads, so scale `custom`,
   not evidence): 102 hard failures, `exit 1`. The counters: `recovered_not_prefix` 19, `unjustified_refusals` 18,
   `member_regressed` 15, `reference_divergence` 14, `unclassified_cutover` 10, `convergence_bound_exceeded` 10,
   `reference_mismatches` 9, `product_panics` 5, `tampered_snapshot_accepted` 1, `unsynced_acks` 1; `stale_accepts`,
   `acked_revocation_missing`, `promise_not_durable` and `orphan_hazard` were 0. The longest wait to converge was 26 rounds
   against the bound of 6 (see the note on that counter at the end of the chronology); the leader changed once. The
   product defects behind these counters are the six findings below.
3. *The product fixes* (the six findings below) were made, the harness was re-run at the same scale (pilot 1001 again:
   0 hard failures, longest wait 29 rounds, leader changes 0) and then at the pinned scale (0 hard failures, 47 of 48
   scenarios judged, longest wait 88 rounds, `leader_changes` 0, so the coverage rule `leader_changes >= 3` was the one
   rule unmet).
4. *The `depose` option was added after that*: a heal scenario now isolates the leader, elects member 2 and rejoins the
   old leader, because the pilot showed the schedules never changed the leader (`leader_changes` 0 against a rule that
   asks for 3). The rule was in `criteria.toml` before; the schedule that satisfies it was not. The bound of 6 rounds was not
   changed after seeing any pilot and is not tuned to the final pilots' maximum of 2: it is looser than the product needs,
   so `convergence_bound_exceeded` is sensitive to a stalled catch-up (the mutation `lost-snapshot-is-never-reported`
   shows it) and not to a modest slowdown.
5. *An independent review of the ledger changes* (a reviewer who did not write them) found three defects of the fixes and
   six smaller ones; every one was fixed before the pilots below, and each that a schedule of the harness can see has a
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
6. *Harness defects found by the tests and pilots that followed*, fixed in the harness, none a product defect:
   the proposer re-proposed an event the leader had accepted but not yet committed when a message was dropped, so a
   history held an event twice (it now waits up to `proposal_settle_rounds` = 6 heartbeats; pinned in `config.toml`);
   a failpoint armed while the three members were first created could fire there, outside any contained call, and was
   counted as a product panic (pilot 1003, `product_panics` 1, exit 1, found by reading the panic's origin): failpoints now
   count and fire only after the members exist; and one unit test compared a reduced run of seed 1002, which draws no lie
   image at that scale, with seed 1001 (it takes the first other seed whose reduced run is valid).
7. *The first full mutation run on that code* (30 planted defects) killed 26 and left four not killed, which were fixed in
   the harness and the plan, not in the product, and the harness was then run again (item 8):
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
8. *Final pilots on the final code* (pinned scale, seeds 1001, 1002, 1003; `exit 0` each, 0 hard failures, 0 coverage rules
   unmet, 0 falsifiers tripped; 261 to 289 s each on one machine): recoveries 821, 828 and 868 (of which cutover 384 and
   member 437, 444 and 484), group scenarios 51 (50, 51 and 49 judged), leader changes 3, snapshots recorded 169, 171 and 162,
   installed 88, 89 and 83, restarts between an installation and its accounting 38, 40 and 43, panics observed 240, lie
   images 24, 18 and 33 (the naive recovery accepted a stale revocation in 20, 15 and 32 of the baseline B1 recoveries),
   forged-snapshot controls 3 each, `rounds_to_converge_max` 2 and `rounds_to_converge_total` 189, 191 and 186. The longest
   wait of 2 is the maximum over runs of the rounds one convergence phase took (a phase stops at the bound of 6 and counts an
   exceedance, so no run can report more than 6); the 26, 29 and 88 of the earlier pilots are larger than that and so
   were not per-run maxima: an earlier version of the counter aggregated differently, and the final version, which takes a
   maximum, replaced it. This was not re-derived, and the earlier numbers are reported only because they were seen. The
   proposer's settle rounds (item 6) also let the leader carry a follower along before the explicit phase starts, which
   changes what the counter sees. It is descriptive and the bound of 6 is unchanged.
9. *The mutation record.* The first full run killed 26 of 30; the four others were fixed (item 7). After the last change of the
   group arm, the 15 product mutations of the member code (`raft_node.rs`, `raft_storage.rs`) and one bookkeeping mutation
   of the cutover harness were run again, in three batches, and all 16 were killed. The other 14 mutations (the cutover arm
   of `ptr-ledger` and the harness's oracle) were not run again after that change, which touches only the group arm
   (`group.rs`), and they stand on the earlier run (killed). `results/mutations.json` is written by the whole run, so the freeze
   procedure of section 11 asks for one whole run from the frozen commit.

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
   is intact, under the rule of item 5 of the chronology (`torn-tail-refuses-to-open`).
5. Opening a log did not synchronize what it read, so a record the previous process wrote and never synced was answered
   for (`open-does-not-sync-the-log-it-read`).
6. A leader that handed a member a snapshot that was lost stopped sending it anything for ever: the transport's knowledge
   is now reported to it (`RaftNode::report_snapshot`, `report_unreachable`; `lost-snapshot-is-never-reported`).

None of the findings is an authority failure (`stale_accepts` and `acked_revocation_missing` were 0 in the first pilot);
they are lost state, lost availability and an unvetted snapshot adopted, which is what the experiment was built to find.
Negative results of the design itself: the first schedules never changed the leader (item 4), and two harness defects
(item 6) put harness faults into the product's counters (`product_panics`, and a duplicated event in a recovered history)
until the pilots were read for their causes.

## 10. What the claim does not cover

- A real power cut, a disk that lies about a flush (only as the control), a torn sector below the synced length, another
  filesystem than the recorded one.
- The syscall of a directory fsync and the order of persisting and sending inside `RaftNode::drain` (limits 3 and 6).
- `RaftEngineLedger`'s rollback detection (limit 1).
- The real wire, a real election timer, membership changes, groups of other sizes than three.
- Availability under a corrupt acknowledged log: a member whose log is damaged below its commit index is refused and needs an
  operator (`damage_below_the_committed_index_is_refused_not_trimmed`).
- Any claim about a workload larger than 16 events or more than 4 subjects, or snapshots larger than the harness's text
  payloads.

## 11. Registration and freeze procedure

Registered (this commit): `[experiment.L002.required]` in `experiments/preregistration.toml` lists `seeds` (`int-list`),
`harness` (`str`), every non-prose key of `config.toml`'s `[pinned_thresholds]` with its type and the two files
`preregistration_text` and `criteria`. `config.toml` holds the `[preregistration]` table with those keys
(`harness = "raft-recovery"`, values equal to `[pinned_thresholds]`; a test pins them) and `experiment.toml` the `entrypoint` and
the `preflight`. The status is `planned`; `criteria_sha256` and `preregistration_text_sha256` are the placeholder
`must-be-pinned-before-prepared`.

Freeze procedure, for the head (nothing here is done by the channel that built L002):

1. Check the tree is the one that passed `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets --locked -- -D
   warnings` (and with `--features ledger-experiments` for `ptr-bench`, and `--features raft-rs-backend,failpoints` for
   `ptr-ledger`), `cargo test -p ptr-ledger --features raft-rs-backend,failpoints --locked`, `cargo test -p ptr-bench
   --features ledger-experiments --locked l002`, `python3 scripts/run_experiment.py validate`, `python3
   scripts/check_research_gates.py`, `python3 scripts/tests/test_l002_aggregate.py`, `python3 scripts/mutation_check.py L002
   --check`.
2. Run `python3 scripts/mutation_check.py L002` on a quiet committed tree (nothing else building; the directory on a
   disk-backed filesystem) and commit `results/mutations.json`: 30 of 30 killed. A survivor is a harness defect, not a result.
3. Run `python3 scripts/run_experiment.py preflight L002`; it must exit 0.
4. Pin `preregistration_text_sha256` and `criteria_sha256` in `config.toml`'s table (the SHA-256 of the files as committed),
   then `preregistration_sha256` and `preregistration_rules_sha256` in `experiment.toml`, and move the status past `planned`
   (`python3 scripts/run_experiment.py prepare L002`).
5. Run each seed once through the runner (`python3 scripts/run_experiment.py run L002 --seed <s>`, seeds 17, 29, 43, 71, 101);
   a pilot seed is any seed from 1001 up, and it is not run through the runner.
6. `python3 experiments/lifecycle/L002-raft-recovery/aggregate.py`; commit the aggregate with the seed records.

Freeze scripts (rule of 2026-10-11): sign and pin only with the head's `freeze_sign.py --id <ID> --dir <experiment dir>` (never by text
replacement), run it with `--dry-run` as the first step before any commit or file change, wrap the span from signing to the freeze
commit in `freeze_guard.sh` `guard_on`/`guard_off`, and check the script with `bash -n` and a dry run. No such script was written by
the channel that built this experiment.

Seeds 17, 29, 43, 71 and 101 have not been run by any process; the pilot seeds are 1001 and above.
