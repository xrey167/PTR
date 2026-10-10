# L001-v2 revocation under crash: preregistration

Preregistration text of L001-v2: the `preregistration_text` file of `config.toml`'s `[preregistration]` table, whose
SHA-256 is frozen with the table. Until the freeze commit L001-v2 is `planned`: nothing here is a result and no run is
evidence. The decisions are made by `criteria.toml` and applied by
`aggregate.py`; this file gives their reasons, the definitions, the modelling choices, the limits the owner has accepted
and what the claim does not cover. Every criterion is the text of `experiment.toml` (hypothesis, falsification, metrics)
and the pins of `config.toml`'s `[pinned_thresholds]`; the harness is `bins/ptr-bench/src/experiments/l001/` (subcommand
`revocation-crash <seed>`, built with the `ledger-experiments` feature).

Entrypoint (`experiment.toml`'s `entrypoint`, the runner's command):
`cargo run --release --locked -p ptr-bench --features ledger-experiments -- revocation-crash <seed>`

L001-v2 supersedes L001 (owner decision of 2026-10-09). L001 checked one subject and one stale action, compared the number
of recovered events and not the events, ran against the v1 journal format and has no oracle, no criteria and no mutation
evidence. L001 is marked `superseded`; its `results/` are untouched and are not reused.

## 0. The two limits the owner has accepted

These are limits of the claim, stated where a reader of a COMPLETED verdict will see them. Neither is closed by this
experiment.

1. **RaftEngineLedger has no anchor.** It keeps no hash chain and no protected anchor, so it cannot tell that it lost an
   acknowledged record: its default recovery mode (`TolerateTailCorruption`, `vendor/raft-engine/src/config.rs`) drops a torn
   tail and reports success. Arm C therefore holds it to the honest-model images only (the recovered history must be the
   acknowledged prefix plus at most the record in flight). On the fsync-lie images (acknowledged bytes dropped) the engine
   opens without error on fewer records; that is counted in `undetected_rollback`, and the stale accepts that follow in
   `engine_lie_stale_accepts`. **Both are descriptive: no rule reads them and COMPLETED does not depend on them.** No product
   work is attempted on it (owner decision of 2026-10-09). A COMPLETED L001-v2 says nothing in favour of the engine's
   rollback detection; its numbers say the opposite and they are recorded.
2. **Power loss is a fsync-honest crash-image model, not a real power cut.** The model assumes the filesystem honours
   fsync: bytes a file had synced before the crash are intact, and of the bytes written after that a seeded subset
   survives (section 3). It does not model a disk that lies about a flush (except as the control of section 5), a torn
   sector below the synced length, or the behaviour of another filesystem than the one under the scratch directory. No
   real power is cut and `dm-flakey` is not used. **The filesystem type is recorded** in `environment.filesystem` of every
   result (from `/proc/self/mountinfo`), so a reader knows which kernel path the page cache and the fsync took.

## 1. Claim and scope

On the reference path, no generation whose revocation was acknowledged becomes authorizable through any enumerated crash,
power-loss or leadership schedule, and recovery is consistent: the history a recovery returns is the acknowledged prefix
plus at most the record that was in flight, it never reorders, duplicates or drops an acknowledged record, and a log and its
protected anchor that disagree are classified as the table of `crates/ptr-ledger/src/acknowledged.rs` says (a log ahead of
its anchor lost nothing and is kept; a log behind it lost committed history and is refused).

Single host, loopback only, no sockets: PTRLOG02 `FileLedger` and `AcknowledgedLedger` with its `ProtectedAnchor`;
`RaftEngineLedger` over the vendored raft-engine; three in-process `RaftNode`s on a deterministic router. The real
loopback wire is L002's arm, not this one's.

**What is not claimed.** (a) *The directory fsync.* The hypothesis names the anchor's publication as four failpoints (write,
fsync, rename, directory fsync), but the directory fsync is a failpoint position only: its effect is taken on trust, because
a directory entry is not in a file's page cache. What is checked is that `sync_parent` is entered (the `ledger.sync_parent`
failpoint, counted against the number of creations and anchor publications of the run), not that the kernel was asked to
write the entry. (b) *Raft's persist-before-send order.* The members return all their messages from one call, so the order of
persisting and sending inside `RaftNode::drain` is not exercised and is taken on trust. (c) *Recovery from a sector-torn tail.* Every power-loss image with zeroed sectors after the synced length is classified by
the shape of those bytes: (A) every byte after the synced length is zero (the file was extended, none of the new data reached
the disk: nothing written survives in it, so trimming loses nothing and the anchor shows the bytes unacknowledged), (B) a
zero gap, then the written bytes intact, (C) the written bytes intact, then zeros, (D) any other mix. **Shape A must
recover**; a refusal there is `unjustified_refusals`. A refusal of B, C or D is allowed (fail closed: a full-length frame
with a zero run is an invalid frame to the format, which never repairs input) and counted in `refusals_corrupt_tail`, within
a pinned share: `refusal_share_excess` (a falsifier) is 1 when `refused_share_damaged_permille` (refusals over recoveries
of damaged images, A included) exceeds `max_refused_share_damaged_permille`. Per-shape recoveries and refusals are reported
(`recoveries_shape_*`, `refusals_shape_*`). **Found on pilot 1001 before the freeze and fixed in `<commit>`
(the product change, `ptr-ledger`; to be filled at the commit):** with this rule and the product as it was, 255 of the 310
recoveries of damaged images in arms A and B were of shape A and all 255 were refused (`integrity::scan` rejected a frame that
starts with a zero byte with `PTR_LOG_FRAME_MAGIC`), and so were all 310: an availability defect, the usual state after a power
loss with delayed allocation being unrecoverable without an operator. The fix trims a tail only when every byte from the end of
the last complete, valid frame to the end of the file is zero, only in the anchor-directed recovery, only when the anchor (or the
trusted tail anchor) is a boundary of the complete prefix (an anchor beyond the zero region is a lost suffix, refused as before),
and reports it (`TailRecovery::trimmed_zero_tail_bytes`); without an anchor it is allowed only because a zero region can hold no
valid frame, and no acknowledgement is invented. Shapes B, C and D keep refusing. On the fixed product pilots 1001 to 1003 refuse no
shape A image and refuse 147, 134 and 151 per mille of all recoveries of damaged images (B 26, 14, 22 of 26, 14, 22; C 29, 33, 33 of
34, 34, 35; D none of 2, 2, 1 in arm C), so the bound is `max_refused_share_damaged_permille = 300`. The mutation
`refuse-every-damaged-image` (the old behavior) is killed by `unjustified_refusals` and `refusal_share_excess`. Arm C (the engine)
refused none. (d) *Counters by construction.* `crash_points_at_revocation` counts crashes
that really happened and were then observed, but which append indices are crashed is chosen by the sampler so that a
revocation is always among them: it is partly by construction, like the other plan-derived counters. (e) *Directory
fsync.* The `ledger.sync_parent` failpoint sits at the entry of the function, so it counts calls, not completed syscalls.
(f) *The shipped binary.* The
evidence is about the failpoint-instrumented build (`ptr-ledger` with `failpoints` and `raft-engine-failpoints`, whose
callbacks run inside the product's threads and in the children before they `abort()`); the production build differs from it
only in that the failpoints compile to nothing (the only edit of product code is `fail_point!` statements behind the feature
and a `fail_point!` in `sync_parent`; `write_atomically` ends in `Ok(())` after the same calls).

A **recovery** here is what a host does after a crash: for `FileLedger`, `recover_unacknowledged_tail` against the
commitment the host holds (the host knows the events it sent, so it holds the anchor of the acknowledged prefix and of
that prefix plus the one record in flight, and accepts the first that matches); for `AcknowledgedLedger`,
`AcknowledgedLedger::open` with `TailPolicy::Acknowledge` and the epoch the host retained as its rollback witness; for the
engine, `RaftEngineLedger::open`; for a Raft member, `RaftNode::open` on its directory.

## 2. Workloads and the oracle

Seeded, 6 to 14 events over 3 or 4 subjects (`capsule:l001:k`), project `l001`: `CapsuleCommitted`, `CapsuleSuperseded` and
`Revoked`, every history valid for the runtime (a commit never names a revoked generation or a lower one than the live
one; a supersession names the live generation) and every history holding a **revoke-then-recommit** (subject 0 is
committed, revoked at that generation, committed again at a higher one, at three slots drawn from the seed). The universe
of a history is every (subject, generation) from 1 to one past the highest generation the history used.

The **oracle** (`model.rs`, about sixty lines) holds nothing but the history the harness fed it. A pair may be acted on iff
the subject's live generation is that generation and no revocation names it; its standing is Revoked, Live, Superseded or
unknown. It never reads a ledger, a runtime or any other state of the code under test and shares no code with it (it does
use any type of `ptr-ledger` or `ptr-runtime`; only `Ev::to_ledger`, which the product's side of the comparison is fed with, builds the product's event). A hand-written table pins it in a unit
test. For every recovered history the harness replays the events through `PtrRuntime::replay` and compares, for every pair
of the universe, `authorize_action` (the action boundary with the capability and the effect granted, so only the
generation decides) and `generation_validity` with the oracle's reading **of the same prefix**. It also compares the
recovered events exactly (not their number) and their commit indices, which must be 1 to n.

## 3. Failure classes and the crash-image synthesizer

Kept distinct, because they leave different bytes:

- **Process abort** keeps the page cache, so every byte the process wrote survives. Two forms, both on the real files they
  leave: an in-process panic at a failpoint (the unwinding drops the ledger, as `crates/ptr-ledger/tests/failpoints.rs`
  does), and a child process that calls `std::process::abort()` at the failpoint, so nothing is dropped or unlocked by the
  process itself. The parent reads the `ACK n` lines the child printed (one per append that returned) and requires the
  child to have died of SIGABRT after exactly the expected number of them; anything else is a harness error (exit 2).
- **Power loss** keeps the synced prefix of each file and, of the bytes written after the last sync, a seeded subset. The
  honest policies are `none`, `all`, `prefix` (a seeded number of bytes), `half`, and `holes` (all of them, but seeded
  512-byte sectors read back as zeros). The images are made from what the harness saw at each failpoint (a callback records
  the files as they stand there). For `FileLedger` the synced length at a point before the sync is the previous record
  boundary, because the log syncs after every record; the boundaries are the lengths the harness measured after each
  acknowledged append. For the anchor's publication (write the scratch file, fsync it, rename it over the anchor, fsync
  the directory) the images at the rename and before the directory fsync include **both** outcomes, the old anchor with
  the scratch file and the new anchor, as a pair. For the engine, which preallocates its log, the logical length is
  measured by closing the engine after each append.

The synthesizer never changes a byte under the synced length, never makes an image longer than what was written, and
writes nothing but the written byte or a zero after the synced length; `check_in_model` says so for every image, and an
image outside the model is a harness error (exit 2). Images are de-duplicated before recovery.

## 4. Arms

| arm | code under test | crashes | judged by |
|---|---|---|---|
| A, `file` | `FileLedger` recovered against the host's anchor | 4 log failpoints at every append, honest images, in-process panics, child aborts | the hard counters |
| B, `ack` | `AcknowledgedLedger` with its `ProtectedAnchor` | the 4 log and the 4 anchor failpoints at every append (8), honest images including the rename pair, in-process panics, child aborts | the hard counters, and the classification table |
| C, `engine` | `RaftEngineLedger` | the 3 raft-engine failpoints (`file_pipe_log::append`, `log_file::write::err`, `log_fd::sync::err`) at every append, honest images, child aborts | the hard counters on honest images; the lie is descriptive (section 0) |
| D, `raft` | 3 `RaftNode`s on a deterministic router | minority leader, term change, restarts, message drop, duplication and reordering | the hard counters, `raft_revocation_visible_before_commit` |

Arm B's classification table is the harness's own: from the lengths it measured, the number of whole records in the log
image, whether bytes of another frame follow, and the records the anchor on disk covers, it computes Aligned,
Unacknowledged{n, partial} or LostSuffix, and compares it with the product's `Split`. Complete-length frames with a hole are
not classified (the product may refuse a corrupt tail; it is counted in `refusals_corrupt_tail`, descriptive and an
availability limit, not a safety finding). After every successful recovery of a set the next event of the workload is
appended and acknowledged and the set reopened: recovery must leave a ledger that can go on.

Arm D, scenario kinds (3 each per seed): **minority-leader** (an extra subject is committed; the leader is isolated and
proposes a revocation of it, which no majority ever holds; the majority elects another member and commits the rest of
the workload; the isolated leader restarts while it still believes it leads; the partition heals; the deposed leader
restarts again); **term-change** (a revocation is committed, the leader is deposed, the new leader goes on, the old one
rejoins); **restarts** (members and leaders are closed and reopened between proposals; a restarted member must still hold
everything it had committed); **message-faults** (the router drops 12%, duplicates 15% and reorders 20% of the messages
while the leader proposes, then the faults stop and the group converges). **Arm D has clean close and reopen only: there are no
power-loss images of the raft-rs storage and no failpoint inside `RaftNode::drain` or the storage's persist.** A member that
cannot be opened again from its own directory is `unjustified_refusals`; a proposal that a leader must be able to make and is
refused is `reference_mismatches`; a step of a valid schedule that returns an error ends the scenario and is
`reference_mismatches` too; every scenario must run at least one check (`scenarios_checked`). The harness decides what is committable from the
schedule and not from the members (an event proposed by an isolated leader is not), and every member's committed events,
at every checkpoint, must be a prefix of the committed order and must never contain the isolated leader's revocation.

## 5. Baselines and controls

- **B0**, the reference: the same workload without a fault, recovered through the same path and held to the same
  checks. A baseline run that trips any other falsifier, whose append fails, or whose engine does not reopen on the records
  it acknowledged is counted in `reference_mismatches` and **falsifies**. The oracle is separated from the runtime by a
  check at the start of every run: it is evaluated against a hand-written table (`model::self_check`, sixteen pairs over three
  subjects, including a supersession through generation 3 and revoked later generations) and a disagreement ends the run with
  status 2 before a case is judged. So an oracle bug is status 2, a disagreement of the runtime with a correct oracle on the
  reference path is status 1 (a defect of the runtime's authority path is never hidden behind a harness error).
- **B1**, naive recovery: the log alone, its tail trimmed, whatever is left trusted (no anchor).
- **The fsync-lie control** (outside the honest model, `lie_images`): the file as it stood after some acknowledgements,
  cut back before the last acknowledged revocation (at the record boundary and in the middle of the record); in arm B also
  the anchor file of an earlier epoch, authentic and older than the witness. B1 must accept the cut log with the
  revocation gone (a stale accept); the anchored product must refuse it (`Split::LostSuffix`, or the stale-epoch refusal).
  **If B1 does not accept a stale revocation on a lie image, the run is invalid (exit 2): it could not tell a recovery
  that notices lost records from one that does not.** The anchored product's acceptance of a lie is the falsifier
  `anchored_lie_accepted` (with the ordinary counters), and the aggregator calls such a seed FALSIFIED and never INVALID:
  only the sensitivity of the control (`naive_accepted_stale >= 1`, `lie_images >= 1`) is a condition of validity.

## 6. The durability probe

The model needs to know what a file had synced when the process stopped, and the harness does not take the product's
word for it. `probe.rs` asks the kernel (`posix_fadvise(DONTNEED)` drops the clean pages of a file and keeps the dirty
ones; `mincore` says which pages of a mapping are still cached): a page that stays is a page no flush has written. It is
read where the product claims a record durable (after `sync_data`, before the next record is written), before the anchor's
scratch file is renamed, before the anchor moves (the log must be durable first) and after every engine append. A cached
page there is `unsynced_acks`, a falsifier; so is a `RaftEngineLedger` append that returns without ever reaching the
sync of the engine's log file (the `log_fd::sync::err` failpoint is hit once per synchronized write, and a write the
engine was asked not to synchronize never hits it). Only clean readings are taken inside a run, because asking about a dirty
file starts its writeback; that the probe sees a dirty file is shown once, by a calibration on the scratch directory
(`probe_calibrated`, `probe_dirty_seen`), and a filesystem on which it cannot (tmpfs) ends the run with status 2: the
harness cannot observe what its clean readings are meant to show. The `ledger.sync_parent` failpoint, hit once per
directory synchronization, is counted against what the run must have made (the creation of the log, the first anchor and
one anchor per acknowledged append for arm B; the creation for arm A); a shortfall is `unsynced_acks`. It detects a
`sync_parent` that is not entered, not one that is entered and skips its syscall.
**It sees file data pages only**: a missing directory fsync (after the rename of the anchor, after the creation of the
log) is outside what it can detect, and the model takes the directory fsync on trust. This is the only use of `unsafe`
code (four libc calls), in a test harness.

## 7. Decision rule, exit statuses

Five seeds (17, 29, 43, 71, 101), one JSON object each, `criteria.toml` applied mechanically (a "< N" prerequisite is
written `>= N`, a "> N" violation `<= N`). COMPLETED iff every seed meets every coverage rule and no falsifier trips **and**
`results/mutations.json` counts every mutation of `tests/mutations.toml` killed (killed == total == the length of the plan);
FALSIFIED if a coverage rule (the run is vacuous) or a falsifier fails on any seed; INVALID if a record is missing, does not
describe the frozen run, a counter is missing or mistyped, a control did not fool the naive recovery, the child processes or
the probe did not run, or the exit status disagrees with the rules; a falsifier that fired in any readable record falsifies
whatever else made that record invalid. The harness exits 0 when every rule holds, 1 when one fails, 2 when it
cannot run. **Exit 2 is for harness integrity only**: the oracle failing its hand-written table, a synthesized image outside
the honest model, the fsync-lie control not fooling the naive recovery (unless the watchdog already cut the run short), a
durability probe that cannot tell a dirty file from a synced one on the scratch directory, a child process whose output is out
of sequence or that refuses its command line, a plan that is not the same twice, a failpoint that never fires on a
workload the product never failed. A defect of the code under test is never exit 2: it is a falsifier. Every product
failure path is a counter and the run goes on: an append or creation that returns an error on a valid workload (in a
crash run, a child, a restarted Raft member or a proposal a leader must accept) is `reference_mismatches` (a refused
reopening `unjustified_refusals`); a child that ends in another signal or status than the abort it was asked for, a panic, or
an append sequence that panics, is `product_panics`; a recovery, an in-process sequence of appends or a child process that
outlives its watchdog (60 s; the child is killed; the run stops issuing cases after 3, and after a hung in-process append
issues no case that needs the failpoint scenario that append still holds) is `product_hangs`. A panic of the harness itself ends the process with status 101, which the aggregator reads as INVALID.

**Coverage** (the run is vacuous if unmet; every counter is read from an event the run observed): workloads and scenarios
run; recoveries per arm; members replayed; each of the twelve failpoints hit (the eleven crash points and the directory synchronization); images that really lost bytes; rename image
pairs; fsync-lie images; child processes that died of SIGABRT; in-process panics observed; Raft leader changes, proposals
by an isolated leader, restarts of a deposed leader, messages dropped, duplicated and reordered; revoke-then-recommit
histories; the smallest history and the fewest subjects; the probe's calibration and its clean readings; and the three
observations of the claim itself: recoveries that returned a history holding an acknowledged revocation
(`recoveries_with_acked_revocation`), acknowledged-revoked pairs the replayed runtime then denied (`revoked_pairs_denied`),
and crashes that really landed on the append of a revocation (`crash_points_at_revocation`; the crash indices of the
in-process and child runs always include a revocation). Each Raft scenario must run a check (`scenarios_checked`). Forty-three
rules; thresholds keep a quarter or so of margin below the plan's own counts and below what the pilot seeds showed.
Those that hold by construction of the plan and not by measurement (they guard against a broken harness; they are named in
`criteria.toml`): the workload and scenario counts, `child_aborts_*` and `unwind_panics_observed` (each counts a crash the
plan asks for and the run saw), `events_min`, `subjects_min`, `revoke_then_recommit_histories`, `probe_calibrated` and
`probe_dirty_seen`.

**Falsifiers** (fourteen): `stale_accepts`, `acked_revocation_missing`, `recovered_not_prefix`, `reorder_or_duplicate_index`,
`unclassified_split`, `unjustified_refusals`, `raft_revocation_visible_before_commit`, `product_panics`, `product_hangs`,
`unsynced_acks`, `authority_mismatches`, `reference_mismatches`, `anchored_lie_accepted`, `refusal_share_excess`. Their definitions are in the
falsification text of `experiment.toml`. Every one is named by at least one planted mutation (`tests/mutations.toml`, 24 of
them: **21 product mutations** in `ptr-ledger` and `ptr-runtime` and **3 harness mutations**, reported separately; the harness
mutations show that the oracle and the bookkeeping are not vacuous, not that the product is sound). The coverage rules have no
mutation evidence beyond the tests that set each counter to a violating value.

Descriptive, in no rule: `undetected_rollback`, `engine_lie_stale_accepts`, `engine_lie_refused`, `refusals_*`,
`b1_recoveries`, `b1_stale_accepts`, `follow_on_appends`, `restarts`, `raft_stalled`, `raft_unconverged_members`, the
messages delivered, and every wall-clock time (`timing_ns`).

## 8. What the claim does not cover

- A real power cut, a disk that lies about a flush (only as the control), a torn sector below the synced length, another
  filesystem than the recorded one (section 0).
- The syscall of a directory fsync (section 6) and the order of persisting and sending inside a Raft member (the members
  return all their messages from one call; there is no failpoint inside `RaftNode::drain`). A `sync_parent` that is not entered
  is in the mutation plan; one that is entered and skips its syscall, and a send before the persist, are not.
- `RaftEngineLedger`'s rollback detection (section 0).
- Compaction, snapshots, retention, the real wire and a real election timer (L002).
- Availability: a corrupt unacknowledged tail the product refuses needs an operator (`refusals_corrupt_tail`).
- Any claim about a workload larger than 14 events or more than 4 subjects.

## 9. Registration and freeze

Registered (this commit): `[experiment.L001-v2.required]` in `experiments/preregistration.toml` lists `seeds` (`int-list`),
`harness` (`str`), every non-prose key of `config.toml`'s `[pinned_thresholds]` with its type (the workload, arm and
watchdog sizes, `power_loss_policies`, the failpoint lists, the fault rates, every coverage minimum,
`max_refused_share_damaged_permille`, `max_every_falsifier`, `timing_gated`, `engine_lie_gated`) and the two files
`preregistration_text` and `criteria`. `config.toml` holds the `[preregistration]` table with those keys
(`harness = "revocation-crash"`, values equal to `[pinned_thresholds]`) and `experiment.toml` the `entrypoint`.

At the freeze (the head): pin `preregistration_text_sha256` and `criteria_sha256` in the table, then `preregistration_sha256`
and `preregistration_rules_sha256` in `experiment.toml`, and move the status past `planned`.

## 10. What was run before the freeze (disclosure)

- **Pilot seeds** (not experiment seeds): 1001, 1002 and 1003 at the pinned scale, release build, on the final source (git-independent source digest, computed in the repository root as `sha256sum $(find crates/ptr-ledger/src bins/ptr-bench/src -type f | sort) | sha256sum`: `63a3ccd7961f1bc43b42d2d34b6f3e6b2ee008e83a39fb902c377d2bae58d759`; it covers only those two directories, so it is a cross-check and not the record: the freeze commit hash, recorded in the freeze entry, covers the whole tree (ptr-runtime, the harness criteria, config and mutation plan included), and the pilots are valid for the frozen tree only if the digest recomputed there equals this one; the last change to this source was rustfmt, after the mutation run of the changed and new mutations, which were killed on the same code; the registration commit then changed the source digest to `6b2ccf9952b6219c5834f40e90886e48b2e062502f79b3ebc19eefab19acbb0c` by exactly three edits and no other: the `include_str!` path in `l001/mod.rs` follows the rename of this file, the test `the_entrypoint_is_the_subcommand_and_the_status_is_planned_or_later` accepts a frozen manifest, and two doc comments in `l001/params.rs` and `l001/mod.rs` say the same; no product or harness logic changed; on the registered tree (source digest `6b2ccf99…`, after the product commit 2cf22d9a) pilot 1001 was rerun once at the pinned scale, release build: exit 0, `hard_failures` 0, `unjustified_refusals` 0, refused share of damaged images 147 per mille, identical to its earlier run): exit 0, all coverage rules and falsifiers hold, `hard_failures` 0, 6, 5 and 7 s (the times are descriptive and never judged). Pilot 1001 ran before the fix
  of section 1 (c) and showed the defect that section describes (hard_failures 255, all `unjustified_refusals`, exit 1); that
  run is the finding, and no threshold other than `max_refused_share_damaged_permille` was set from the later ones. The coverage
  thresholds were set from the pilots of earlier states of the harness and the plan's own counts, with margin, and the final pilots
  satisfy all of them.
- **Further pilot seeds 1004 to 1008** (not experiment seeds), pinned scale, release build, same source digest: exit 0, `hard_failures` 0, 0 coverage unmet, 7 to 13 s; shape A (all-zero tail) refusals 0 on every seed; `refused_share_damaged_permille` 126, 146, 144, 133 and 156 (with seeds 1001 to 1003: 147, 134, 151; maximum 156), so the bound 300 stays about twice the largest observed share. They were run to check the bound, and nothing was tuned on them.
- **Plan-scale baseline** (the scale of the mutation plan, `revocation-crash 2 1001`, unmutated): run twice on the final source,
  `hard_failures` 0 both times, exit 1 both times (33 coverage prerequisites unmet at that scale, which is no evidence), 1 and 2 s.
  So a mutation's counter is not already above 0 without the mutation. At that scale 13 of 112 recoveries of damaged images were
  refused (116 per mille), none of shape A.
- **Seed 17** was dry-run once, on 2026-10-09, in a release build of the harness as it then stood (`l001v2-dry17.json` in
  the head's scratchpad: exit 0, 0 coverage unmet, 0 falsifiers tripped, 81 s). The harness was changed afterwards, after an
  independent review (the decision rule for lie images, counters for every product failure path, three observed coverage
  counters, the oracle self-check, the narrowed corrupt-tail allowance, Raft-arm failure counters, the directory-sync
  failpoint, watchdog and margin changes). No threshold was tuned to the seed-17 numbers, and seed 17 is not run again before
  the freeze; the unit tests use pilot seeds only. Seeds 29, 43, 71 and 101 have not been run.
