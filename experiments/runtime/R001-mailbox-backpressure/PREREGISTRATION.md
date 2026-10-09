# R001 mailbox-backpressure: preregistration

Pre-freeze text of the preregistration of R001; at the freeze it becomes the `preregistration_text` file of
`config.toml`'s `[preregistration]` table and its SHA-256 is frozen with the table, as R007's was. R001 is `planned`:
nothing here is a result and no run is evidence. The decisions are made by `criteria.toml` and applied by `aggregate.py`;
this file gives their reasons, the definitions, the modelling choices, what the harness cannot produce and the limits of
the claim. Nothing here changes, weakens or adds a success criterion of `experiment.toml`: section 6 proposes the
manifest text the freeze would write, to be confirmed. The harness is `bins/ptr-bench/src/experiments/r001/` (subcommand
`mailbox-backpressure <seed>`); the product under test is `crates/ptr-exec` (`Mailbox`, `MailboxSender`, `Receive`).

Entrypoint (`experiment.toml`'s `entrypoint`, the runner's command, set at the freeze):
`cargo run --release --locked -p ptr-bench -- mailbox-backpressure <seed>`

Scope (owner decisions of 2026-10-09, binding): `ptr_exec::Mailbox<T>` only. Latency is gated in virtual ticks of a
discrete-event simulation that uses the real `Mailbox` for every admission; wall-clock p99 and the resident set come from a
threaded arm with real producer threads and are descriptive, never gated; memory is the peak depth and the peak payload
bytes, read from a drop-tracked payload type. The baselines are an unbounded `std::sync::mpsc` channel and a `VecDeque`
reference queue that serves as the differential oracle; a bounded `std::sync::mpsc::sync_channel` (the channel the
mailbox once wrapped) is an independent second oracle. The `RequestAdmission` arm is deferred and not in scope, and so are
supervision, restart, cancellation and async mailboxes. The claim has the character of a regression guard: a bounded
mailbox keeps its promises, and the negative controls (section 4, C9) are what show the guard can fail.

## 1. Decision rule

Five seeds (17, 29, 43, 71, 101), one JSON object each, `criteria.toml` applied mechanically (a "< N" prerequisite is
written `>= N`, a "> 0" violation `<= 0`). COMPLETED iff every seed meets every coverage rule and trips no falsifier **and
the mutation evidence is complete**: `results/mutations.json` is present for this checkout's code and plan and
`killed == total ==` the number of mutations `tests/mutations.toml` lists. FALSIFIED if a coverage rule (the run is vacuous)
or a falsifier fails on any seed; a falsification is never downgraded by the mutation evidence (`mutations_ok` is reported
next to it). INVALID if a record is missing, does not describe the frozen run, a counter is missing or mistyped, the exit
status disagrees with the rules, or the mutation evidence is missing or incomplete (INVALID says nothing about the
hypothesis). The reason for the gate: the experiment's only claim to discriminate a sound mailbox from a broken one is the
mutation plan, so a pass over a harness that was never shown to fail is the vacuous pass the negative controls try to
prevent.

The harness exits 0 when every rule holds, 1 when one fails, 2 when it cannot run: a thread that cannot start, the reference
queue or the sync channel failing the oracle (alone, in underload or anywhere), or a negative control that does not trip.
**A panic or a hang of the product is a falsification, not a harness fault**: every call into `ptr_exec` is wrapped in
`catch_unwind`, a panic is counted in the hard counter `product_panics` (the harness hands back a stand-in answer and goes
on to print its judgement, exit 1), and the threaded stage runs under a 20 s watchdog with a poisonable barrier, a stage
that misses it being counted in the hard counter `product_hangs` (exit 1). A hang is only detectable in the threaded arm
(the single-thread arms have no thread to time out). A panic in the harness's own code ends the process with status 101 and
the seed is INVALID. No rule changes after a run.

Coverage (the run is vacuous if unmet). Every counter below is counted from events the run observed, not read from the
configuration:

- `overload_phases >= 18`
- `refused_full >= 1`
- `delivered >= 1`
- `unbounded_exceeding_cells >= 6`
- `capacities_exercised >= 3`
- `payload_classes >= 2`
- `recoveries_observed >= 6`
- `closed_path_cases >= 7`
- `closed_while_full_cells >= 4`
- `closed_while_nonempty_cells >= 6`
- `owner_drop_cases >= 1`
- `threaded_fill_rounds >= 2000`
- `threaded_round2_admitted >= 1`
- `threaded_round2_full >= 1`
- `threaded_round2_closed >= 1`
- `threaded_producers_with_admissions >= 2`
- `threaded_interleaved_fills >= 20`
- `threaded_delivered >= 9`
- `canary_misses <= 0`

(`negative_controls_tripped` is printed but has no rule: a control that does not trip ends the run with exit 2 before any
rule is read, so a rule on it could never be the failing one.)

Falsifiers (each tripped one falsifies R001 on that seed):

- `capacity_violations <= 0`
- `silent_drops <= 0`
- `ownership_violations <= 0`
- `fifo_violations <= 0`
- `model_divergences <= 0`
- `spurious_refusals <= 0`
- `bounded_p99_wait_ticks <= bounded_wait_bound_ticks`
- `wait_bound_violations <= 0`
- `bounded_peak_payload_bytes <= bounded_payload_bytes_bound`
- `payload_bound_violations <= 0`
- `determinism_mismatches <= 0`
- `closed_path_mismatches <= 0`
- `threaded_ledger_violations <= 0`
- `threaded_capacity_violations <= 0`
- `product_panics <= 0`
- `product_hangs <= 0`

The two rules that compare two measured numbers read, for the p99 and for the peak payload bytes, the cell of the sweep
whose value is largest relative to its own bound (compared by cross-multiplication in integers), and print that cell's
value next to that cell's bound; the rule therefore holds iff it holds in every cell. `wait_bound_violations` and
`payload_bound_violations` are the per-message and per-event counts of the same bounds, so the maximum, not only the p99,
is gated. No rule reads anything under `descriptive` or `timing_ms`.

Corollaries, stated so that they are not read as independent evidence. `wait_bound_violations` and
`payload_bound_violations` follow from "the depth never exceeds the capacity", FIFO delivery, the model agreeing, and the
harness's own work-conserving single consumer (the wait bound is `capacity * S_max` because at most `capacity - 1`
messages and one service are ahead of an admitted message; the live payload bytes are tied to the depth by the drop-tracker's
own live count): they cannot trip unless `capacity_violations`, `fifo_violations` or `model_divergences` does. They are kept
because they are the experiment's stated latency and memory claims. `bounded_p99_wait_ticks <= bounded_wait_bound_ticks` is
implied by `wait_bound_violations <= 0` (a p99 is at most the maximum); it adds no information and is kept as the claim in
its stated form. Memory means the queue bound only: the mailbox's own allocation (the `VecDeque` buffer, which grows to the
largest depth reached) is not gated and appears at most in the descriptive resident set.

## 2. Model and definitions

- **Virtual time.** Integer ticks; events fire in `(tick, seq)` order from one queue (`experiments/vtime.rs`, shared with
  S002). Every arrival is scheduled before any completion, so at an equal tick the arrival fires first. Nothing reads a
  wall clock.
- **One server.** The consumer takes a message the moment it is idle and one is queued. The service time of message `id` is
  drawn from the seed, uniformly in `S_min..=S_max` = 16..=32 ticks, before anything runs, so every arm sees the same
  schedule whatever it admits. The **wait** of a message is the tick its service starts minus the tick it arrived.
- **Phases** (arrival gaps drawn uniformly in `gap_lo..=gap_hi`; the mean service time is 24): underload, 200 arrivals, gaps
  33..40 (longer than `S_max`, so nothing queues); overload_x2, 400, gaps 8..16 (load 2); overload_x8, 800, gaps 2..4
  (load 8); burst, 300, gap 1; recovery, 600, gaps 33..40. A phase is **overload** when `gap_lo + gap_hi < S_min + S_max`.
  Pinned in `config.toml`'s `[pinned_thresholds]`.
- **Cells.** Capacity {1, 8, 128} x payload class {small = 64 body bytes, large = 4096 body bytes}: six cells. A payload is
  a `Tracked { id, body }` that reports its creation and its drop to a shared log.
- **Analytic wait bound.** An admitted message has at most `capacity - 1` messages ahead of it in the mailbox and at most one
  in service with at most `S_max` ticks left, so its wait is at most `capacity * S_max` ticks (`bounded_wait_bound_ticks`
  is the bound of the printed cell).
- **Payload bound.** Between events at most `capacity` queued payloads and one in service are alive, so at most
  `(capacity + 1) * body bytes` bytes (`bounded_payload_bytes_bound`). The count of live bytes is the drop-tracker's, not
  the queue's.
- **Arms.** B: the real `Mailbox`, offered to alternately through the mailbox and through a cloned `MailboxSender`, closed
  through the owner (small class) or through a sender (large class). U: an unbounded `std::sync::mpsc` channel, same
  schedule. R: a `VecDeque` with an explicit capacity that shares no code with the mailbox. S: a bounded
  `std::sync::mpsc::sync_channel` of the same capacity, the independent second oracle for Full, Closed and FIFO (a channel
  has no "refuse sends but keep the queue receivable" state, so its wrapper drops the sending half at the close and refuses
  later sends itself). T: the threaded arm (below).
- **Differential oracle.** Every operation on the queue under test is mirrored on a reference model (ids only) and the two
  answers are compared at every event, offers, takes and depth. A refusal where the model has room is a
  `spurious_refusals` and a `model_divergences`; any other disagreement is a `model_divergences`; an answer where exactly
  one side says `Closed` is a `closed_path_mismatches`. The run of B is also compared with the run of R and with the run of S as a
  whole (trace digest, waits, phase counts, peak depth, admitted, refused, delivered, end tick, refused-Closed, messages
  delivered after the close), and the same for the scheduled-close run of B against S. The shadow model and arm R are one
  type, so R alone adds nothing beyond the shadow; S is what is independent. The depth is sampled twice per event, once
  after the offer and before the idle consumer's take (so a transient over capacity inside an offer is seen) and once
  after the take; `Mailbox::peak_depth()` must equal the model's peak (a `model_divergences`). The oracle encodes the
  specification (Closed beats Full; a closed queue drains before it says Closed): a change of specification is a change of
  the model, by design.
- **Ownership ledger.** Every offered id must end in exactly one terminal state: delivered, refused `Full` with its value
  handed back, or refused `Closed` with its value handed back. The ledger holds the fingerprint of every payload as
  created, and counts: an id delivered twice, a delivery of an id nobody offered or whose offer was refused, a delivered or
  returned payload that is not the one created, a payload dropped while a queue or the consumer was meant to own it, a
  completion for an id not in service, live payloads other than queue depth plus the one in service (after every event),
  ids not dropped exactly once, and ids with no terminal state (`silent_drops`, which is `offered - delivered -
  refused_full - refused_closed`). `ownership_violations` is the sum of the first group; `fifo_violations` counts
  deliveries out of arrival order (ids are assigned in arrival order).
- **Panics and hangs.** Every call into the product is contained. A panic is counted in `product_panics` and answered with
  a stand-in (a refusal for an offer, `Empty` for a take); the run goes on so that the judgement is printed. a threaded stage
  that does not finish within 20 s is counted in `product_hangs`.
- **Capacity.** `capacity_violations` counts events after which the queue's own depth, or the ledger's count of admitted and
  not yet taken messages, exceeds the capacity.
- **Determinism.** The whole pass (schedules, corpora, every simulation) runs twice from the seed and the trace digests are
  compared; a differing digest is a `determinism_mismatches`.

## 3. Metrics

Every top-level integer of the harness's JSON is a metric; all are listed with what they count. The first block is
summed over the sweep (six cells), the six closed-path runs, the capacity-0 unit case and, where named, the threaded arm.

| metric | definition | rule |
|---|---|---|
| `offered`, `admitted`, `refused_full`, `refused_closed`, `delivered` | ledger totals of arm B over the six cells | coverage: `refused_full`, `delivered` |
| `silent_drops` | offered ids with no terminal state | falsifier |
| `ownership_violations` | sum of the ledger's ownership causes (printed under `internal.ownership_causes`) | falsifier |
| `capacity_violations` | see section 2 | falsifier |
| `fifo_violations` | deliveries out of arrival order | falsifier |
| `model_divergences` | disagreements with the reference model, with arm R and with arm S, and a peak depth the mailbox records differently from the model's | falsifier |
| `spurious_refusals` | `Full` where the model has room | falsifier |
| `wait_bound_violations` | admitted messages that waited more than `capacity * S_max` ticks | falsifier |
| `payload_bound_violations` | events after which more than `(capacity + 1) * body bytes` are alive | falsifier |
| `closed_path_mismatches` | closed-path disagreements (simulation, unit case, threaded arm) | falsifier |
| `determinism_mismatches` | differing trace digests between the two passes | falsifier |
| `canary_misses` | canaries of the oracle (section 4, C9) that did not trip | coverage |
| `product_panics` | panics of the product contained during the run (simulation, unit cases, threaded arm) | falsifier |
| `product_hangs` | 1 when the threaded stage missed its 20 s watchdog | falsifier |
| `threaded_ledger_violations` | threaded ledger: ownership causes, silent drops, per-producer order, and offers admitted after the producer saw `close()` return | falsifier |
| `threaded_capacity_violations` | a fill admitting other than the capacity, a queue holding another number than was admitted, a depth or peak above the capacity | falsifier |
| `threaded_admitted_after_close` | the last of those (also part of `threaded_ledger_violations`) | none |
| `bounded_peak_depth` | largest queue depth of arm B | none |
| `unbounded_peak_depth` | largest queue depth of arm U | none |
| `bounded_peak_payload_bytes`, `bounded_payload_bytes_bound` | worst cell by peak / bound | falsifier |
| `unbounded_peak_payload_bytes` | largest live bytes of arm U | none |
| `bounded_p99_wait_ticks`, `bounded_wait_bound_ticks` | worst cell by p99 (nearest rank) / bound | falsifier |
| `bounded_max_wait_ticks` | largest wait of arm B | none (the counter above gates every message) |
| `unbounded_p99_wait_ticks` | largest p99 of arm U | none |
| `refusal_permille_overload` | refused / offered in the overload phases of arm B, per mille | none |
| `overload_phases` | (cell, overload phase) pairs in which B was observed refusing | coverage |
| `unbounded_exceeding_cells` | cells in which arm U's peak depth exceeds the capacity | coverage |
| `capacities_exercised`, `payload_classes` | capacities at which B's queue was observed full; payload classes of which a body was observed alive | coverage |
| `recoveries_observed` | cells in which B's queue was empty during the recovery phase and admitted arrivals afterwards | coverage |
| `closed_path_cases` | scheduled-close runs (and the capacity-0 unit case) that refused at least one offer as `Closed` | coverage |
| `closed_while_full_cells` | cells in which an offer was refused `Closed` while the model's queue was full (Closed beats Full) | coverage |
| `closed_while_nonempty_cells` | cells in which messages were delivered after the close (drain before `Closed`) | coverage |
| `owner_drop_cases` | owner-drop units run: dropping the `Mailbox` closes the senders and releases the queued payloads | coverage |
| `threaded_producers` | producer threads | none |
| `threaded_fill_rounds` | exact fills completed | coverage |
| `threaded_producers_with_admissions` | producer threads with at least one admitted offer | coverage |
| `threaded_interleaved_fills` | fills in which at least two producers got a message in (the producers met at the queue; a race in the capacity check needs this, and a run in which the scheduler serialised the producers cannot show one) | coverage |
| `threaded_round2_admitted`, `threaded_round2_full`, `threaded_round2_closed` | outcomes observed in the contended stage | coverage |
| `threaded_offered`, `threaded_admitted`, `threaded_refused_full`, `threaded_refused_closed`, `threaded_delivered` | threaded totals | coverage: `threaded_delivered` |
| `negative_controls_tripped` | negative controls that tripped (a miss is exit 2, so no rule reads it) | none |
| `hard_failures` | sum of the fifteen hard counters (the mutation checker reads it) | none |

## 4. Cases

- **C1 underload.** The first phase, per cell: no refusal, no wait. The reference queue alone, in underload, must be clean
  at every capacity or the oracle is inconsistent with it (exit 2).
- **C2 overload x2, C3 overload x8.** Refusals per phase and cell; capacity, ownership, FIFO and model hold throughout.
- **C4 burst and recovery.** The burst, then the recovery phase: the queue drains to empty and admits again
  (`recoveries_observed`).
- **C5 capacity sweep, C6 payload classes.** The six cells through arms B, U and R: peak depth, p99 and maximum wait against
  the bound, peak payload bytes against the bound, arm U's depth and tail.
- **C7 capacity zero and the closed path.** Unit: a capacity-0 mailbox refuses every offer as `Full` and hands the value
  back, through the owner and through a sender, then, closed, as `Closed`. Owner drop: two messages are queued, the
  `Mailbox` is dropped, and a surviving sender must find the mailbox closed, empty, and its next offer returned as `Closed`
  with the two queued payloads dropped exactly once (`owner_drop_cases`). Scheduled: the mailbox is closed just before
  arrival 80 of a short x8 schedule (so a full queue is closed: `Closed` must win over `Full`, observed by
  `closed_while_full_cells`), every later offer returns its value as `Closed`, the queued messages are still delivered
  (`closed_while_nonempty_cells`), and a drained closed mailbox says `Closed`, not `Empty`.
- **C8 threaded arm.** Four producer threads, each with its own clone of a `MailboxSender`, capacity 8, in three stages
  separated by a poisonable barrier. (1) 2000 exact fills: the consumer is parked, the producers release together and
  each offers 16 pre-built messages; exactly 8 must be admitted in total, and the consumer must then find exactly 8 in the
  real queue (not the depth mirror). The producers are released by an atomic they spin on (a condvar wake-up would start them
  microseconds apart, longer than a whole fill). Repeating the fill is what lets a race in the capacity check (check,
  release the lock, push) show: about 5% of the fills have two producers meeting at the queue (`threaded_interleaved_fills`,
  required to be at least 20, so that a run in which the scheduler serialised the producers is vacuous, not a pass), and the
  racy mutation of section 5 was killed in 60 of 60 runs (40 + 20) with 15 to 119 violations per run. (2) The contended stage: the consumer drains while the producers offer 20000 each, and closes the mailbox
  after 10000 receipts or when half the offers were made. The closer sets an atomic flag with `SeqCst` right after
  `close()` returns; every producer reads it with `SeqCst` before each send, and an `Ok` after the flag was observed set is a
  violation (`threaded_admitted_after_close`, counted into `threaded_ledger_violations`); after a producer is told `Closed`,
  every later answer to it must be `Closed`. (3) After the close every offer must come back `Closed`. The ledger is checked
  over every producer's record and the consumer's after the join; each producer's messages must arrive in its own order.
  The totals of stage 2 depend on the scheduler and are not gated, but the run must have *observed* admitted, `Full` and
  `Closed` outcomes in it, and at least two producers with admissions. Wall-clock latency (p50, p99, max) and the peak
  resident set are recorded under `descriptive` and read by no rule.
- **C9 negative controls and canaries.** Negative controls (exit 2 if one does not trip, because the experiment could then
  not tell a broken queue from a sound one): the unbounded channel through the same oracle must trip `capacity_violations`
  and the wait bound, and a drop-oldest wrapper must trip `silent_drops` and a drop of an owned payload. Canaries (a miss is
  `canary_misses`, so the run is vacuous): a duplicating queue, a payload-mutating queue, a lossy consumer path, a LIFO
  queue, a mailbox of capacity + 1, a mailbox of twice the capacity, an early-refusing queue, a queue that ignores `close`,
  and one that reports a drained closed queue as empty.

## 5. Mutation plan and honesty

`tests/mutations.toml` lists 27 defects for `scripts/mutation_check.py R001`: twenty-two in
`crates/ptr-exec/src/mailbox.rs` (capacity + 1, capacity x 2, a full mailbox dropping its oldest message, `Full` reported
as `Closed`, LIFO delivery, a receive that loses the message, an early refusal, a stale depth counter, `close` that does
nothing, a drained closed mailbox reporting `Empty`, a sender clone that forks the queue, a sender whose `close` does
nothing, a send that panics when full, a receive that panics when empty, a send that hangs in a producer thread, a send that
checks the length outside the lock, a send that ignores the close when the queue is empty, Full checked before Closed, a
receive that reports Closed while messages are queued, a `close` that clears the queue, an owner drop that does not close,
a peak-depth record that is never raised) and five in the harness (a ledger blind to duplicates, a ledger blind to mutation,
a loose wait bound, a schedule mixed with the clock, a reference model not told of the close). Every hard counter is named
by at least one mutation, and a mutation counts as detected only when the harness exits 1 with a counter it names. A harness
defect that makes the harness refuse to run (exit 2) is its own guard and is not planned as a mutation. The mutation
checker passes `<cases> <seed>`; the harness accepts the pair only when both are the same seed (R001 has no case count; a
differing pair is a usage error, never another seed). The racy mutation is only visible to threads and is killed by the
repeated exact fills; the run that records the plan is `mutation_check.py R001` (all 27 must be killed for COMPLETED).

What the harness cannot produce: wall-clock tail latency as evidence (the threaded arm measures it, nothing gates it); the
behaviour of a real isolate's service-time distribution; any claim about the `RequestAdmission` queue or about an async
mailbox. A pass is the expected outcome for a mutex-guarded `VecDeque`; the value of the experiment is that a regression of
the capacity, the ownership or the order is caught with a named counter.

## 6. Manifest text proposed for the freeze (to confirm)

- `hypothesis`: a bounded typed mailbox (`ptr_exec::Mailbox`) never holds more than its capacity, hands every refused
  message back to its sender, loses, duplicates and alters none, admits exactly as a VecDeque reference queue of the same
  capacity does at every event, and keeps its admitted messages' waits within capacity x S_max virtual ticks and its live
  payload bytes within (capacity + 1) x body bytes, in capacities 1, 8 and 128 and payload classes of 64 and 4096 bytes
  through underload, overload x2 and x8, a burst and a recovery, and from at least two producer threads; an unbounded channel
  exceeds both bounds in the same schedule.
- `metrics`: the metric names of section 3 in the order the harness prints them (`bins/ptr-bench/src/experiments/r001/
  metrics.rs`, `METRIC_NAMES`).
- `baseline`: an unbounded `std::sync::mpsc` channel and a `VecDeque` reference queue, same schedule and oracle.
- `falsification`: the coverage and falsifier rules of section 1 in words, with every "<" written as in `criteria.toml`.
- `entrypoint`: the command above. `config.toml` then gets its `[preregistration]` table, `experiments/preregistration.toml`
  its `[experiment.R001]` entry, and the digests of this file and of `criteria.toml` are frozen.

Open points for the owner: the threaded stage-2 close threshold and totals (timing-dependent, not gated, only required to
show all three outcomes); the 20 s watchdog (a wall-clock deadline that decides what a stuck stage is called); whether
`bounded_max_wait_ticks` should replace the p99 in the cross-reference rule (it is already gated per message by
`wait_bound_violations`); the choice of three capacities and two payload classes; the single-server model.
