# P0.8 — Durable execution audit, fencing and at-most-once effects

Baseline: `aad4352329531d70c9fc642f433561c76cdb8266`.
Owner: `ptr-runtime::execution`, record format in `ptr-ledger`.

Issue #20's *Execution/network trust boundaries* gate asks for six things. This
contract covers four of them — journaled execution audit, durable fencing,
outcome reconciliation and idempotency. Authenticated session admission and
project-scoped Pod access are the other two and are **not** covered here; they
are a separate package with a separate contract.

## The defect this replaces

P0.1 fenced the runtime when an executor's outcome was ambiguous. It did so with
an in-memory `bool`, and that bool was reconstructed as `false` on every open.

So the window at the centre of the gate — a crash after the effect was dispatched
and before its outcome was known — resolved in the unsafe direction by default.
The process died holding the knowledge that something might have happened
outside, and the next process started up believing nothing had. Nothing in the
journal disagreed with it, because nothing in the journal mentioned the effect at
all: `LedgerEvent` had nine variants and none of them described an action being
executed.

A fence is a claim about the outside world. It cannot live somewhere that a
restart clears.

## The three properties that carry the design

### The fence is a record, not a flag

An `EffectAttempted` record is committed **before** the executor is called. An
`EffectSettled` or `EffectReconciled` record closes it. An attempt with no
closing record *is* the fence:

```
is_fenced() = ledger-append ambiguity  OR  some attempt has no settlement
```

There is no separate state to persist, because the fence is derived from
committed history every time the runtime is built. A restart cannot lose it and a
bug cannot forget to write it — the only way to not be fenced is for a
settlement to exist.

This is the same log-first ordering `AcknowledgedLedger` uses for its anchor, and
for the same reason: of the two directions a crash can leave, only one is
recoverable. A record written after the effect would describe the case that
cannot happen (a crash *after* we already knew the outcome) and miss the one that
can.

An executor error and an executor panic are treated identically, and neither
settles. An `Err` cannot distinguish "did not apply" from "applied and could not
say so"; a panic says even less. Both leave the attempt open, so both outlive the
process.

### Failure to know is not failure to record

`EffectSettled` always carries the response digest. It carries the response bytes
only up to `MAX_RETAINED_RESPONSE` (1 MiB), and above that the `response` field
is absent.

The digest is unconditional on purpose: "we did not keep the response" must never
become "we do not know what happened". A retry under an at-most-once key whose
response was not retained is **refused** — `ResponseNotRetained` — because the
two alternatives are worse. Re-executing would apply the effect twice, which is
the one thing the key exists to prevent, and synthesising a response would answer
a caller with something the effect never produced.

Truncating a large response would be the same fault in quieter form: a second,
wrong answer rather than a missing one. So a retained response that is longer than
the bound, or that disagrees with the digest beside it, is refused during
validation — before append and again during replay.

### A settlement that fails is not a refusal

The attempt is committed before the executor is called and the settlement after it
returns, so the second append can fail after the effect applied: the ledger has no
index left, or a durable write fails. `execute_prepared` and `settle_detached`
return `SettlementNotRecorded { attempt, error }` for that, never `Audit`. `Audit`
is what a failed attempt record returns, and what a failed reconciliation returns,
which leaves the attempt awaiting one. For the attempt record nothing was
attempted and no unsettled attempt is left to fence on, so the execution wire
reports it as `Refused`; if the ledger's own append failed, the runtime is still
fenced by the ambiguity of that append until it is reopened, and only a record
refused by validation before the append leaves it unfenced. Earlier builds
returned `Audit` for a failed settlement too, so a requester was told that nothing
had been attempted for an effect that had applied, and a retry under a fresh key
would have applied it a second time.

Nothing about the failure clears the fence. The failed append leaves this process's
own last append ambiguous and the attempt unsettled, so no permit is prepared, no
journal anchor is produced and detached work stays outstanding. Neither a second
settlement nor a reconciliation can be committed in this process, because
ambiguity about the runtime's own last append blocks every settlement (below): a
second `settle_detached` of the same answer returns `SettlementNotRecorded` again,
with `ExecutionFenced` as its error, and `reconcile_effect` returns
`Audit(ExecutionFenced)`. The response is not handed on, since the runtime could
not record it, and the wire reports the effect as `AppliedWithoutResponse`
(`32-execution-wire.md`).

Reopening clears that ambiguity and replays what the ledger holds. After a failed
durable write that is the way forward: a torn tail is first repaired against the
retained anchor with `FileLedger::recover_unacknowledged_tail`, because a plain
open refuses an incomplete frame (`23-persistence-integrity.md`), and an attempt
still unsettled after the reopen is what reconciliation is for. When the failure
was the ledger running out of room, it is not: the reopened runtime replays the
attempt and no settlement, so it is fenced until the attempt is reconciled, and a
reconciliation is an append of its own, which the full ledger refuses the same way
(`PTR_LEDGER_INDEX_EXHAUSTED`, or `PTR_LOG_RECORD_LIMIT` once a durable log holds
`MAX_RECORDS`). A fenced runtime also produces no journal anchor and no compacted
snapshot, so compaction cannot make room. The fence is then permanent in band,
which is the safe direction and is listed under "What this does not close".

Asserted with a ledger one index below its ceiling, so the attempt takes the last
index and the settlement finds none: for a synchronous effect
(`an_effect_whose_settlement_cannot_be_recorded_is_reported_as_applied_and_fences_the_runtime`
in `crates/ptr-runtime/tests/execution_audit.rs`), for an adapter's answer
(`an_adapter_answer_that_cannot_be_recorded_is_reported_as_applied_and_the_window_stays_open`
in `crates/ptr-runtime/tests/detached_effects.rs`) and over the wire
(`an_effect_whose_settlement_the_host_cannot_record_is_reported_applied_not_refused`
in `crates/ptr-execwire/tests/wire.rs`). Each fails with the settlement mapped
back onto `Audit`, as it was before the fix. The same runtime restored with the
attempt it wrote is still fenced and cannot reconcile
(`a_settlement_lost_to_an_exhausted_ledger_fences_even_after_a_reopen`). The other
side of the boundary is asserted with no index left at all: the attempt record
fails, the executor and the adapter are never called, the runtime returns `Audit`
and the wire reports `Refused`
(`an_attempt_the_ledger_cannot_record_is_an_audit_failure_and_reaches_no_executor`,
`an_attempt_the_host_cannot_record_is_refused_and_reaches_no_adapter`).

### Reconciliation records, it does not infer

`reconcile_effect(attempt, applied, evidence)` commits what an operator
established from the system that received the effect. It never guesses and never
retries: a runtime able to work out what happened would not have been fenced in
the first place.

The two outcomes are not symmetric in their consequences, and that asymmetry is
the point of recording which one it was. `applied: false` means the at-most-once
budget was never spent, so the key binds no action and a later attempt under it
may proceed, for any action. `applied: true` means it was: from then on a retry of
the action the reconciled attempt recorded is answered from history without a
response — `ResponseNotRetained`, because reconciliation establishes *whether* the
effect applied and never *what* it returned — and any other action under the key
is refused (below).

## Rules worth stating explicitly

- **An at-most-once key is part of the preparation, not of the action.**
  `prepare_execution` gives no such guarantee and `prepare_execution_once` does.
  Only the caller knows whether an identical request means "the same one again"
  or "do it once more", so defaulting either way would be wrong. Two *live*
  attempts under one key are refused. After an attempt under a key has applied,
  the key may be reused only for the same action — the same project and
  principal that attempt recorded, and the same action digest taken at the
  revision and generation it recorded — and any other is refused with
  `KeyBoundToAnotherAction` before anything is recorded. A key reconciled as not
  applied binds nothing. Keys are one namespace per runtime, shared by every
  principal and project, so a key one principal spent is refused to all others.
- **A denied preparation writes no record.** A hard finding, a failed
  verification, an expired permit or a missing capability leaves nothing outside
  the runtime, so a record for it would be a fence with nothing behind it. The
  audit describes attempted effects.
- **The recorded verification level is the one that admitted the action**, not
  the level the grant required. A grant that accepts `FullSemantic` and an action
  verified `Deterministic` are different facts and the stronger one is what
  happened.
- **Settling bypasses the fence it ends, and only that fence.** Ambiguity about
  this runtime's own last ledger append still blocks a settlement, because
  appending onto a history whose last outcome is unknown would build on a
  position the runtime cannot describe.
- **No floor can rise past an open attempt, and no new rule was needed for it.**
  `journal_anchor` already refuses while the runtime is fenced, and
  `export_compacted_snapshot` begins by asking for that anchor. A compacted
  snapshot is what lets a floor rise, so a fenced runtime cannot produce the
  artifact that would discard the record saying an effect may have applied. This
  is asserted rather than assumed.
- **Effect and verification codes are data.** `effect_code` and
  `verification_code` are explicit tables, never `as` casts, for the reason
  `26-cognitive-codebook.md` sets out at length: a discriminant follows
  declaration order, so inserting one variant would renumber every record already
  written. A test pins all ten values.

## The records

Tags 9, 10 and 11, added beside the existing 0–8 which are unchanged. Tag 12, a
semantic record with an attributed origin, came later
(`22-durable-semantic-state.md`).

| Record | Binds |
|---|---|
| `EffectAttempted` | optional at-most-once key, project, principal, target, operation, capability, effect, generation, revision, admitting verification level, SHA-256 of the action |
| `EffectSettled` | the attempt's commit index, the response if retained, the response digest always |
| `EffectReconciled` | the attempt's commit index, whether it applied, the operator's evidence |

An attempt is addressed by **its own commit index**. It needs no separate
identifier: the position a record was committed at is already unique in the
history that ordered it, and inventing a second identity would create two ways to
name one thing.

`action_digest` is domain-separated (`PTREXEC01-ACTION`) so a digest of these
bytes cannot collide with one taken elsewhere in the system, and it covers the
payload, so an audit record commits to the exact action rather than to its shape.

## Executed evidence

16 integration tests in `crates/ptr-runtime/tests/execution_audit.rs` plus 6 unit
tests in `ptr-ledger`. **286 workspace tests pass, up from 264** — 22 added, on
Rust 1.85.0 and on stable.

Positive:

- An applied effect commits its attempt first and its settlement second, with
  every field asserted individually, including the action digest and the
  admitting verification level; the runtime is unfenced afterwards and ordinary
  commits continue.
- A retry under the same key returns the first response **without calling the
  executor again**, while a different key does execute — both directions, because
  a mechanism that never repeats and one that always repeats are equally useless.
- The same holds across a restart: a fresh process with a fresh executor answers
  the retry from replayed history and the executor is never called.
- Reconciliation lifts the fence for both outcomes and commits the evidence.
- An attempt reconciled as not applied lets the same key execute again.

Negative, one per route back in:

- **A reopened runtime is still fenced by an unsettled attempt** — for an executor
  error and for a panic. The reopened runtime names the same attempt, refuses to
  register a session at all (`AmbiguousOutcome`), and refuses to commit.
- A fenced runtime yields no journal anchor and no compacted snapshot.
- A settlement naming no live attempt is refused during replay, including a second
  settlement of an already-settled attempt.
- A second live attempt under one key is refused; the same key after a settlement
  is accepted.
- A retained response that contradicts its digest is refused; one past the bound is
  refused, and one exactly at the bound is accepted, so the limit is the limit.
- A malformed key is refused at preparation and in replayed history.
- Reconciling an attempt that awaits nothing, and evidence containing a control
  character, are each refused.
- Codec level: all ten effect/verification codes pinned, unknown codes refused at
  a byte position located by diffing two records rather than hand-counted, invalid
  presence and boolean bytes refused, a truncated digest refused at every length,
  trailing bytes refused.

## Work that does not finish inside the call

The window above opens and closes around a synchronous adapter call, which means an
effect handed to something that answers later was audited as though it had finished
when the call returned. "The call returned" and "the effect applied" are different
facts, and for detached work an unbounded amount of time separates them.

`dispatch_detached` commits the attempt and does **not** settle it. The runtime is
fenced from the moment the work is handed over until an answer arrives, so a caller
cannot mistake "accepted" for "done": while the window is open, no permit can be
prepared, no journal anchor is produced and no compacted snapshot can be exported.
A call that returned unfenced would be claiming the effect had finished.

### The grant decides, not the caller

A grant is synchronous or detached, fixed when it is issued. A detached grant is
refused on the synchronous path and a synchronous one is refused on the detached
path, and both refusals happen **before** the attempt is committed — a refusal must
leave no record, because a record with nothing behind it is a fence with nothing
behind it. A caller that could choose would be choosing how its own effect is
audited.

Both paths share one admission sequence. Two copies would be two chances to drift on
the ordering that matters.

### Not accepted is not not-applied

An adapter that fails while handing work off may have handed it off. So the attempt
is committed *before* `start` is called, and an error from `start` leaves the fence
standing — the same answer a synchronous executor's error gets, for the same reason.

### An answer from the adapter, and an answer from a person

`settle_detached` records what the adapter reported. `reconcile_effect` records what
a person established from the receiving system. They are different claims and the
ledger keeps them apart, which is why there are two record types rather than one
with a flag.

Settlement is available only for an attempt **this runtime** dispatched. The record
says an effect was attempted, not how it was dispatched, and it is not rebuilt as
"detached" on reopen: a restarted runtime that accepted an adapter answer for work it
never handed out would be inventing that distinction. After a restart the fence
stands and reconciliation is the way forward — the same answer a crash inside a
synchronous effect gets.

Either answer closes the window, and the first one recorded is the one that holds. A
detached attempt a person reconciles is no longer outstanding, and an adapter answer
that arrives afterwards is refused with `NotDetached` before anything is appended,
whichever way the reconciliation went
(`a_reconciled_detached_attempt_is_no_longer_outstanding_and_a_late_answer_is_refused`).
Before this was enforced the attempt stayed listed as outstanding after
reconciliation, and a late answer reached the ledger, which refused it as naming no
live attempt; since settlement failures are reported as `SettlementNotRecorded`,
that would have claimed an applied effect and a fence that neither existed.

### What detached work does not change

The retention boundary is unchanged: at-most-once holds while the attempt's record is
retained. A response over the bound is not retained and its digest still is, so "we
did not keep the response" never becomes "we do not know what happened" — and a later
retry under that key is refused rather than answered with something the effect never
produced.

Nothing here supervises the adapter. There is no timeout, no cancellation and no
retry: a runtime that could decide the work had failed would be deciding something it
cannot observe. An adapter that never answers leaves a fence, and a fence is
information.

## The obligation travels with the floor

At-most-once used to hold only for as long as the attempt's *record* was
retained, and compaction is precisely the operation that stops retaining it. A
runtime restored from a compacted snapshot rebuilt its execution state by
replaying committed records that the raised floor had removed, so it came back
with no memory of which keys had been spent — and a retry under a spent key
executed the effect a second time. That was demonstrated rather than reasoned
about, and `an_at_most_once_key_survives_the_floor_rising_past_its_attempt`
fails on a build without the fix.

So the compacted snapshot carries a third section holding what the execution
layer cannot re-derive once the floor has moved: the spent at-most-once keys with
their outcomes, and the unsettled attempts. It was introduced as `PTREX001` inside
`PTRCS002`; the current layout is `PTREX002`, which also carries what each key is
bound to (the next section). It is the execution section of `PTRCS003` and of
`PTRCS004`, whose lifecycle section may carry the attestation marker
(`24-protected-anchors.md`).

Three things are worth stating exactly, because each is easy to get wrong in the
telling:

- **The unsettled half was already safe, and is carried anyway.** `journal_anchor`
  refuses while the runtime is fenced and `export_compacted_snapshot` begins by
  asking for it, so a fenced runtime cannot produce a snapshot at all and the
  carried map is empty in practice. It is encoded regardless, so the guarantee
  stops depending on that one invariant holding somewhere else. The defect that
  was real, and is closed here, is the **settled** half.
- **The magic moved rather than only the reserved field.** A `PTRCS001` snapshot
  is refused with `PTR_COMPACTED_VERSION`, not read as a snapshot whose
  obligation set happens to be empty — which would restore exactly the runtime
  that forgets every spent key, silently, which is the failure being closed.
  `PTRCS002` is refused the same way since `PTRCS003`, for the reason the next
  section gives.
- **The magic moved again for the attestation marker, but `PTRCS003` is still
  read.** `PTRCS004` changes the lifecycle section only. A build from before
  attributed records would read a `PTRCS004` snapshot without its marker and
  then append records its history's replay refuses, so it has to stop at the
  magic. The other direction is sound: a `PTRCS003` snapshot describes a history
  of tag-8 semantic records only, so reading it as one without the marker is
  true, and its execution section is `PTREX002` either way. That differs from
  the `PTRCS002` refusal, where reading the older section would have lost what a
  key is bound to.
- **Nothing that belongs to a process travels.** Sessions, the admission policy,
  detached dispatch and commit-level `uncertain` are this process's, not
  committed history's. Carrying any of them would let a restored runtime claim
  knowledge it never had — the same reason `detached` is not rebuilt from history
  on an ordinary restart.

## A key names one action

The at-most-once memory was a map from the caller's key to an outcome, and
admission looked the key up and answered from it. The attempt's `action_digest`
was committed in `EffectAttempted` and never compared, so the key named whatever
the next permit under it asked for. Two things followed, both reproduced before
the fix:

- **Another action was answered with the first one's receipt.** Principal `alice`
  applied action A under `invoice-7`; action B, with another digest, under the
  same key got A's response without the executor running and without an attempt
  being recorded. A second principal `bob`, with a grant of his own, got
  `alice`'s receipt the same way.
- **The settling attempt was inferred from bytes.** A detached retry reported the
  first `EffectSettled` whose response digest matched, so of two keys settled with
  equal bytes the second reported the first's attempt, and after a compacted round
  trip — the settlement record gone — it reported `CommitIndex(0)`. A
  `ResponseNotRetained` refusal named the first attempt record under the key,
  which is not the one that settled it once a key has been reconciled as not
  applied and attempted again, and `CommitIndex(0)` after a compacted round trip.

Now every entry carries what its attempt recorded. An unsettled attempt holds an
`ActionIdentity` — project, principal, revision, generation and `action_digest`,
taken from the attempt record — and `settle` moves it, with the attempt's own
index, into the key's outcome when the effect applied:
`Applied { attempt, identity, response }` or
`AppliedWithoutResponse { attempt, identity }`. `NotApplied` carries neither,
because a key reconciled as not applied binds nothing. Project, principal and
digest are all compared because the digest covers the action, payload included,
and not the project or the principal the record names beside it: without the
principal, `bob` repeating `alice`'s action is indistinguishable from `alice`
retrying it.

The digest also covers the revision and generation the action carried, and those
say where the runtime stood rather than what was asked for. Admission requires
every permit to carry the current ones, so a digest compared as recorded would
refuse `alice`'s own retry once an unrelated semantic delta had committed or the
capsule's generation had been superseded, although her effect had applied — and
the wire would report that as `Refused`, inviting a retry under a fresh key. So a
keyed permit is digested at the revision and generation its key's attempt
recorded: a retry carrying the current position is answered, and another target,
operation, capability, input type, payload or effect is still refused.

Admission compares a keyed permit's identity with the key's entry before anything
is appended. A match is answered from the entry: with the retained response, or
refused with `ResponseNotRetained`, and a detached retry and that refusal name the
attempt the entry holds. A mismatch is refused with
`KeyBoundToAnotherAction { attempt }`, which renders as
`PTR_EXECUTION_DENIED: KeyBoundToAnotherAction { .. }` like every other execution
refusal, reaches no executor and writes no record. The live, reconciled and
replayed paths all build the entry in the one place that holds both halves,
`settle`, so a restarted runtime rebuilds the same binding from the log that it
rebuilds the fence from. The execution wire reports the refusal as `Refused` with
the `Runtime` code (`32-execution-wire.md`).

Keys are one namespace per runtime: the at-most-once memory is addressed by the
key alone, not by principal or project. So the binding has two consequences
besides the protection above. A principal refused under a key learns that somebody
else spent it, since under a key nobody had spent the same request would have
applied. And whoever spends a key first makes it unusable for every other
principal and project for as long as the key is remembered. A caller should
therefore choose keys nobody else can guess, and principals should not share a
key space.

The compacted section had to carry the same facts, because the records they come
from are below the floor. `PTREX002` appends project, principal, revision,
generation and action digest to each unsettled attempt, and writes each settled
key's tag first: an applied outcome follows it with the settling attempt's index,
project, principal, revision, generation and action digest (and, for tag 0, the
retained response); `NotApplied` is the tag alone.

A snapshot must not refuse a string a log it compacts could hold. So the project
and principal are written at any length, the empty string included, as a length
and the bytes. No build has bounded either: a session's principal and a grant's
project must be identifiers of any length, and a directly committed
`EffectAttempted` may name any string at all. `PTREX001` carried neither, so a
bound here would have been new, and an earlier revision of this change had one,
4096 bytes like the section's other strings. It made a history with a longer or
empty principal, which compacted on the previous build, fail every export once a
key under it applied, and it made the runtime refuse sessions and peers the
previous build admitted. Both are gone: what an attempt records is recorded as
given, and no string of it makes an export fail. What the section holds in total
is still bounded, and `PTREX002` weighs more per applied key than `PTREX001`
did, so a history near that bound on the previous build can pass it after the
upgrade (under "What this does not close").

A key is still written as 1 to 4096 bytes, as `PTREX001` wrote it, because a key
is chosen by the requester rather than by the host and every later snapshot
carries it again. `prepare_execution_once` refuses a key past `MAX_KEY_BYTES`
(4096) with `InvalidKey`, and `PtrRuntime::commit`, through which admission
appends its attempts, refuses a keyed `EffectAttempted` under one with
`InvalidEffectKey`, before anything is appended, so no new key can leave every
export failing.

Stored history is not held to that bound. Earlier builds held a key only to
being an identifier, of any length (up to 64 KiB from a peer on the execution
wire, and any length through a direct commit), so a log or recovery snapshot
written before may hold a longer one. Every path that rebuilds a runtime from
stored history replays such a record as it is: `PtrRuntime::replay`, and through
it `restore_recovery_snapshot`, `read_recovery_snapshot`,
`restore_durable_snapshot` and `migrate_legacy_log`; `open_durable` and
`open_durable_at`; and the records `restore_compacted` replays above a floor.
Replay still checks that a key is an identifier, which every build has required.
An earlier revision of this change refused such records on replay, to fail at
open rather than at every export. The Codex security review of PR #39 showed why
that is wrong: a peer granted a keyed request on the previous build could have
spent a key of up to 65,536 bytes, and after an upgrade the runtime would refuse
its own log and not start.

Preparation lets a key past the bound through once it has settled here, with
either outcome; while its attempt is unsettled the runtime is fenced and no
session can be registered to prepare under it. The request that spent it is
therefore answered from history as it would have been before: its response, or
`ResponseNotRetained`. Another request under it is refused as bound to another
action, which the previous build, binding no key, did not do. Refusing the
retry with `InvalidKey` instead, as another earlier revision did, would tell the
requester on the wire that nothing was attempted, when its effect had applied,
and invite a retry under a fresh key that applies it twice. Such a key is never
spent again: an attempt under it after it was reconciled as not applied is
refused at commit, with nothing attempted. What remains is what the previous
build did too: once such a key has settled, with either outcome, every
`export_compacted_snapshot` fails with `PTR_COMPACTED_SECTION_LIMIT` (under "What
this does not close").

A `PTRCS002` snapshot, or a `PTREX001` section, is refused with
`PTR_COMPACTED_VERSION` rather than read: a key restored from it has no attempt
and no identity, so this build could only answer any action under it — the
defect — or invent a binding. `PTRCS001` was refused the same way when
`PTRCS002` replaced it. It means a runtime whose log the previous build
compacted, discarding the records below the snapshot's floor, cannot be restored
by this build; it has to be rebuilt from a log or recovery snapshot that still
holds its whole history. No binary in this repository compacts a log, so the
step falls to a host that embeds the runtime and compacts through the library.

Asserted by `crates/ptr-runtime/tests/at_most_once_keys.rs`, each test failing on
the code before the fix:

- another payload under a spent key, and the same action from another principal,
  are refused, reach no executor and write no record, while the request that spent
  the key is still answered and the other principal still executes under a key of
  his own (`a_spent_key_is_refused_for_another_action_rather_than_answered_with_the_first_response`,
  `a_spent_key_is_refused_for_another_principal_rather_than_answered_with_its_receipt`);
  the detached path hands nothing out
  (`a_detached_dispatch_under_a_key_spent_on_another_action_is_refused_and_hands_nothing_out`);
- a key reconciled as applied is bound to the action its attempt recorded
  (`a_key_reconciled_as_applied_is_bound_to_the_action_its_attempt_recorded`), and
  one reconciled as not applied binds nothing until another action applies under
  it (`a_key_reconciled_as_not_applied_binds_no_action`);
- the binding survives an ordinary restart for the action and the principal
  (`a_spent_key_stays_bound_to_its_action_and_principal_across_a_restart`), and for
  the project, from an attempt written into the log directly because no live path
  records one action under two projects — after the restart and again after a
  compacted round trip
  (`a_key_spent_under_another_project_is_refused_after_a_restart_and_after_compaction`);
- a restored runtime keeps each key's action, principal and attempt, including a
  `ResponseNotRetained` refusal naming an attempt whose records are gone
  (`a_restored_runtime_keeps_each_spent_key_bound_to_its_action_principal_and_attempt`);
- a retry carrying the current revision and generation is answered from history,
  reaching no executor and writing no record, after an unrelated delta moved the
  revision and after the capsule's generation was superseded — live, after a
  restart and after a compacted round trip — while another payload and another
  principal at the same position are still refused
  (`a_retry_is_answered_after_the_revision_and_generation_move_and_another_request_is_still_refused`);
- the admission policy admits a peer under a 64 KiB principal, and a session
  registered under one spends a key, after which the runtime still exports,
  restores and answers the retry, while another principal is refused under the
  key (`a_principal_of_any_length_spends_a_key_and_a_snapshot_carries_it`);
- a keyed attempt committed directly under an empty principal, a 64 KiB one, or
  one with surrounding whitespace or a control character is recorded, opens with
  `PtrRuntime::replay`, `open_durable`, `open_durable_at` and
  `restore_recovery_snapshot` however its key ended, and once reconciled as
  applied exports and restores with the key bound to it, refusing a session's
  request. A log an earlier build could have written with an empty or 64 KiB
  principal reopens whether the attempt was settled, reconciled either way or
  left unsettled: it exports unless it is fenced, its key refuses alice's
  request once applied and binds nothing once reconciled as not applied, and a
  settled one still executes under a key nobody spent
  (`a_keyed_attempt_under_any_principal_is_committed_and_a_snapshot_carries_it`);
- a key past `MAX_KEY_BYTES`, and one of the execution wire's full 64 KiB, is
  refused by `prepare_execution_once` with `InvalidKey` and at commit with
  `InvalidEffectKey`, reaching no executor. A history holding one opens on the
  same four paths however its key ended, and so does a log an earlier build
  could have written, whatever ended the attempt. There the request that spent
  the key is answered from history, with its response once settled and with
  `ResponseNotRetained` once reconciled as applied, and runs nothing; once
  reconciled as not applied, a new attempt under the key is refused at commit
  with nothing attempted; and export fails with `PTR_COMPACTED_SECTION_LIMIT`
  once the key has settled either way. The longest key allowed opens on each
  path, and is spent, exported, restored and answered
  (`a_key_longer_than_a_snapshot_carries_is_refused_before_it_is_spent_and_a_logged_one_is_still_answered`);
- a keyed attempt committed directly under an empty or 64 KiB project is
  recorded, opens on the same four paths however its key ended, and once
  reconciled as applied exports and restores with the key bound to it; a log an
  earlier build could have written with such a project reopens and exports as
  the principal's does
  (`a_keyed_attempt_under_any_project_is_committed_and_a_snapshot_carries_it`);
- a resealed snapshot whose recorded principal has a length past the bytes that
  remain is refused with `PTR_COMPACTED_LENGTH`, and one whose recorded project
  is not UTF-8 with `PTR_COMPACTED_NONCANONICAL_SECTION`, while the snapshot as
  exported restores
  (`a_snapshot_whose_recorded_project_or_principal_is_malformed_is_refused`);
- a detached retry names its own key's attempt when another key settled equal bytes,
  and after a compacted round trip; a refusal names the attempt that settled the
  key, not the first attempt under it
  (`a_detached_retry_names_the_attempt_that_settled_its_own_key_when_another_key_settled_equal_bytes`,
  `a_detached_retry_after_a_compacted_round_trip_names_the_attempt_that_settled_its_key`,
  `a_refused_retry_names_the_attempt_that_settled_its_key_not_the_first_attempt_under_it`);
- `PTRCS002` and a `PTREX001` section are each refused by version after resealing
  (`a_snapshot_whose_keys_bind_no_action_is_refused_by_version`).

## What this does not close

- **At-most-once holds across compaction, but not across a lost snapshot.** The
  obligation now travels in the snapshot (below); what remains is the ordinary
  anchor-retention obligation of `24-protected-anchors.md` — a host that loses
  the snapshot or its anchor has lost the memory, and no format prevents that.
- **A settlement lost to a full ledger fences for good.** When the settlement of
  an applied effect fails because the ledger has no index or record left, the
  attempt stays unsettled across a reopen, and reconciling it needs another
  append, which the ledger refuses the same way; the runtime produces no journal
  anchor or compacted snapshot while fenced, so nothing in band frees room. The
  runtime keeps refusing rather than forgetting that an effect applied, which is
  the safe direction, but lifting that fence needs a ledger with room, which is
  outside the runtime.
- **At-most-once, not exactly-once.** An ambiguous outcome stays ambiguous until
  someone supplies evidence. Nothing here makes an applied effect reversible or
  an unknown one knowable.
- **The audit trusts the executor's report.** A `Ok` that did not actually apply
  the effect is recorded as applied. Adapters remain responsible for their own
  truthfulness, the same boundary P0.2 records for dependency declarations.
- ~~**No network admission and no project-scoped Pods.**~~ **Closed, and this
  bullet was wrong to still be here.** It said `PodRegistry::resolve` "still
  matches on capability and input type alone"; it has taken a `ProjectId` since
  `c9ae32e` (`crates/ptr-pods/src/lib.rs`), and a Pod registered for one project is
  invisible to another, refused identically to one that does not exist — asserted
  by `a_pod_registered_for_one_project_is_invisible_to_another` and
  `two_projects_may_each_register_the_same_pod_id_without_shadowing`. Sessions are
  admitted from an authenticated peer by `29-peer-admission-and-pod-scope.md`, and
  `a635980` is a host that establishes one from a QUIC connection
  (`32-execution-wire.md`).

  It is worth naming what kind of error this was, because it is the one this
  repository keeps finding: a "what this does not close" section is written once
  and then describes a tree that has moved. Nothing checks a bullet. It is the same
  class as the stale M001 record and the `component.toml` that claimed the opposite
  of its own tests, both recorded in #20 — and it was found by reading these
  sections rather than by any check, which is the honest statement of how far the
  checking goes.
- **What replaces it, and is true:** reconciliation remains unauthenticated (below),
  and `PodRegistry` scopes by a project a Pod *declares* in its own manifest, which
  is a claim by whoever registers it rather than a proof.
- **Reconciliation is not authenticated here.** `reconcile_effect` is a
  privileged host API like the rest of the gateway; who may call it is the
  embedding host's question.
- **A spent key is refused at admission, not at validation.** Validation refuses a
  second *live* attempt under a key, not an attempt under a spent one, so a host
  that commits an `EffectAttempted` record directly under a spent key can have it
  replayed, and the key then holds what that later attempt's settlement
  established — no binding at all if it is reconciled as not applied. No path
  through admission writes such a record.
- **An attempt's target and operation are not bounded.** The materialized
  projection records them as `effect:{index}:target` and
  `effect:{index}:operation`, and the lifecycle section writes every materialized
  value as a string of 1 to 4096 bytes, so an attempt, keyed or not, whose target
  or operation is empty or longer leaves every later export failing, as it did
  before this change. Admission requires both to be identifiers and the target
  to be a committed capsule, but bounds neither's length, so a grant with an
  operation past 4096 bytes reaches it. The key bound above is the only bound
  validation puts on a string an attempt names; strings other records
  materialize are no more bounded by validation than these two.
- **The settled keys are not bounded in total.** `ExecutionObligations::encode`
  writes every settled key into the one execution section, which holds at most
  `MAX_SECTION_BYTES` (8 MiB) and `MAX_SECTION_ITEMS` (65,536) entries, and an
  applied key carries its retained response of up to `MAX_RETAINED_RESPONSE`
  (1 MiB) and, since `PTREX002`, its attempt's project and principal at whatever
  length they were recorded. Nothing removes a settled key: settling only adds
  one, and a restart or a compacted round trip rebuilds them all. So eight keyed
  effects each settled with a full-size response, or 65,537 distinct settled
  keys, make every later `export_compacted_snapshot` fail with
  `PTR_COMPACTED_SECTION_LIMIT` for the rest of the runtime's life although the
  fence has cleared and no string is out of bounds, through admission alone, and
  so do fewer keys spent under a principal or project the host made long. This
  predates the key bound, which does not reach it; closing it needs a retention
  rule for settled keys or a layout that does not put them all in one section.
  `PTREX002` makes it reachable sooner, and on upgrade: it writes each applied key
  with 64 bytes more than `PTREX001` (the settling attempt, revision, generation,
  action digest and two lengths) plus its project and principal, which
  `PTREX001` did not carry. A history whose execution section the previous build
  could still write can therefore fail every export after the upgrade. For
  example, 20,000 applied keys of 36 bytes with 330-byte responses under project
  `p` and principal `alice` take 7.64 MB in `PTREX001` and 9.04 MB in
  `PTREX002`, past the 8 MiB bound, and so do about 130 applied keys under a
  64 KiB principal. (The lifecycle section's three materialized entries per
  attempt already stop every export past about 21,845 attempts, in both
  layouts.)
- **A log written before the key bound can still stop compaction.** The bound
  holds new keys only. A log an earlier build wrote may hold a keyed attempt
  whose key is past `MAX_KEY_BYTES`: up to the execution wire's 64 KiB from a
  peer, any length through a direct commit. It opens, keeps its key bound and
  answers the request that spent it, because refusing it would keep the runtime
  from starting after an upgrade. But once such a key has settled, with either
  outcome, every `export_compacted_snapshot` fails with
  `PTR_COMPACTED_SECTION_LIMIT`, as it did on the previous build, whose section
  wrote keys with the same bound. The recovery snapshot and the log itself are
  unaffected. Compacting such a history needs the same retention rule or new
  layout as the item above, or a migration that rewrites the key.
- **A key is bound to the principal its attempt recorded**, which for a peer is the
  admission policy's principal at the time. A policy that renames a peer's
  principal therefore makes that peer's earlier keys refuse its retries.
- **Keys are not scoped per principal or project.** They are one namespace per
  runtime, so a refusal under a key tells a principal that another spent it, and
  a principal that spends a key first blocks it for every other (above). Scoping
  the memory by principal and project would change what the compacted section
  means and needs a new layout; choosing keys nobody else can guess is the
  caller's obligation until then.
- **Denied preparations are not audited.** That is deliberate, not missing, but it
  does mean this history answers "which effects were attempted", not "what was
  refused".
