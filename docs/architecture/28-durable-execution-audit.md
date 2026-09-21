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

### Reconciliation records, it does not infer

`reconcile_effect(attempt, applied, evidence)` commits what an operator
established from the system that received the effect. It never guesses and never
retries: a runtime able to work out what happened would not have been fenced in
the first place.

The two outcomes are not symmetric in their consequences, and that asymmetry is
the point of recording which one it was. `applied: false` means the at-most-once
budget was never spent, so a later attempt under the same key may proceed.
`applied: true` means it was, and the key is answered from history from then on —
without a response, because reconciliation establishes *whether* the effect
applied and never *what* it returned.

## Rules worth stating explicitly

- **An at-most-once key is part of the preparation, not of the action.**
  `prepare_execution` gives no such guarantee and `prepare_execution_once` does.
  Only the caller knows whether an identical request means "the same one again"
  or "do it once more", so defaulting either way would be wrong. Two *live*
  attempts under one key are refused; reusing a key after its attempt settled is
  ordinary.
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

Tags 9, 10 and 11, added beside the existing 0–8 which are unchanged.

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

## What this does not close

- **At-most-once holds only while the attempt's record is retained.** A
  compaction floor that rises past a settled attempt discards the memory of its
  key, and a retry after that will execute again. Unsettled attempts are
  protected by the fence; settled ones are a retention obligation, in the same
  direction as the anchor-retention obligations in `24-protected-anchors.md`.
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
- **Denied preparations are not audited.** That is deliberate, not missing, but it
  does mean this history answers "which effects were attempted", not "what was
  refused".
