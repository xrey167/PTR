# P0.7 — Checkpoint and neural-state admission

Baseline: `160e6a76dc5ff51fe79a9a7f4ba94092edb5de26`.
Owner: `ptr-runtime::neural`.

Issue #15's *Checkpoint and neural-state admission* gate asks for four things:
bind opaque model/KV/checkpoint state to semantic dependencies, lifecycle
generations, provenance and codebook version; keep old cached or neural state from
regaining admission after a restart, an edit or a revocation; preserve the exact
state/restart/inference-equivalence requirement; and do not confuse any of this with
the semantic history P0.2 and P0.3 already reconstruct.

All four are implemented here. What is *not* closed is stated at the end, and the
most important item there is not a missing feature but a boundary: the whole
mechanism is only as good as the producer's declared input set.

## Why this cannot be a validation problem

Every other durable artifact in this repository can be checked against itself. A
journal record has a digest, a snapshot has a section layout, a semantic delta has a
canonical encoding. Neural state has none of that. It is opaque — no check here can
look inside a tensor and decide whether it is still true — and it is expensive,
which is exactly why a system wants to keep it across the events that should have
invalidated it.

So the question is never "is this state valid?" It is "**may** this state
participate?", and the answer is computed entirely from facts that live outside the
state:

| Bound to | Why |
|---|---|
| anchored journal position | which history it was computed on, not just how far along |
| per-input value digests | which semantic values it actually consumed |
| lifecycle generations | which capsule/constraint/procedure versions were live |
| codebook version **and** assignment fingerprint | what its integer identities mean |
| provenance | who produced it, inside the digested binding |

`StateBinding` holds exactly those. It is not a credential: its fields are public
and anyone can construct one. That is harmless, because nothing in it grants
anything — `PtrRuntime::admission` re-derives every field from committed state — and
the one field that is hard to forge is hard to forge because forging it requires
possessing the real history.

## The three properties that carry the design

### Admission is decided at every use, never remembered

A cache that validated on insertion hands out state that was admissible an hour ago,
and revocation is precisely the event that arrives afterwards. `NeuralStateCache`
therefore exposes no accessor that returns a payload: `admit(&runtime, key)` is the
only way in, and it re-decides every time. `evict_denied` exists for housekeeping
and is documented as *not* the safety boundary — a host that never called it would
still never be handed a denied payload.

The payload is unreachable except through `AdmittedState`, which only an admission
decision produces. `NeuralState::seal` does serialize it, because retaining a state
is not using it; the boundary is the shape of the API, not a sandbox.

### History is identified by its anchor, not by a number

Two different histories reach `Revision(7)`, and a restart that replays a different
journal reaches the same counters. P0.2 already recorded this trap for semantic
snapshots ("matching revision numbers cannot admit foreign/pre-restart snapshots");
neural state has the same exposure and more incentive to ignore it.

A binding carries the `LogAnchor` of the position it was computed at, and admission
requires this runtime's own history to carry that exact digest at that exact index.
Because the digest chains every earlier record, agreement at one index is agreement
about the whole prefix. A test builds two runtimes that agree on commit count,
revision, live generations *and* the bound input's value, and the state from one is
still refused by the other.

The recorded revision is checked only when the bound position **is** the runtime's
current position. At an earlier position the revision that held then is not
recoverable without replaying, so the anchored prefix and the per-input digests carry
the weight there. This is a deliberate narrowing, not an oversight: see
*What this does not close*.

### Failure to verify is a denial, not an error

Every refusal is one `Denial` type, including the ones that mean "this runtime cannot
check that claim". Splitting *unverifiable* into a separate error channel is how an
unverifiable state gets used anyway, because one `if` treats a non-denial as a pass.

`UnverifiablePosition { bound, from, through }` is a single verdict for both ends of
the range — above what the runtime has reached, or below a floor whose records it no
longer holds — with the range saying which side it fell off. One meaning, one
variant, one construction site.

## Rules worth stating explicitly

- **Generations compare by equality, not by "at least".** A state computed under
  generation 1 describes generation 1; generation 2 is a different thing, not a newer
  view of the same thing. Supersession therefore denies.
- **Revocation is evaluated first.** A revocation tombstone must deny a state even
  when everything else about it is unverifiable, so the check cannot sit behind one
  that fails for another reason. Tombstones are monotone and replayed, so this
  survives restart by construction.
- **An unrelated commit does not deny.** Binding by declared input digests rather
  than by the global revision is what makes the mechanism useful at all; a test
  asserts both directions, because a cache that denies on every commit and a cache
  that never denies are equally worthless.
- **A removal denies.** Removing a semantic value commits a removal record; the value
  stops being current, and a state that consumed it stops being admissible. A
  derivation the semantic layer evicts on change reaches the same outcome through
  `MissingInput`.
- **The codebook needs both version and fingerprint.** A version alone is not enough:
  tables can be edited without a version bump, and a checkpoint that keeps loading
  under a changed assignment is exactly the silent remapping the codebook exists to
  prevent. `CodebookAssignmentChanged` is a distinct verdict from
  `UnknownCodebookVersion` so an operator can tell "wrong generation of checkpoint"
  from "this build's tables moved".
- **A fenced runtime admits nothing and binds nothing.** An ambiguous execution
  outcome already stops a runtime from producing a trusted journal anchor; admission
  inherits that, so it cannot vouch for a position in a history it cannot vouch for.
- **A compacted in-memory restore admits nothing at all.** It retains committed
  semantic and lifecycle state but no chain base, so every content check would pass
  and the position still cannot be verified. `UnanchoredHistory` is the honest
  answer, and it is asserted rather than assumed.
- **`bind_state` proves its own output.** It builds the binding from committed state
  and then runs it through the same admission rules a later use will apply, so there
  is one rule set instead of a producer's copy and a consumer's copy that drift.

## The artifact

`PTRNEU01`: a 32-byte header (magic, binding length, payload length, reserved),
a canonical binding section (`PTRNB001`), the opaque payload, then a 32-byte
SHA-256 over everything above. `NeuralAnchor` — journal anchor, codebook version,
digest — is retained outside the artifact, in the same trust direction as
`ProtectedAnchor` and `CompactedAnchor`: a digest read back out of the file it
describes proves nothing.

The anchor's first two fields are redundant with the binding, and that redundancy is
the point: a catalog entry states what it believes it is holding, so a substituted
artifact is caught by the catalog and not only by its own digest.

Canonical means strictly ascending keys for the two maps, because they are sets and a
reordering is not a different value. Provenance is written in declaration order,
because it is a list whose order is part of what was recorded. A presence byte other
than 0 or 1 is refused, since a third value would be a second encoding of the same
meaning.

Decoding is not admission. A state can be perfectly intact, match its retained anchor
exactly, and still be inadmissible — which is the normal case after a revocation, and
one of the tests below asserts exactly that sequence.

## Executed evidence

15 integration tests in `crates/ptr-runtime/tests/neural_admission.rs` plus 6 unit
tests in `ptr-runtime::neural`. **255 workspace tests pass, up from 234** — 21 added.

Positive:

- An admitted state reproduces the **identical** inference event sequence after a
  process-level restart, with the binding and the opaque payload byte-for-byte equal,
  through a deterministic backend seeded only from the admitted payload.
- The state stays admissible after an unrelated commit and after the runtime has
  moved past the position it was bound at, because the prefix it names still agrees.
- A binding round-trips through its canonical encoding; a sealed state round-trips
  and carries its payload.

Negative, one per route back in:

- Revocation after sealing: the artifact reopens intact against its anchor and
  admission is refused; rebinding at the current position is refused too, so the
  state cannot be laundered into a fresh binding.
- Edited input, removed input, superseded generation, unknown target.
- A foreign history with identical commit count, revision, live generations and bound
  input value.
- A position the runtime cannot check; a compacted restore that can check nothing.
- A recorded revision contradicting its position.
- An unknown codebook version; a changed assignment under a known version.
- A fenced runtime: admission, admit and bind all refuse.
- Every single-bit mutation of every byte of a sealed artifact.
- Wrong outer magic, wrong binding magic, a set reserved field, a mismatched length
  field, one byte removed, one byte inserted, and an artifact too short to hold a
  header — each **after resealing the digest**, so each is a real constraint rather
  than something the digest happens to cover.
- A reordered inputs section and a third presence-byte value, both refused as
  noncanonical.

## What this does not close

- **An undeclared dependency is invisible.** Every check above works from the
  producer's declared input set. A state that secretly consumed a value it did not
  declare will be admitted after that value changes. This is the same boundary P0.2
  recorded for semantic recomputation — producers remain responsible for complete
  dependency declarations — and no check on this side can substitute for it.
- **The recorded revision is only checked at the current position.** A binding whose
  recorded revision is wrong *and* whose position is below the runtime's current one
  passes that particular field. The position itself is still verified against the
  chain, so this cannot admit a foreign history; it can only carry a misleading
  diagnostic.
- **No real model, no real checkpoint.** The payload is opaque bytes and the test
  backend is a deterministic hash, not an LLM. Nothing here is evidence about model
  quality, inference cost, or that caching neural state helps. The equivalence test
  proves that identical admitted bytes produce identical output, which is a property
  of the plumbing.
- **Burn A0 does not use any of this yet.** Nothing in the training stack records a
  `CodebookVersion`, a `StateBinding` or a fingerprint; that half sits outside the
  production workspace on a different toolchain with its own CI job, and remains open
  together with the Burn half of the codebook gate.
- **No durable catalog.** `NeuralAnchor` has to be retained somewhere the artifact
  cannot reach, and this crate does not provide that store — the same deployment
  obligation the other anchors carry.
- **Payloads are held in memory**, bounded at 64 MiB. Streaming or memory-mapping a
  large checkpoint is not implemented; this layer is about identity, not storage.
- **Two guards are unexercised and listed rather than claimed.**
  `UnverifiablePosition`'s lower bound needs a runtime backed by a log above a
  compaction floor, which this crate cannot yet construct; a compacted restore
  currently denies everything with `UnanchoredHistory`, which *is* tested.
  `UncheckableInput` covers a committed value whose digest cannot be recomputed,
  which the semantic layer's own limits make unreachable — it exists because denying
  is the only safe answer if that ever changes.
- **Nothing is erased.** Denying admission does not destroy a state. Getting rid of
  the bytes is the retention problem, and its boundaries are in
  `25-erasure-and-retention.md`; model-derived state is explicitly out of reach
  there.
