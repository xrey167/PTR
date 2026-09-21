# ptr-cluster — Raft over authenticated transport

> **Role:** Carries raft messages between authenticated PTR endpoints on `ALPN_RAFT`, composing `ptr-ledger`'s consensus with `ptr-net`'s transport.
> **Maturity:** prototype; the protocol properties are proven deterministically in `ptr-ledger` and this crate proves that a real connection carries them.

<!-- PTR:STATUS:BEGIN -->
## Current implementation status

> **Generated section.** Source of truth: [`component.toml`](component.toml) plus code-derived metrics from `src/`. Run `python3 scripts/update_component_docs.py --write` after editing implementation metadata. Do not hand-edit inside this block.

**Maturity:** `prototype`  
**Last reviewed:** 2026-09-21  
**Code footprint:** 3 Rust source files · 721 nonblank source lines · 1 integration-test files · 6 `#[test]` markers

### Implemented now

- ClusterMember carries raft messages between authenticated endpoints on ALPN_RAFT, composing ptr-ledger's RaftNode with ptr-net's transport so neither of them has to know about the other
- The sender of a frame is the authenticated connection and never the message's own from field: a peer whose key is admitted as one member cannot step messages attributed to another, and a message addressed to a different member is refused
- A batch frame format with an explicit layout version, a bound on the frame and on the number of messages, and refusals for foreign bytes, an unknown layout, truncation, trailing bytes, an invented count and an undecodable message — each with its own diagnostic code
- A whole frame is checked before any of it is stepped, so a refusal never leaves part of a batch applied
- A refused frame is answered with an empty batch rather than leaving the peer hanging: the refusal is the receiver's business, not a diagnosis to hand out
- A member that cannot be reached is reported rather than raised: a leader that abandoned a proposal because one follower was down could not commit while any member is down, so a transport failure toward one peer is recorded and the round continues while a forged sender or a malformed frame still aborts
- One exchange has a deadline, because how long to wait for a member is this layer's policy and not the peer's: without one a member that stopped answering holds up every round it appears in
- take_installed_snapshot_matching lets a host that retains snapshot anchors out of band refuse a payload the elected leader sent but the host did not expect; take_installed_snapshot remains for deployments with no second channel and documents what it trusts
- propose_membership adds or removes one voter from the leader and drives it to quiescence, and voters() reports the group in ascending order
- Snapshot payloads are carried unread, and the test carries a real PTRCS002 compacted snapshot end to end: it is recorded by the leader, arrives byte for byte at a member the leader cannot replay to, and a runtime restores from exactly those bytes
- No background loop, timer or retry policy: the caller drives accept and dispatch, which keeps scheduling decisions where they can be made deliberately and keeps the tests free of sleeps
- The whole composition is feature-gated, so iroh and raft stay out of every other workspace build exactly as ptr-net and ptr-ledger keep their own backends out

### Missing for the target architecture

- An accept/tick loop that a deployment would run, including when to give up on a member that does not answer and what backpressure applies to one that answers slowly
- Discovery of a joining member's address: propose_membership changes who votes, but a new voter must still be admitted with an address out of band
- The snapshot anchor travels from the authenticated leader rather than being retained independently, which is weaker than an out-of-band anchor: a compromised leader could send a consistent pair, and what this rules out is a stranger doing so
- Session reuse: every exchange opens a connection, which is correct and wasteful
- Any claim about latency, throughput or behaviour under load; the tests establish protocol and identity properties on localhost

### Next milestones

- Retain snapshot anchors independently of the leader that sends them
- Decide where the accept/tick loop belongs before adding one, since a loop in the wrong place makes the scheduling implicit again

### Linked experiments

- None recorded.

### Technology evaluations

- None recorded.

### Decision records

- [ADR-0001-rust-runtime.md](../../research/decisions/ADR-0001-rust-runtime.md)

### Current automated checks

- a two-member group elects a leader over a real ALPN_RAFT connection and both members end with the same history in the same order; a majority of two needs the peer's vote, so the wire is necessary rather than incidental
- a peer holding the key admitted for one member cannot send messages claiming to come from another; the refusal names both ids and the member's state is unchanged
- a peer that was never admitted is refused by key; a message addressed to another member is refused; a damaged frame arriving on a real connection is refused before it reaches the node
- a refused frame still receives an answer, and that answer is an empty batch
- refusing a stranger's frame does not take a serving member off the air: after the refusal an admitted member still reaches it, so an unadmitted peer cannot silence an admitted one by connecting once
- a member that cannot be reached is named in the report while the write is still decided by the majority that answered
- a real PTRCS002 snapshot travels as the payload to a restarted member the leader compacted past, arrives byte for byte, and a runtime restores the state it described from exactly those bytes
- frame codec unit tests: round trip including an empty batch, every prefix refused, foreign bytes, trailing bytes, an unknown layout version, an invented count refused by the bound rather than by allocation, an oversize frame refused before parsing, and one distinct code per refusal

<!-- PTR:STATUS:END -->

## Why this crate exists

`ptr-ledger` must not depend on transport and `ptr-net` must not depend on the log.
Something has to know both, and this is that something — so that neither of them
does.

## The property it adds

Plumbing would be the uninteresting part. The load-bearing rule is that **the
sender is the authenticated connection, never the message's own `from` field.** A
peer can put any id into a raft message; a receiver that believes it will accept a
vote or an append attributed to a member that never sent it. A frame is therefore
stepped only when the id its messages claim is the id bound to the public key the
connection actually authenticated, and a whole frame is checked before any of it is
stepped.

## What it is not

It is not a daemon. There is no background loop, no timer and no retry policy here:
a caller drives `serve_once` and `dispatch`. That keeps the scheduling decisions —
how often to tick, when to give up on a peer — somewhere they can be made
deliberately, and it keeps the tests free of sleeps.

See [`docs/architecture/31-cluster-integrity.md`](../../docs/architecture/31-cluster-integrity.md).
