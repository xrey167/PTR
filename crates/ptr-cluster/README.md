# ptr-cluster — Raft over authenticated transport

> **Role:** Carries raft messages between authenticated PTR endpoints on `ALPN_RAFT`, composing `ptr-ledger`'s consensus with `ptr-net`'s transport.
> **Maturity:** prototype; the protocol properties are proven deterministically in `ptr-ledger` and this crate proves that a real connection carries them.

<!-- PTR:STATUS:BEGIN -->
## Current implementation status

> **Generated section.** Source of truth: [`component.toml`](component.toml) plus code-derived metrics from `src/`. Run `python3 scripts/update_component_docs.py --write` after editing implementation metadata. Do not hand-edit inside this block.

**Maturity:** `prototype`  
**Last reviewed:** 2026-09-21  
**Code footprint:** 3 Rust source files · 591 nonblank source lines · 1 integration-test files · 6 `#[test]` markers

### Implemented now

- ClusterMember carries raft messages between authenticated endpoints on ALPN_RAFT, composing ptr-ledger's RaftNode with ptr-net's transport so neither of them has to know about the other
- The sender of a frame is the authenticated connection and never the message's own from field: a peer whose key is admitted as one member cannot step messages attributed to another, and a message addressed to a different member is refused
- A batch frame format with an explicit layout version, a bound on the frame and on the number of messages, and refusals for foreign bytes, an unknown layout, truncation, trailing bytes, an invented count and an undecodable message — each with its own diagnostic code
- A whole frame is checked before any of it is stepped, so a refusal never leaves part of a batch applied
- A refused frame is answered with an empty batch rather than leaving the peer hanging: the refusal is the receiver's business, not a diagnosis to hand out
- No background loop, timer or retry policy: the caller drives accept and dispatch, which keeps scheduling decisions where they can be made deliberately and keeps the tests free of sleeps
- The whole composition is feature-gated, so iroh and raft stay out of every other workspace build exactly as ptr-net and ptr-ledger keep their own backends out

### Missing for the target architecture

- An accept/tick loop that a deployment would run, including when to give up on a member that does not answer and what backpressure applies to one that answers slowly
- Membership changes over the wire: the configuration is recorded by each member and never negotiated between them
- Snapshot transfer, so a member the leader compacted past can be caught up over this transport rather than by full replay
- Session reuse: every exchange opens a connection, which is correct and wasteful
- Any claim about latency, throughput or behaviour under load; the tests establish protocol and identity properties on localhost

### Next milestones

- Carry PTRCS001 snapshots to a member behind the leader's log floor
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
