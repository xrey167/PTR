# 31 — Cluster integrity: the part of Raft that lives on disk

Status: **partial.** This document describes durable Raft state, which is one
requirement of issue #20's cluster-integrity gate. Multi-node composition,
leadership fencing across nodes and snapshot transfer are **not** closed by it,
and the last section says so in detail rather than leaving it to be inferred.

## What a node must not forget

Raft's safety argument is not only about messages. Three pieces of state have to
survive the process, and each one has a specific failure when it does not:

| Forgotten | What becomes possible |
|---|---|
| the term it has seen | joining a stale term and accepting an old leader |
| its vote in that term | voting twice in one term, so two leaders are elected |
| committed entries | acknowledging a history the node no longer has |

`raft::storage::MemStorage` holds all three in memory. A process that dies has
therefore agreed to things it can no longer account for, and nothing in the group
can tell: its peers remember the promise it does not.

So `ptr_ledger::raft_storage::FileRaftStorage` is that state on disk.

## Two files, and the rules the format cannot enforce

`state` holds the hard state (term, vote, commit), the configuration and the
position of the retained snapshot. It is rewritten whole and atomically — a new
file, flushed, renamed over the old, with the directory flushed after. A
half-written state file is worse than an old one, because it describes a term or a
vote that was never held.

`log` holds the entries, each as one record: magic, index, term, an explicit
entry-type code, the context and data payloads, and a SHA-256 digest over the
whole record. `snapshot` holds the retained snapshot payload, when there is one.

Two rules matter and neither is expressible in the file layout, so the code holds
them and the tests check them:

1. **Entries and hard state are flushed before anything that depends on them
   leaves the node.** `append` and `set_hard_state` flush before returning, so the
   ordering is simply the order the calls are made in. With one member nothing
   leaves; the ordering is the contract a group relies on, and it is not something
   to introduce later.
2. **A record is only ever replaced by truncation.** A conflicting append from a
   new leader shortens the file to the offset the new entries start at and then
   writes, so what is on disk is always a prefix of one history rather than two
   spliced together. A torn write can lose the tail; it cannot corrupt the prefix.

### Why the records are not hash-chained

Everything else in this repository that appends chains its records —
`23-persistence-integrity.md` for the ledger, `24-protected-anchors.md` for the
anchor — so that a *removed* record is detectable. A Raft log is different: losing
its tail to truncation is legal and routine, so a chain would only ever describe
the prefix.

What a chain would have protected is the middle, and that is protected directly:
indexes must be consecutive from the snapshot position onward, so a record removed
or duplicated inside the log is refused on open. The per-record digest covers the
other half, a record whose bytes changed. The tests separate the two faults
deliberately — a bit flipped in a payload is refused by the digest, a bit flipped
in a length field is refused earlier while reading — because one message for every
kind of damage tells an operator nothing.

### Entry types are coded, never cast

`entry_type_code` is an explicit table, for the reason
`26-cognitive-codebook.md` gives about codes in general: a protobuf enum's numbering
outlives nothing, and a code written into a file outlives the declaration that
produced it. A renumbering would make every record already on disk decode as a
different kind of entry than it was written as.

## Restarting is not replaying twice

A reopened node replays its committed entries and then tells raft what it has
already applied. Telling raft `applied: 0` instead makes it re-deliver the whole
committed prefix as newly committed, and the recovered history is applied a second
time — a restart turns two events into four. That is what
`a_restarted_node_restores_exactly_the_committed_state` catches, and it caught it:
the test failed before the fix.

Entries above the recorded commit index are deliberately excluded from the replay.
They are replicated, not decided, and treating them as history is how a node
acknowledges something the group never agreed.

## Snapshots: this store does not invent one

`MemStorage::snapshot` fabricates a snapshot at the requested index, which is
sound there because it *is* the state machine. A durable store is not. When the
retained snapshot is older than what raft asks for, `FileRaftStorage` answers
`SnapshotTemporarilyUnavailable` — "not yet", which is true — rather than
producing a payload whose contents do not match its claimed position and which
would be sent to a follower as truth.

`record_snapshot` is the other direction: the application hands in a snapshot it
produced, at an index it has committed, and the log below that index is then
dropped. Compaction rewrites the log file, because a prefix cannot be removed from
a file in place, so its cost is proportional to what is *kept* — the opposite of
what compaction is for. That is acceptable because compaction is rare, and it is
written down rather than discovered later.

## A group, driven message by message

`RaftNode` is one member that hands its outbound messages back instead of dropping
them. The single-node harness could throw them away because a one-member group has
nobody to send to; everything a group does — voting, replicating, deposing a leader
— is in those messages.

It deliberately knows nothing about transport. Messages come out and go in, and
whether that happens through a socket, an ALPN on `ptr-net` or a test's own queue
is not its business. That separation is what makes a partition testable without a
timer: **nothing here is driven by wall-clock time**, so a test decides exactly
which message arrives and which does not, and a failure means the code is wrong
rather than the machine busy.

Ticks are a count, not a duration, for the same reason. `heartbeat_tick` is three,
so a test that ticks once and expects contact is testing its own arithmetic — which
is a mistake this repository's tests made once and now has a helper against.

`SingleNodeRaftConsensus` is now a thin wrapper over the same type. Two copies of
"persist entries, flush hard state, apply committed, advance" would be two chances
to get that order wrong.

### The order messages leave in

Raft distinguishes `messages` from `persisted_messages`, and the distinction is not
a preference: the first may go out before the write, the second only after it.
Sending a vote or an append the sender has not yet flushed is how a crash becomes a
promise nobody kept. `RaftNode::drain` collects them in that order.

### What the group tests establish

Eight tests in `crates/ptr-ledger/tests/raft_cluster.rs`, all deterministic:

| Property | What would otherwise pass unnoticed |
|---|---|
| exactly one leader after an election, with the followers in its term | two leaders, or a follower left in an older term |
| a majority commits while a minority decides nothing | a minority that commits locally |
| a member that knows no leader refuses a write outright | a proposal buffered until a leadership that may never come |
| a partitioned follower's forwarded proposal never resurfaces | a queued write the group never ordered |
| a deposed leader appends but cannot commit, and what it wrote alone is overwritten | a resurrected entry from a superseded leader |
| one total order across a leader change, indexes exactly `1..n` | an index assigned twice |
| duplicated and stale messages change nothing | an old append re-applied |
| a restarted member restores exactly the leader's committed state | a node that comes back with a different history |

The minority case is worth stating precisely, because the mechanism is not what it
first looks like. A follower does **not** refuse a proposal: raft forwards it to the
leader it last knew. So the property is not "propose returns an error" but "the
write is not ordered and is not held either" — the message is addressed to a member
it cannot reach, and nothing replays it when the partition heals. The test asserts
that after healing, too.

A message that does not decode is dropped rather than partially applied: a message
is a claim about a term and a log position, and half of one claims nothing.

## What this does not close

- **There is no transport.** `ALPN_RAFT` still carries no traffic, and the raft
  backends and `ptr-net` still have no code path between them. The group is
  composed by a test's queue, which proves the protocol behaviour and proves
  nothing about a network: no framing, no authenticated sender, no bound on what a
  peer may send, and no evidence that a real connection carries these messages at
  all. A composition over `ptr-net` is the next package, and it needs a home —
  `ptr-ledger` must not depend on transport and `ptr-net` must not depend on the
  log.
- **The harness is not a deployment.** It steps nodes in one process and one
  thread, with delivery in FIFO order. Real reordering, concurrent stepping and
  partial writes under load are outside it.
- **Local file locks fence nothing across nodes.** `FileLedger` takes an advisory
  lock, which makes one writer per file on one machine. It says nothing about a
  second machine, and a deposed leader on another host is not excluded by it.
  Durable leadership fencing is a protocol property and it is not implemented.
- **There is no snapshot transfer.** A snapshot can be recorded and read back on
  one node. Sending one to a follower the leader has compacted past, reusing
  PTRCS001 rather than inventing a second artifact, is open.
- **Compaction is not integrated with the ledger's retention floor.** This is
  raft's own log, distinct from the committed ledger in
  `25-erasure-and-retention.md`, and the two floors are not yet related.
- **No performance or availability claim.** Every write flushes, which is a
  correctness choice with an obvious cost, unmeasured here.
