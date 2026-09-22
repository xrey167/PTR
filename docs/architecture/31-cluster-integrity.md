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

## The wire: `ptr-cluster`

`ptr-ledger` must not depend on transport and `ptr-net` must not depend on the log,
so a third component knows both and exists precisely so that neither of them has
to. `ptr-cluster` carries batches of raft messages between authenticated endpoints
on `ALPN_RAFT`.

### The sender is the connection, not the payload

This is the rule the crate exists for. Every raft message carries a `from` field,
and a peer can put any number in it. A receiver that believes that field accepts a
vote or an append attributed to a member that never sent it — which is a forged
message, not a corrupted one, and no checksum detects it.

So a frame is stepped only when the id its messages claim is the id bound to the
public key the connection **actually authenticated**, and the same check is applied
to a reply: the peer answering must be the peer that was asked. A message addressed
to a different member is refused rather than helpfully forwarded.

Authenticating a peer is still not authorizing an action, which is the distinction
`29-peer-admission-and-pod-scope.md` draws for execution. Here it means: the
connection establishes *which member* is speaking, and raft's own rules — terms,
log matching, quorums — decide whether what it says has any effect.

### One batch, one frame

A step can produce several messages, and a transport that takes them one at a time
turns one decision into several round trips whose interleaving nobody asked for. So
a frame carries a batch: magic, an explicit layout version, a count, and each
message length-prefixed.

The count is a **claim**, not an instruction. A frame that says it carries a million
messages is refused by the bound rather than by allocating for them, and the test
asserts that specifically. A frame is bounded, a batch's length is bounded, and a
frame that disagrees with itself — trailing bytes, a truncated message, an
undecodable one — is refused whole. A caller that stepped the first message and then
refused the second would have applied part of a frame it rejected.

A refused frame still gets an answer: an empty batch. The refusal is the receiver's
business, and a peer left hanging on a connection is the receiver's problem too.
What the answer deliberately does not carry is *why*, because a diagnosis handed to
an unauthenticated peer is an oracle.

### What the network tests establish, and what they deliberately do not

Six tests. A two-member group elects a leader over a real connection and both
members end with the same history — and a majority of two needs the peer's vote, so
the wire is necessary rather than incidental. A peer holding the key admitted for
one member cannot send messages claiming to be another. A peer that was never
admitted is refused by key. A message for another member is refused. A damaged frame
arriving on a real connection is refused before it reaches the node — the transport
half of the corruption requirement, whose deterministic half lives in `ptr-ledger`.
And a refused frame is answered rather than dropped.

Nothing here tests a partition, a leader change or a delayed message. Those are
protocol properties, and a network test of them would be a test of the test's own
timing; they are established deterministically in `ptr-ledger`, where no socket is
involved. Two suites, each proving the thing it can actually prove.

### Not a daemon

There is no background loop, no timer and no retry policy in this crate. A caller
drives `serve_once` and `dispatch`. That is a deliberate omission rather than an
unfinished one: a loop in the wrong place makes scheduling implicit, and "when do we
give up on a member that does not answer" is a policy decision, not a detail. It is
also what keeps these tests free of sleeps.

## Snapshot transfer

A member that falls behind past the leader's log floor cannot be caught up by
replay, because the entries it needs are gone. The leader sends its state instead.

The payload is **opaque to every layer that carries it**. `ptr-ledger` records
bytes at a committed position and hands arriving bytes back; `ptr-cluster` frames
them and checks who sent them. Neither has an opinion about what is inside, which
is what lets a PTR deployment put a **PTRCS002 compacted snapshot** there rather
than a second artifact invented for raft. The `ptr-cluster` test carries a real one:
exported by a real runtime, recorded by the leader, delivered byte for byte to a
restarted member, and then restored by a runtime from exactly those bytes.

Three things about it are deliberate.

**A leader that recorded no snapshot cannot invent one.** `FileRaftStorage` answers
`SnapshotTemporarilyUnavailable` rather than fabricating a payload at the requested
index, so a member behind a floor with no recorded state simply stays behind. Being
stuck is the honest outcome; a payload that did not match its claimed position would
be sent to a follower as truth.

**A member must account for what arrived before it applies anything else.** The
events below a snapshot stop being the member's own to report — they are described by
a payload only the application can read — so applying further entries is refused
until the application says how many events the payload covers. Carrying on as though
it were empty would number the next event 1 and disagree with every other member
about what that index means. The test asserts the refusal, and that the event after a
snapshot continues the ledger's numbering.

**The anchor can now be retained independently of the sender.** PTRCS002 is
verified against a trusted anchor the host retains *outside* the artifact, for the
reason `24-protected-anchors.md` gives: a digest read back out of the file it
describes proves nothing. A snapshot arriving over this transport used to come with
its anchor from the authenticated leader and nothing else, so what it ruled out was
a *stranger* sending a consistent pair, not the leader itself.

`take_installed_snapshot_matching` takes a `RetainedSnapshotAnchor` — a position and
a digest the host got from somewhere the sender does not control — and refuses
anything else. Two details decide whether it is worth anything:

- **The digest compared against is computed here, from the bytes that arrived.** A
  digest read out of the payload would say only that the payload agrees with
  itself, which is the mistake `24-protected-anchors.md` exists to name.
- **A mismatch leaves the snapshot installed.** The member stays blocked rather
  than continuing as though nothing had arrived — continuing would renumber every
  event after the snapshot, so a refusal that unblocked the member would trade one
  failure for a worse one.

What this does **not** do is make raft Byzantine fault tolerant, and it is not an
attempt to. A member that accepts a snapshot from the leader its own protocol
elected is the design. `take_installed_snapshot` remains for the deployment that
has no second channel, and says in its own documentation what it is trusting.

## Membership is negotiated, not just recorded

A configuration used to be something each member recorded and recovered, and
nothing could change it: `raft_storage.rs` encodes and decodes `EntryConfChange`
— it must, since it stores whatever raft hands it — but nothing anywhere proposed
one, and `apply_committed` skipped the entry type outright. So the storage could
durably record a membership change that no code could initiate, and a committed
change would have had no effect on raft's own view of the group.

Both halves are implemented. `propose_membership` refuses from a follower rather
than forwarding, because a follower proposing would be asking peers to accept a
configuration nobody agreed. Applying happens where every other committed entry
is applied, and writes the resulting `ConfState` through `set_conf_state`, so the
change is durable at the same moment it takes effect rather than at the next
restart.

`voters()` answers in ascending order. Raft's own `ConfState` keeps voters in
arrival order — a group of four reports `[3, 1, 4, 2]` — and a membership answer
whose order depends on how the changes arrived is one callers compare wrongly.

The test that matters is the restart: a four-member removal is committed, the
cluster is dropped, and a member is reopened from its own disk with the *original*
three passed to `open`. It reports the new configuration, because the recorded one
is the authority — which is the property that makes a membership change more than
an in-memory opinion.

## Durable fencing, as distinct from protocol fencing

Raft's rules already stop a deposed leader from committing: it cannot gather a
majority in a superseded term. What is specific to durability is the moment a
deposed leader **restarts**: the entry it appended alone is on its disk, and a
restart is exactly when it could come back.

The test restarts it from its own files while the group has moved on. It recovers
what was committed rather than what it wrote alone, learns the higher term, steps
down, and the stale tail is truncated — on disk, not merely in memory, which is what
the truncate-then-write rule above buys.

None of that is a *lock*. A local advisory file lock makes one writer per file on
one machine and says nothing about a second machine.

Be exact about what raft gives, because this document said "the protocol" and its
own open section said the opposite fourteen lines later. What raft gives is a
**safety argument among members that run the protocol correctly**: a deposed leader
may append, it cannot gather a majority, and the entry only it held is overwritten.
That is why `a_deposed_leader_cannot_commit_and_what_it_wrote_alone_does_not_survive`
passes (`crates/ptr-ledger/tests/raft_cluster.rs`). It is not a *fence*, which is
what excludes a writer that does not run the protocol correctly.

### The fencing token

`FileRaftStorage` records the highest term it has ever accepted a write under, in
the state file beside term, vote and commit, and refuses any write presenting a
lower one — `PTR_RAFT_FENCED`, naming both terms. That is what excludes the two
writers raft's argument does not cover: one partitioned from its peers but not from
the storage they share, and one resumed from a stale image.

**Every path that touches the shared files is fenced**, and they are enumerated
here because enumerating them in a commit message is how one gets missed:
`persist_state`, `append`, `apply_snapshot` and `record_snapshot`. The last two
matter most and were the easiest to overlook — both destroy history before they
persist state, so a writer checked only at `persist_state` has already truncated
the log and overwritten the snapshot payload by the time it is refused.

**The token is read from disk on every write, never from memory.** This is the
whole mechanism, and reading the cached copy instead would fence nothing: a stale
writer's in-memory token is its own stale copy, so it would happily agree with
itself. `crates/ptr-ledger/tests/raft_fence.rs` opens *two* handles on one
directory — the shared-storage case in miniature — and a mutation that consults
memory rather than disk fails exactly the three tests that use two writers.

Two consequences, stated rather than discovered:

- **A read precedes every write.** That is what a fence costs. It is a small file,
  and the alternative is a token that cannot see the writer it exists to exclude.
- **It is a fence, not a lock.** A writer that has caught up to the recorded term
  proceeds, which is what a legitimate restart does; a second handle is not refused
  for being second. `the_same_writer_proceeds_once_it_is_no_longer_behind` is the
  control for exactly that, because a storage that refused every second writer
  would pass every other test in the file.

The state file format moves to `PTRRST02`. An older file is refused by its magic
rather than read as one whose token happens to be zero — a state file read as
unfenced is the failure this exists to prevent.

## What this does not close

- **The harness is not a deployment.** The deterministic tests step nodes in one
  process, with delivery in FIFO order. Real reordering, concurrent stepping and
  partial writes under load are outside it, and the network tests run two members
  on localhost.
- **Nothing runs the loop.** `ptr-cluster` provides the pieces and no deployment
  drives them: there is no accept/tick service, no policy for a member that stops
  answering, and no backpressure toward one that answers slowly.
- **Membership changes one voter at a time, and a joiner must be admitted
  first.** `propose_membership` adds or removes a single voter; a batch API is
  not offered, because raft's single-step change is safe only while consecutive
  configurations overlap in a majority and changing two at once can produce two
  disjoint majorities. A member being added still has to be `admit`ted so its
  peers have an address for it — the configuration says who votes, not where they
  are, and nothing here discovers that.
- **Every exchange opens a connection.** Correct and wasteful. Session reuse is
  listed as missing rather than quietly assumed.
- **A fence bounds writers, not readers, and only on storage it can see.**
  `PTR_RAFT_FENCED` stops a writer at a superseded term from changing these files.
  It says nothing about a writer reaching some *other* copy of the state — a
  replica, a restored backup, a second mount that is not the same file — because
  nothing there ever sees the token. The older statement follows, narrowed:
- **Local file locks fence nothing across nodes, and neither does consensus alone.**
  `FileLedger` takes an advisory lock, which makes one writer per file on one
  machine and says nothing about a second. What raft adds is a safety argument
  among correct members — a deposed leader cannot commit, tested — and not a fence
  against a writer that is not one. **Durable leadership fencing is a protocol
  property and it is not implemented**, which is the last clause of #15's Gate 2
  sentence and is carried forward in `docs/OPEN_ITEMS_PLAN_20260921.md` rather than
  closed here. This document used to assert the opposite in its own body, fourteen
  lines above this section; that is corrected.
- **Compaction is not integrated with the ledger's retention floor.** This is
  raft's own log, distinct from the committed ledger in
  `25-erasure-and-retention.md`, and the two floors are still independent.
- **A deployment with no second channel still trusts the leader.** The mechanism
  for retaining an anchor independently exists; supplying one is the host's, and a
  host that calls `take_installed_snapshot` is back to trusting the sender. Nothing
  here can create a second channel for a deployment that has none.
- **No performance or availability claim.** Every write flushes, which is a
  correctness choice with an obvious cost, unmeasured here.
