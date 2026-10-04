# ADR-0021 — The six owner decisions of issue #23 (D1–D6)

## Status
Accepted. Each decision is recorded in the contract it governs, as issue #23's
acceptance requires; this record gathers them, with the reasoning that chose each
answer over the alternative, so the choice is not reconstructed from six documents.

| | Question | Answer | Governing contract | Pull request |
|---|---|---|---|---|
| D1 | May `ptr-runtime` depend on `ptr-net`? | **Yes**, on its transport-free base | `32-execution-wire.md`, `29`, `33` | #102 |
| D2 | Should a receipt be signed? | **Yes**, receipts and Pod answers, with the endpoint key | `32-execution-wire.md`, `33-pod-wire.md` | #111 |
| D3 | Should the codebook artifact be authenticated? | **No** | `26-cognitive-codebook.md` | #101 |
| D4 | Where does the accept loop live? | **Mechanism in `ptr-net`, policy in the host (`ptrd`)** | `15-deployment-topologies.md`, `31`, `32`, `33` | #112 |
| D5 | Journaling the admission policy: which design? | **Membership is history; authority is the host's** | `29-peer-admission-and-pod-scope.md`, `COMPONENT_CONTRACTS.md`, `31` | #106 |
| D6 | Should reconciliation be authenticated? | **Yes, as a capability; it never crosses a wire** | `28-durable-execution-audit.md`, `32` | #109 |

## Decisions

**D1 — yes.** The base of `ptr-net` depends only on `ptr-types` and `ptr-protocol`,
both of which the runtime already depended on, so the edge adds no crate to any
build. What it adds is that `admit_peer` can demand an `AuthenticatedPeer`, a type
with no public constructor that only the transport builds from the key the QUIC
handshake verified. Before, the property held in one place where the mistake was
possible and not made; after, the mistake does not compile. Tests reach the type
through a `test-support` feature that `scripts/check_repo.py` refuses outside
`[dev-dependencies]`.

**D2 — yes.** The endpoint's identity is already an Ed25519 public key, so signing
needs no new key infrastructure, and a rotated key is a new identity. What the
signature attests is stated narrowly and in the contract: that runtime R recorded
outcome O for request X bound to digest D — not that the effect happened, nothing
about another principal, not when, and not that R is still trusted. The objection
recorded before ("an artifact that looks like evidence and is not") is answered by
that statement rather than by withholding the signature: the artifact now *is*
evidence of exactly what it says.

**D3 — no.** The authority over the codebook is the kernel compiled into
`ptr-types`, and every loader that matters compares against it instead of believing
the file. A signature would need a key with no trust anchor this repository could
name, and would turn a mismatch that is caught today into a signed mismatch that
looks authoritative. An artifact that crosses a trust boundary does so inside a
release, under the release's own provenance.

**D4 — mechanism in `ptr-net`, policy in the host.** The objection to a loop was
that it makes scheduling implicit and "when do we give up" is policy. Both are kept
explicit: `ServePolicy` has no `Default`, every bound is a required `[wire]` field
in `ptr-config`, over the limit a connection is refused rather than queued, and
nothing retries. `ptrd` composes the wires next to HTTP with one shutdown.
Addresses are a deployment fact, rebuilt from `[[wire.peers]]` at every start, and
deliberately not journaled (C8: no discovery).

**D5 — membership is history; authority is the host's.** The old table held
behaviour (a grant owns a verifier and an executor), so "rebuild it from history"
could not mean what it said. It is split along that line: `PeerAdmitted` and
`PeerWithdrawn` are ledger records replayed on every open and carried across
compaction; `GrantSets` are installed by the host at every start and resolved by
name at use, and a name the host does not supply refuses the peer rather than
granting by default. A session is bound to the admission record it was issued
under, so a re-admission revives nothing. Because the ledger codec is also the raft
proposal payload, the decision brought the codec versioning `COMPONENT_CONTRACTS.md`
requires: a version table, refusal of an unknown tag by name, and a committed codec
ceiling that a leader raises only when every voter is reported to read it. The Pod
access policy follows the same design.

**D6 — yes, as a capability.** `reconcile_effect` takes a
`ReconciliationAuthority`: no public constructor, no `Clone`, handed out once per
runtime. `ExecutionHost::bind` seals it away unless the host took it first, so code
that reaches a runtime through the wire host cannot declare what happened to an
effect. C4's settlement channel is a different path for a different party: the peer
a detached attempt was bound to may report that it applied, and only that.

## Consequences
- `LedgerEvent` is at codec version 2 (tags 27–30, and the Pod scope records), the
  compacted snapshot is `PTRCS005`/`PTRLC003`, and the raft state file is
  `PTRRST03`. Older builds stop at each by name; a log holding no version-2 record is
  byte-identical and still readable by them.
- Every commit retires every prepared permit, so a withdrawal — now a commit —
  makes another peer's in-flight permit stale; that peer prepares again.
- `RECOMMENDATIONS_20260924.md` O4 advised deferring all six. They were taken
  instead, because each either removed a way to write a wrong program (D1, D6) or
  stated a property the code already almost had (D2–D5); none widened what the
  system claims about multi-host deployment, which the contracts still do not make.
