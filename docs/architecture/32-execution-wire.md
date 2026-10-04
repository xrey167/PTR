# 32 — The execution wire: authenticating a peer is not authorizing an action

Status: **implemented for the request/receipt exchange**, and this document says in
its last two sections exactly what that exchange does not establish. It closes the
wire-protocol requirement of issue #20's Gate 1 and nothing beyond it.

The sentence in the title is the whole design. A QUIC connection proves which key is
at the other end. It does not say what that key may ask for, who it counts as inside
this system, or whether the action it is asking for is one anybody granted. Those are
three separate questions with three separate answers, and a protocol that conflates
any two of them has a hole in it.

## Where the layer lives, and why it is a third crate

`ptr-runtime` holds execution authority and must not know about transport.
`ptr-net` holds transport and must not know what an action is. Something has to know
both, and `crates/ptr-execwire` is that something so that neither of those two
becomes it.

This is the same shape as `ptr-cluster` (`31-cluster-integrity.md`): the
composition lives here, so the runtime never learns what a connection is.

### Peer identity is unforgeable at the type level (D1 in issue #23: yes)

`ptr-runtime` depends on `ptr-net` — on its **base**, which has no transport and
depends only on `ptr-types` and `ptr-protocol`, both of which the runtime already
depended on. The edge adds nothing to the build and one property to the system:

`admit_peer` takes an `AuthenticatedPeer`, and `AuthenticatedPeer` has **no public
constructor**. The only code that builds one is `ptr-net`'s transport, from the key
the QUIC handshake verified (`Connection::remote_id`), the way `VerifiedDispatch` is
constructible only inside the runtime. So no code anywhere can pass a peer identity
the transport did not prove — not this crate, not `ptr-podwire`, not a composition
written later — because a `NodeId` read out of a payload does not compile where an
`AuthenticatedPeer` is required. What this crate had before was one place where
that mistake was possible and was not made; it is now unwritable.

Held by two `compile_fail` doctests on `AuthenticatedPeer` (a struct literal; a
conversion from a bare id) and one on `admit_peer` (a bare `NodeId`), the last with
a compiling control taking an `AuthenticatedPeer`. Tests without a network use
`AuthenticatedPeer::assume_authenticated`, which exists only with `ptr-net`'s
`test-support` feature; `scripts/check_repo.py` refuses any manifest that enables
that feature outside `[dev-dependencies]`
(`test_a_normal_dependency_may_not`, with `test_a_dev_dependency_may_enable_it` as
the control), so no production build of any package here can reach it.

Authenticating a peer is still not authorizing an action: the type proves who is on
the other end of the connection, and every table that decides what that peer may do
is still the host's policy.

## Its own ALPN

The exchange runs on `ALPN_EXEC` (`ptr-exec/1`), not on `ALPN_PODWIRE`. The pod wire
is Pod access; this one asks for effects. Two protocols sharing one ALPN is how a
request meant for one gets parsed by the other, and the parse that succeeds by
accident is the dangerous one.

That separation is no longer hypothetical: `ALPN_PODWIRE` now carries a protocol of
its own (`33-pod-wire.md`). The two are deliberately not variants of one format —
their magics differ, their digest domains differ, and a test on each side asserts
that the other's frame is refused by the first eight bytes.

## The order of the protocol, which is an order and not a set

A server accepts one connection and answers it. Each step below can only be taken
because the ones above it were:

1. **The sender is the authenticated connection.** A request frame has *no field*
   naming its sender, so there is nothing to forge and nothing to prefer over the
   connection. This is the shape of the format rather than a check performed on it.
2. **The frame must decode.** A refusal here spends nothing — those bytes were never
   this runtime's request.
3. **The request must name this endpoint's key.** `addressed_to` is the one identity a
   request carries, and it is the *receiver's*. Without it, a request that was
   legitimate at one runtime could be relayed to another and be indistinguishable
   from one composed for it.
4. **The peer must be admitted by policy** — as a peer, not as an authority for
   anything in particular. `admit_peer` takes no principal and chooses no grant; both
   come from the policy table the host installed, re-derived on every request
   (`29-peer-admission-and-pod-scope.md`).
5. **Only now is the request id spent**, so every admitted peer's request is treated
   alike whatever the runtime then decides. See the replay window below.
6. **The runtime prepares and dispatches.** Everything about what is permitted is
   `21-scoped-execution.md`'s and `28-durable-execution-audit.md`'s; this layer adds
   nothing to it and takes nothing away.
7. **The session is revoked**, because it existed for exactly one request.

One session per request is deliberate. Caching it would keep the grants the policy
returned at admission time, so a policy later replaced with *narrower* grants for the
same peer would not narrow anything until the cache expired. Authority re-derived at
use means re-derived at use.

## Four outcomes, because two would be a lie

| Outcome | What it means |
|---|---|
| `Applied` | the effect applied, and this is what it returned |
| `AppliedWithoutResponse` | it applied and the bytes are not coming: not retained, too large for a frame, or the host could not record its settlement |
| `Uncertain` | it **may** have applied and the receiver does not know |
| `Refused` | nothing was attempted |

`Uncertain` is the one that earns this table. The runtime keeps a fence precisely
because "the adapter did not answer" is not "nothing happened"
(`28-durable-execution-audit.md`). A wire that reported that as a refusal would
invite a retry of an effect that had already applied, throwing the fence away at the
last step — and the fence is the expensive part.

The mapping from the runtime's refusals onto these four is an **exhaustive** match.
A catch-all would quietly report a refusal a later build invents as "nothing was
attempted", and if that new refusal happened to mean the opposite, a requester would
retry something that had already happened. Adding a variant must break the build
instead.

That is not only about variants a later build adds. An effect the adapter applied
whose settlement record the host then could not commit — no ledger index left, a
failed durable write — reached the requester as `Refused`, because the runtime
returned it as the same `Audit` error as an attempt record that failed before
anything happened. The runtime now returns it as `SettlementNotRecorded`
(`28-durable-execution-audit.md`) and the wire reports `AppliedWithoutResponse`:
the effect applied, so a refusal would invite a retry under a fresh key that
applies it again, and the response is not handed on because the host could not
record it. The host stays fenced. After a failed durable write it is reopened,
replays what it wrote and, if the attempt is still unsettled there, its operator
reconciles it; when the ledger has no index left, reconciling fails the same way
and the fence cannot be lifted in band
(`an_effect_whose_settlement_the_host_cannot_record_is_reported_applied_not_refused`).
An attempt record the host cannot commit is the other side: nothing was attempted,
the adapter is never reached, and the requester is told `Refused`
(`an_attempt_the_host_cannot_record_is_refused_and_reaches_no_adapter`).

## One refusal code for every reason of authority

A refused receipt carries a code, and every authority refusal maps to the same one:
a peer the host never admitted, an action outside the grant, a project the grant does
not cover, an expired permit, a revoked generation, a verification that failed, a
fenced runtime. A requester that could tell those apart would have an oracle for the
host's policy — the same reason a cross-project Pod request is refused *identically*
to a Pod that does not exist.

The other three codes — malformed, misaddressed, replayed — are not about authority
and tell a requester nothing it did not already know: how it framed its own bytes,
which key it addressed, which request id it chose.

The full reason still exists; it stays on the host, in `Serviced::refused`, for an
operator. A diagnosis is a thing you give the person running the system, not the
person asking it for something.

**Two things an admitted peer can learn**, stated rather than glossed. First,
because the request id is spent only after admission (step 5), a peer that replays
its own frame gets "replayed" if it is admitted and "refused" if it is not. That is
a fact about its own key. The alternative is spending window entries before
admission, which makes a table anyone can fill, and availability is what a host
owes. Second, a refusal under an at-most-once key can tell it that another
principal spent that key, which is a fact about somebody else's activity; the next
section says how, and what a requester does about it.

## Two mechanisms against repetition, and neither stands in for the other

**The replay window** is bounded, in memory and per authenticated peer. It refuses a
frame sent twice, cheaply, before the runtime is asked anything. It does **not**
survive a restart, it does **not** extend past `MAX_REPLAY_WINDOW` ids for one peer,
and beyond `MAX_TRACKED_PEERS` admitted peers the least recently used peer's window
is dropped rather than a request being refused. So an id that has left the window can
be spent again. All of that is what a cheap check is allowed to be.

**The at-most-once key** is the runtime's, recorded in the ledger, and it is what
makes a *retry* apply once and return the original answer. Its guarantee lasts as
long as the attempt's record is retained, which is a retention obligation
(`25-erasure-and-retention.md`), not something this layer can promise.

Therefore: **retrying an intent means a new `request_id` with the same `once_key`.**
That is the only combination that says "this one again" rather than "do it once
more", and it is why the two fields are separate.

"This one again" means the same action from the same principal. The runtime binds a
spent key to the project and principal of the attempt that spent it and to that
attempt's action digest, compared at the revision and generation the attempt
recorded — so a retry carrying the host's current revision and generation is still
answered after either has moved. A request under that key for anything else —
another payload, or the same action from a peer admitted as another principal — is
`Refused` with the `Runtime` code: nothing was attempted for it, and answering it
with the receipt of the request that spent the key would tell it that its own effect
had applied.

Keys are one namespace per host runtime, shared by every principal and project, and
that has two consequences a requester has to know. A peer that sends an in-grant
request under a key another principal spent is refused, where under a key nobody had
spent the same request would have applied — so it can learn that a key it guessed
was spent by somebody else, at the cost of spending that key when it was not. And
whichever principal spends a key first makes it refused to every other principal and
project for as long as the host remembers it. So a requester should choose keys
nobody else can guess, such as random 128-bit values, and should not share a key
space with another principal.

## A receipt, and what a receipt is not

A receipt names its own author and is bound to a digest of the exact bytes that
arrived. A requester checks four things, in this order: the author against the peer
the connection authenticated — first, because a receipt from somebody else is not
worth reading further; then the signature, against that same authenticated key; then
the request id; then the digest. Without the digest, a host, or anything between,
could answer an expensive request with a cheap request's receipt and both would be
well formed.

The digest is taken over the received bytes rather than over a decoded request, so
even a request this build cannot parse gets a receipt its sender can check.

### A receipt is signed (D2 in issue #23: yes)

This section used to say *a receipt is not signed*, and that this crate offered no
way to verify one out of band rather than producing an artifact that looks like
evidence and is not. D2 is decided **yes**, and the reason it costs nothing new is
that the key already exists: an iroh endpoint id *is* an Ed25519 public key, and the
QUIC handshake proves possession of its secret half. The host signs every receipt —
refusals included — with that key, so a signature is attributable to exactly the
identity a connection to the host authenticates, and there is no second key to issue,
rotate or revoke. **Rotation is a new identity**: a receipt signed by an old key
verifies under the old key and under nothing else.

- **What is signed.** `ptr-exec-receipt/v2\0` followed by every byte of the receipt
  frame before the signature: magic, layout version, `responder`, `request_id`,
  `request_digest`, outcome code and the outcome's own fields. The frame is the
  message, so there is no second encoding of the fields to drift from the first. The
  receipt layout is `RECEIPT_FORMAT_V2`; requests keep `FORMAT_V1`, because nothing
  in a request changes. A receipt in the old unsigned layout is refused by name
  (`PTR_EXECW_UNSIGNED_RECEIPT`), not as an unknown version — and a V1 label on a
  signed frame is refused the same way, so a stripped signature cannot pass as an old
  peer (`an_unsigned_receipt_from_an_earlier_build_is_refused_by_version`).
- **What it attests:** that the runtime holding key *R* recorded outcome *O* for
  request id *X*, bound to request digest *D* of the bytes it received. Every word is
  about *R*'s own statement.
- **What it does not attest:** that the effect happened — `Applied` is the runtime's
  record of what its adapter reported, now attributable and no truer; anything about
  another principal — not who sent the request (a request names no sender), not which
  principal the host audited it under, not anything about another runtime; when it
  was made, or that it is the only receipt for *X* — nothing in it is a timestamp or
  a ledger position, and the plan's audit position is deliberately absent because a
  receipt carries none (see below); and that *R* is still trusted, which is the
  verifier's policy.
- **How it is checked.** `verify_receipt(bytes, &NodeId)` needs no transport: it
  refuses a frame that is not a signed receipt, then one naming another author, then
  a bad signature, and is the only way this crate reads a receipt out of bytes. A
  stored receipt verifies offline, and every field edited after signing, every bit
  flipped, a signature by another key and a receipt re-addressed to another responder
  are each refused, with the untouched receipt as the control
  (`a_stored_receipt_verifies_offline_and_a_change_to_any_field_is_refused`,
  `a_receipt_signed_by_another_key_is_refused`,
  `a_receipt_readdressed_to_another_responder_is_refused`). Over a real connection
  the client refuses a receipt from the authenticated peer whose signature is not that
  peer's (`a_receipt_with_a_bad_signature_from_an_otherwise_honest_endpoint_is_refused`),
  and a receipt from a real exchange verifies after the host has closed
  (`a_receipt_from_a_real_exchange_verifies_offline_against_the_host_alone`).
- **The key never leaves `ptr-net`.** A composition gets a `NodeSigner` from
  `IrohTransport::signer()`, which signs only under a `SigningDomain` — a `ptr-`
  prefixed, NUL-terminated tag checked at compile time, so the key is never an oracle
  for bytes a caller chose in full — and has no accessor for the secret.

## What this does not close

- **A signed receipt is evidence of a statement, not of an effect.** As above: it
  makes the host's record attributable to the host's key and says nothing more. Whether
  a receipt from a peer since withdrawn still counts is the verifier's policy, and
  nothing here records a key's admission history for it to consult.
- **No accept loop, no timer, no retry.** A caller drives `serve_once` and
  `request`. The same deliberate omission as `ptr-cluster`: a loop in the wrong place
  makes scheduling implicit, and "when do we give up" is a policy decision.
- **Detached work cannot be requested over this wire.** A detached grant is audited
  by an attempt that stays unsettled until the adapter reports back
  (`28-durable-execution-audit.md`), and there is no channel here for that report. So
  it is refused — by the runtime, before the attempt is committed, which is why the
  refusal leaves the runtime unfenced. An outcome the requester can never be told is
  not an outcome. A settlement channel is a second protocol.
- **Nothing here reconciles, and nothing here can.** An `Uncertain` outcome is
  reported and then it is the host operator's, through `reconcile_effect`, which
  needs the runtime's `ReconciliationAuthority`; `ExecutionHost::bind` seals that
  away unless the host took it first (D6 in issue #23,
  `28-durable-execution-audit.md`). The requester's only handle on it is its own
  at-most-once key, deliberately: a remote peer must not be able to declare what
  happened to an effect.
- **No position in the host's ledger crosses the wire.** How far that ledger has got
  is a fact about every other principal's activity too. A requester gets the outcome
  of its own request and not a measure of the host's traffic — with the one
  exception stated above: a refusal under an at-most-once key can say that another
  principal spent that key.
- **One connection per request**, which is correct and wasteful — the same open item
  `31-cluster-integrity.md` records.
- **No discovery.** Which address belongs to an id nobody told you is a mechanism
  choice with its own trust question, and it is the owner's.

  **Address authority is no longer missing**, and the sentence that used to stand
  here — *a wrong address gets a refusal rather than a wrong execution only because
  step 3 exists* — is no longer true. `request` takes a `ptr_net::PeerAddress`, which
  only a `PeerBook` the deployment installed can produce, and a wrong address fails
  on the authenticated key before step 3 is reached. Step 3 is now the fourth layer
  rather than the only one; `34-address-authority.md` has the three beneath it.
- **No claim about latency, throughput or behaviour under load.** The tests establish
  protocol and identity properties on localhost.

## Non-goals

Byzantine peers; confidentiality beyond what the transport gives; authorization
delegation, where a peer acts on behalf of a third party; any statement about the
trustworthiness of what an adapter does with an action it was given.
