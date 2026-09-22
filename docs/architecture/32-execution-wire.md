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

This is the same shape as `ptr-cluster` (`31-cluster-integrity.md`) and it was chosen
for the same reason: a composition crate adds no dependency edge to any existing
component. The alternative — `ptr-runtime` depending on `ptr-net` — would allow a
strictly stronger property, and it is recorded here as an open decision rather than
taken:

> A peer identity could be made **unforgeable at the type level**: a type
> constructible only from an authenticated connection, the way `VerifiedDispatch` is
> constructible only inside the runtime. Then no code anywhere could pass a peer
> identity the transport had not proved, because there would be no way to build one.
> What this crate has instead is a single place where that mistake is possible and is
> not made, which is weaker: a second composition written elsewhere could get it
> wrong, and nothing would stop it compiling.

The stronger version changes a dependency graph this repository tracks deliberately,
so it is the owner's decision. Everything below holds either way.

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
| `AppliedWithoutResponse` | it applied and the bytes are not coming: either not retained, or too large for a frame |
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

**One thing an admitted peer can learn**, stated rather than glossed: because the
request id is spent only after admission (step 5), a peer that replays its own frame
gets "replayed" if it is admitted and "refused" if it is not. That is a fact about
its own key. The alternative is spending window entries before admission, which
makes a table anyone can fill, and availability is what a host owes.

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

## A receipt, and what a receipt is not

A receipt names its own author and is bound to a digest of the exact bytes that
arrived. A requester checks three things, in this order: the author against the peer
the connection authenticated — first, because a receipt from somebody else is not
worth reading further; then the request id; then the digest. Without the digest, a
host, or anything between, could answer an expensive request with a cheap request's
receipt and both would be well formed.

The digest is taken over the received bytes rather than over a decoded request, so
even a request this build cannot parse gets a receipt its sender can check.

**A receipt is not signed.** Inside that exchange the connection vouches for its
author. Once the bytes are stored, forwarded or shown to somebody, nothing does:
anyone can write those bytes. This crate offers no way to verify a receipt out of
band, on purpose — the alternative is an artifact that looks like evidence and is
not. A receipt is the answer to your own request, not a certificate.

## What this does not close

- **No signed receipts, so no third-party evidence.** As above. Making one would
  need a signing key and a statement about what it attests to, which is a larger
  decision than a wire format.
- **No accept loop, no timer, no retry.** A caller drives `serve_once` and
  `request`. The same deliberate omission as `ptr-cluster`: a loop in the wrong place
  makes scheduling implicit, and "when do we give up" is a policy decision.
- **Detached work cannot be requested over this wire.** A detached grant is audited
  by an attempt that stays unsettled until the adapter reports back
  (`28-durable-execution-audit.md`), and there is no channel here for that report. So
  it is refused — by the runtime, before the attempt is committed, which is why the
  refusal leaves the runtime unfenced. An outcome the requester can never be told is
  not an outcome. A settlement channel is a second protocol.
- **Nothing here reconciles.** An `Uncertain` outcome is reported and then it is the
  host operator's, through `reconcile_effect`. The requester's only handle on it is
  its own at-most-once key, deliberately: a remote peer must not be able to declare
  what happened to an effect.
- **No position in the host's ledger crosses the wire.** How far that ledger has got
  is a fact about every other principal's activity too. A requester gets the outcome
  of its own request and not a measure of the host's traffic.
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
