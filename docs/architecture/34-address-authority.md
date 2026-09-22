# 34 — Address authority: an address is not an identity

Status: **implemented for address authority; discovery is deliberately not
implemented** and the last section says why.

It closes the second clause of **C8** in issue #23 — *no discovery and no address
authority* — and the first clause only by recording it as the owner's.

Owner: `crates/ptr-net`, with `crates/ptr-execwire` and `crates/ptr-podwire` as its
only consumers.

## What C8 actually said, and the half of it that was already done

> A requester is given an endpoint address by its host out of band. A wrong address
> gets a refusal rather than a wrong execution **only because** step 3 of the
> protocol exists.

Step 3 is the host checking that a request names *its* key. That sentence is a
complaint about a single point of defence, and it was accurate for the two
request/response wires.

It was **not** accurate for the cluster wire, which had address authority already
under a different name. `MemberAddress` pairs a raft id, a key and an address, its
own doc comment says why — *"without it an address is just a hint that anyone could
answer"* — and `exchange` refuses a member with no entry as `NoAddress`. So one of
three protocols had this. The two that dialled a bare `EndpointAddr` did not, and
this document is about them.

## The shape: make the mistake unrepresentable, not checked

`PeerBook::record` takes **the address alone** and derives its key from the id the
address itself carries. There is no `(peer, address)` pair, so there is nothing to
transpose:

- An operator who pastes the wrong address does not thereby point one peer's name at
  another peer's socket. The wrong address lands under **its own** owner's id, and a
  requester asking for the intended peer still does not find it.
- There is no `insert(peer, address)` for a caller to get backwards, so no check is
  needed and none can be forgotten.

`PeerBook::locate` returns a `PeerAddress`, which has **no public constructor** —
that is the device `VerifiedDispatch` uses in the runtime, applied to the question
*where did this address come from*. Both wire clients take a `PeerAddress` and
nothing else, so a bare address does not satisfy the signature and a requester
cannot dial one it was handed by a peer, read out of a payload, or learned from the
network. A `compile_fail` doctest in each client holds that, and each was checked
against its own positive control so that it fails on the type rather than on a path
that did not resolve.

Nothing in `PeerBook` reads bytes. The absence of such a method is the mechanism:
there is no path by which the network can add an entry.

## Why this table replaces and the authority tables refuse

`AdmissionPolicy::admit` and `PodAccessPolicy::admit` refuse a second entry for one
peer. `PeerBook::record` **replaces** it and returns what it replaced. That is a
deliberate difference and it is about what the entry means:

- an **address** is a fact about where a node is, and a node legitimately moves.
  Refusing an update would leave a deployment unable to follow one.
- an **admission grant** is a decision. A decision that changed silently would be
  authority nobody took.

`forget` takes effect on the next lookup, because the address is read from the book
at every lookup rather than remembered.

## Three layers, and none of them load-bearing alone

The gate's complaint was that one check was doing all the work. It is now three,
and each test disables reliance on the others.

1. **The key is the address.** An address cannot be filed under another peer's name.
2. **An unrecorded peer is refused before a connection is attempted**, so the refusal
   costs nothing and reveals nothing.
3. **An address that lies about the right peer fails rather than reaching the liar.**
   This is the one the gate's sentence was really about. Hand the deployment an
   address carrying the *honest* node's id and an impostor's socket — the one thing
   the book cannot detect, since the id it files under is the id it was given. What
   stops it is that the key at the far end is authenticated: this backend pins it in
   the TLS handshake, which the impostor cannot complete without the honest node's
   private key, and `IrohTransport::request` compares the authenticated id against
   the one asked for as a second line for a backend that did not pin.

Step 3 of each wire protocol — the host checking the key a request names — is now
the fourth layer rather than the first.

`an_address_that_lies_about_a_peer_fails_rather_than_reaching_the_liar` asserts the
**property** rather than an error string: the impostor serves nothing, and the same
requester reaching the honest node at its true address does succeed. So the failure
is about the lie and not about the fixture, the ALPN or the requester.

## One check deliberately not added

With a book, a client knows which peer it is dialling, so it *could* refuse a
request whose `addressed_to` names somebody else before sending it.

It does not, and that is a decision rather than an omission. It would buy no
property — the host checks the name it was sent, and must, since a requester is not
trusted about it — and it would mean a well-behaved client could no longer send a
misaddressed request at all, leaving the host's own check reachable only from a raw
transport. A convenience that blinds a test to the real defence is the wrong trade.

## Executed evidence

5 integration tests in `crates/ptr-net/tests/peers.rs`, plus one `compile_fail`
doctest on `PeerAddress` and one in each wire client. Every existing wire test — 22
in `ptr-execwire`, 9 in `ptr-podwire` — now dials through a book, because there is no
other way to dial.

Positive:

- An address recorded is locatable under the id it carries, and the located address
  is the one recorded.
- Re-recording returns the address it replaced and leaves one entry.
- A truthful address from the book reaches its peer, as the control for every
  refusal below.

Negative:

- A peer whose address was never recorded is refused by name, with a stable code.
- A forgotten peer is refused from the next lookup, and forgetting twice reports
  that there was nothing to forget.
- An address pointing the honest id at an impostor's socket fails, and the impostor
  serves nothing.
- A bare `EndpointAddr` does not compile as an argument to either wire client.
- A `PeerAddress` cannot be constructed by a struct literal.

Mutation-checked: a `locate` that falls back to any recorded address, a `forget`
that always reports success, and a `record` that keeps the first address each fail
exactly their own test. Making `PeerAddress`'s fields public fails its doctest, and
restoring `session_id`-style construction in the client doctests was verified
against a compiling control.

## What this does not close

- **Discovery.** Finding the address for an id you were never told is a mechanism
  choice — a DNS zone, a relay's address lookup, a gossip protocol, a static bundle
  — and each brings its own trust question: whose answer do you believe, and what
  happens when two disagree. `iroh` ships address lookup and this build disables it
  (`RelayMode::Disabled`, `clear_ip_transports()`), so turning it on is a decision
  about trusting a third party, not a code change. Recorded as the owner's rather
  than invented, alongside D4 in issue #23. What the layers above give meanwhile: a
  wrong answer from any future discovery mechanism costs a failed request, because
  it cannot produce a working connection for an id whose key it does not hold.
- **A stale address.** The book cannot tell a stale address from a current one; the
  dial fails and nothing here re-resolves. A deployment whose nodes move must update
  the book, which is what `record` replacing rather than refusing is for.
- **The book is in memory.** It is not journaled and does not survive a restart —
  the same sentence `33-pod-wire.md` records for the access policy, and the same
  open question (D5).
- **Nothing here authenticates the *operator*.** Whoever can call `record` decides
  where a peer is dialled. That is the host's own trust boundary and this layer is
  inside it.
- **`ptr-cluster` keeps `MemberAddress`.** Its key is a raft id rather than a
  `NodeId`, a different key space, and migrating it to `PeerBook` would move code
  without adding a property. Stated here so the two are not mistaken for a drift
  nobody noticed.
